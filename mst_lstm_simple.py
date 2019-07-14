"""
For Hydrogen;
%load_ext autoreload
%autoreload 2
"""
from pathlib import Path
from datetime import datetime
import json
from typing import List,Tuple,Union,Optional,Dict
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchsummary import summary
from data_processor import load_iterator
from util import set_logger

# set logger
logger = set_logger(__name__)
# Fix seed
torch.manual_seed(1)
# manage device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
logger.debug(f"Computation on {device}")

class BiLSTM_Parser_simple(nn.Module):

    def __init__(self,vocab_size,pos_size,
                 word_embed_dim,pos_embed_dim,
                 lstm_hidden_dim,mlp_hidden_dim,
                 num_layers):

        super(BiLSTM_Parser_simple,self).__init__()

        # Hidden dimension must be an even number for now
        # This is the TOTAL dimension of the bidirectional hidden layer
        assert lstm_hidden_dim % 2 == 0

        # Hyperparam
        self.vocab_size      = vocab_size
        self.word_embed_dim  = word_embed_dim
        self.pos_embed_dim   = pos_embed_dim
        self.lstm_hidden_dim = lstm_hidden_dim

        # Layers
        self.word_embeds = nn.Embedding(vocab_size, word_embed_dim)
        self.pos_embeds  = nn.Embedding(pos_size,   pos_embed_dim)
        self.lstm = nn.LSTM(input_size   = word_embed_dim+pos_embed_dim,
                            hidden_size  = lstm_hidden_dim // 2,
                            num_layers   = num_layers,
                            bidirectional= True,
                            batch_first=True)
        self.Linear_head = nn.Linear(lstm_hidden_dim,mlp_hidden_dim // 2)
        self.Linear_modif = nn.Linear(lstm_hidden_dim,mlp_hidden_dim // 2)
        self.output_layer = nn.Linear(mlp_hidden_dim,1)  # output layer

    # For test : word_tensor = data[0]; pos_tensor = data[1]
    # self = model
    # batch = next(iter(train_data))
    # word_tensor,word_lengths = batch.text
    # pos_tensor,pos_lengths = batch.pos
    def forward(self,word_tensor,word_lengths,pos_tensor) -> torch.tensor:
        """
        Compute a score matrix where
        (i,j) element is the score of ith word being the head of jth word
        """

        sentence_len = len(word_tensor[0])

        # Word/POS embedding
        word_embeds = self.word_embeds(word_tensor)     # word_embeds.shape = (batch_size,sentence_len,word_embed_dim)
        pos_embeds  = self.pos_embeds(pos_tensor)       # pos_embeds.shape = (batch_size,sentence_len,pos_embed_dim)
        embeds = torch.cat((word_embeds,pos_embeds),2)  # embeds.shape = (batch_size,sentence_len,(word_embed_dim+pos_embed_dim))

        # Bidirectional LSTM
        packed_embeds = pack_padded_sequence(embeds,word_lengths,batch_first=True)
        packed_lstm_out, _ = self.lstm(packed_embeds)
        lstm_out,_ = pad_packed_sequence(packed_lstm_out,batch_first=True)  # lstm_out.shape = (batch_size,sentence_len,lstm_hidden_dim)

        # Compute score of h -> m
        head_features = self.Linear_head(lstm_out)  # head_features.shape(batch_size,sentence_len,mlp_hidden_dim//2)
        modif_features = self.Linear_modif(lstm_out)  # modif_features.shape(batch_size,sentence_len,mlp_hidden_dim//2)
        score_matrix = torch.empty(size=(len(batch),sentence_len,sentence_len))
        for m in range(sentence_len): # m=1
            for h in range(sentence_len): # h=2
                feature_func = torch.cat((head_features[:,h],modif_features[:,m]),dim=1)  # feature_func.shape = (batch_size,mlp_hidden_dim)
                neuron = torch.tanh(feature_func)
                score_matrix[:,h,m] = self.output_layer(neuron).view(len(batch))  # self.output_layer(neuron).view(len(batch)).shape = (batch_size,)

        return score_matrix

    # heads = batch.head
    # batch.dataset.examples[0].text
    # batch.dataset.examples[0].head
    # len(batch.dataset.examples[0].text)
    def calc_loss(self,score_matrix,word_lengths,heads):
        loss_func = nn.CrossEntropyLoss()
        loss = 0
        for x_i,heads_i,word_len in zip(score_matrix,heads,word_lengths): # x_i = score_matrix[0];heads_i=batch.head[0];word_len=word_lengths[0]
            word_len = word_len.item()
            x_i = x_i.transpose(0,1)[1:word_len]  # x_i.shape
            heads_i = heads_i[1:word_len].long()  # len(heads_i)
            loss += loss_func(x_i,heads_i)
        return loss / len(score_matrix[0])

if __name__ == '__main__':

    # Load test
    from pathlib import Path
    import json
    from datetime import datetime
    from pathlib import Path
    import torch.optim as optim
    import matplotlib.pyplot as plt

    model_param = {"word_embed_dim"  : 100,
                   "pos_embed_dim"   : 25,
                   "lstm_hidden_dim" :250,
                   "mlp_hidden_dim"  :100,
                   "num_layers"      : 2}
    config_dict = {
        "model_param":model_param,
        "word_dropout":True,
        "learning_rate":0.001,
        "epoch_num":30,
        "batch_size":32
    }

    # Directory for storing result
    result_dir_path = Path("result",str(datetime.now()))
    result_dir_path.mkdir()
    # Store the config in json
    config_path = result_dir_path / "config.json"
    with config_path.open(mode="w") as fp:
        json.dump(config_dict, fp)

    # Load iterator
    train_data = load_iterator("train")
    dev_data   = load_iterator("dev")
    TEXT = train_data.dataset.fields["text"]
    POS  = train_data.dataset.fields["pos"]
    config_dict["model_param"]["vocab_size"] = len(TEXT.vocab.itos)
    config_dict["model_param"]["pos_size"]   = len(POS.vocab.itos)

    # Init model
    model = BiLSTM_Parser_simple(**config_dict["model_param"])
    model.to(device)
    # print(model)

    # Train setting
    optimizer = optim.Adam(model.parameters(),config_dict["learning_rate"])
    epoch_num = config_dict["epoch_num"]
    batch_size = config_dict["batch_size"]

    # Start train
    loss_tracker = []
    for epoch in range(epoch_num):  # epoch = 0
        running_loss = 0
        for i,batch in enumerate(train_data):
            # stop at the 30th batch
            if i > 30:
                continue
            word_tensor,word_lengths = batch.text
            pos_tensor,pos_lengths  = batch.pos
            score_matrix = model(word_tensor,word_lengths,pos_tensor)
            loss = model.calc_loss(score_matrix,word_lengths,batch.head)
            loss.backward()
            running_loss += loss.item()
            optimizer.step()
            model.zero_grad()
            # Record the scores
            logger.debug(f"Now at {i}th batch in epoch {epoch}")
            logger.debug(f"Current loss is {running_loss}")
            loss_tracker.append(running_loss)
            running_loss = 0

        # Save model
        result_path = result_dir_path / f"model_epoch{epoch}.pt"
        torch.save(model,str(result_path))
