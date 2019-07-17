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
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from data_processor import load_iterator
from util import set_logger

# set logger
logger = set_logger(__name__)
# Fix seed
torch.manual_seed(1)

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

    # For test :
    # self = model
    # batch = next(iter(train_data))
    # word_tensor,word_len = batch.text
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


    # for i,batch in enumerate(train_data):
    #     if i == 0:
    #         break
    # score_matrix = model(batch.text[0],batch.text[1],batch.pos[0])  # score_matrix.shape
    # score_matrix = score_matrix.transpose(1,2)
    # i = 0
    # x_i = score_matrix[i];word_len_i=batch.text[1][i];heads_i=batch.head[i]
    def calc_loss(self,score_matrix,word_len,heads):
        loss = 0
        score_matrix = score_matrix.transpose(1,2)
        logger.debug(f"Device of score matrix is {score_matrix.device}")
        for x_i,word_len_i,heads_i in zip(score_matrix,word_len,heads):
            logger.debug(f"Device of x_i is {x_i.device}")
            logger.debug(f"Device of head_i is {heads_i.device}")
            word_len_i = word_len_i.item()
            x_i = x_i[1:word_len_i]  # x_i.shape = (word_len-1,word_len)
            heads_i = heads_i[1:word_len_i].long()  # heads_i
            loss += F.cross_entropy(x_i,heads_i)  # TODO normalize so the loss scale will be same for different sentence length
        return loss / score_matrix.shape[0]  # devide by minibatch size

    def predict(score_matrix):
        score_matrix = score_matrix.transpose(1,2)
        return torch.argmax(score_matrix,dim=2)

if __name__ == '__main__':

    # Load test
    from pathlib import Path
    import json
    from datetime import datetime
    from pathlib import Path
    import torch.optim as optim
    import matplotlib.pyplot as plt

    # Config
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    logger.debug(f"Computation on {device}")

    model_param = {"word_embed_dim"  : 100,
                   "pos_embed_dim"   : 25,
                   "lstm_hidden_dim" :250,
                   "mlp_hidden_dim"  :100,
                   "num_layers"      : 2}
    model_config = {
        "model_param":model_param,
        "word_dropout":True,
        "learning_rate":0.001,
        "epoch_num":100,
        "batch_size":32
    }

    iterator_config = {"batch_size":32,
                       "shuffle":True,
                       "sort_key":lambda x:len(x.text),
                       "sort_within_batch":True,
                       "device":device}

    # Directory for storing result
    result_dir_path = Path("result",str(datetime.now()))
    result_dir_path.mkdir()
    # Store the config in json
    config_path = result_dir_path / "config.json"
    with config_path.open(mode="w") as fp:
        json.dump(model_config, fp)

    # Load iterator
    train_data = load_iterator("train",**iterator_config)
    dev_data   = load_iterator("dev",**iterator_config)
    TEXT = train_data.dataset.fields["text"]
    POS  = train_data.dataset.fields["pos"]
    model_config["model_param"]["vocab_size"] = len(TEXT.vocab.itos)
    model_config["model_param"]["pos_size"]   = len(POS.vocab.itos)

    # Init model
    model = BiLSTM_Parser_simple(**model_config["model_param"])
    model.to(device)

    # Train setting
    optimizer = optim.Adam(model.parameters(),model_config["learning_rate"])
    # optimizer = optim.SGD(model.parameters(),model_config["learning_rate"])
    epoch_num = model_config["epoch_num"]
    batch_size = model_config["batch_size"]

    # Start train
    loss_tracker = []
    for epoch in range(epoch_num):  # epoch = 0
        running_loss = 0
        for i,batch in enumerate(train_data):  # batch = next(iter(train_data))
            word_tensor,word_len = batch.text
            pos_tensor,pos_lengths  = batch.pos
            original_text = [TEXT.vocab.itos[i.item()] for i in word_tensor[0]]
            # logger.debug(f"The original text of the first sample in the minibatch is: {original_text}")
            # logger.debug(f"The average length os this minibatch is {np.average(word_len)}")
            logger.debug(f"The device of word_tensor is {word_tensor.device}")
            score_matrix = model(word_tensor,word_len,pos_tensor)
            loss = model.calc_loss(score_matrix,word_len,batch.head)
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
        # result_path = result_dir_path / f"model_epoch{epoch}.pt"
        # torch.save(model,str(result_path))

# Visualize the loss
fig,ax = plt.subplots()
ax.plot(range(len(loss_tracker)),loss_tracker)
fig.savefig(result_dir_path/"loss_tracker.png")


# acc = []
# for batch in train_data:
#     acc.append(batch.text[1][0].item())
# fig,ax = plt.subplots()
# ax.plot(range(len(acc)),acc)
