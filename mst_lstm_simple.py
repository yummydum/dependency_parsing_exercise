"""
For Hydrogen;
%load_ext autoreload
%autoreload 2
"""
from typing import List,Tuple,Union,Optional,Dict
import numpy as np
import torch
import torch.nn as nn
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

class BiLSTM_Parser_simple(nn.Module):

    def __init__(self,
                 vocab_size,
                 pos_size,
                 word_embed_dim,
                 pos_embed_dim,
                 lstm_hidden_dim,
                 mlp_hidden_dim,
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
                            bidirectional= True)
        self.Linear_head = nn.Linear(lstm_hidden_dim,mlp_hidden_dim // 2)
        self.Linear_modif = nn.Linear(lstm_hidden_dim,mlp_hidden_dim // 2)
        self.output_layer = nn.Linear(mlp_hidden_dim,1)  # output layer

    # For test : word_tensor = data[0]; pos_tensor = data[1]
    def forward(self,
               word_tensor:torch.LongTensor,
               pos_tensor :torch.LongTensor)  \
               -> List[List[torch.Tensor]]:
        """
        Compute a score matrix where
        (i,j) element is the score of ith word being the head of jth word
        """
        sentence_len = len(word_tensor[0])
        # Word/POS embedding
        word_embeds = self.word_embeds(word_tensor)     # word_embeds.shape = (1,sentence_len,word_embed_dim)
        pos_embeds  = self.pos_embeds(pos_tensor)       # pos_embeds.shape = (1,sentence_len,pos_embed_dim)
        embeds = torch.cat((word_embeds,pos_embeds),2)  # embeds.shape = (1,sentence_len,(word_embed_dim+pos_embed_dim))
        embeds = embeds.view(sentence_len,1,-1)         # embeds.shape = (sentence_len,1,(word_embed_dim+pos_embed_dim))
        # Bidirectional LSTM
        lstm_out, _ = self.lstm(embeds)                 # lstm_out.shape = (sentence_len,1,lstm_hidden_dim)
        lstm_out = lstm_out.view(sentence_len, self.lstm_hidden_dim)
        # Compute score of h -> m
        ## Precompute the necessrary components
        head_features = self.Linear_head(lstm_out)  # head_features.shape(sentence_len,mlp_hidden_dim//2)
        modif_features = self.Linear_modif(lstm_out)  # modif_features.shape(sentence_len,mlp_hidden_dim//2)

        ## Predict the head
        score_matrix = torch.empty(size=(sentence_len,sentence_len))
        for m in range(sentence_len):
            for h in range(sentence_len):
                feature_func = torch.cat((head_features[h],modif_features[m]))
                neuron = torch.tanh(feature_func)      # neuron.shape = [mlp_hidden_dim]
                score_matrix[m][h] = self.output_layer(neuron)

        return score_matrix

    # golden_head = data[2]   # score.shape
    def calc_loss(self,score_matrix,golden_head):
        loss = nn.CrossEntropyLoss()
        target = torch.tensor(golden_head[1:])
        input = torch.transpose(score_matrix,0,1)[1:]
        return loss(input,target)

if __name__ == '__main__':

    # Load test
    from pathlib import Path
    from data_processor import ConllDataSet
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

    # Read data
    train_path = Path("data","en-universal-train.conll")
    train_data = ConllDataSet(conll_path=train_path,
                              word_dropout=config_dict["word_dropout"])
    dev_path   = Path("data","en-universal-dev.conll")
    dev_data   = ConllDataSet(conll_path=dev_path,
                              word2index=train_data.word2index,
                              pos2index=train_data.pos2index)
    config_dict["model_param"]["vocab_size"] = train_data.vocab_size
    config_dict["model_param"]["pos_size"]   = train_data.pos_size

    # Init model
    model = BiLSTM_Parser_simple(**config_dict["model_param"])
    model.to(device)

    # Train setting
    optimizer = optim.Adam(model.parameters(),config_dict["learning_rate"])
    epoch_num = config_dict["epoch_num"]
    batch_size = config_dict["batch_size"]

    # Start train
    loss_tracker = []
    for epoch in range(epoch_num):  # epoch = 0
        running_loss = 0
        for i,data in enumerate(train_data):  # i = 0; data = train_data[i]
            word_tensor = data[0].to(device)
            pos_tensor  = data[1].to(device)
            score_matrix = model(word_tensor,pos_tensor)
            loss = model.calc_loss(score_matrix,data[2])
            (loss / batch_size).backward()
            running_loss += loss.item() / batch_size
            # Accumualte the gradient for each batch
            if (i % batch_size == (batch_size-1)) or (i == len(train_data)-1):
                # Update param
                optimizer.step()
                model.zero_grad()
                # Record the scores
                logger.debug(f"Now at {i}th data in epoch {epoch}")
                logger.debug(f"Current loss is {running_loss}")
                loss_tracker.append(running_loss)
                running_loss = 0

            # stop at the 30th batch
            if i > batch_size*30:
                continue

        # Save model
        result_path = result_dir_path / f"model_epoch{epoch}.pt"
        torch.save(model,str(result_path))


    # # Start train
    #  loss_tracker = []
    # for epoch in range(epoch_num):  # epoch = 0
    #     running_loss = 0
    #     for i,data in enumerate(train_data):  # i = 0; data = train_data[i]
    #         score_matrix = model(data[0],data[1])
    #         loss = model.calc_loss(score_matrix,data[2])
    #         (loss / batch_size).backward()
    #         running_loss += loss.item() / batch_size
    #         # Accumualte the gradient for each batch
    #         if (i % batch_size == (batch_size-1)) or (i == len(train_data)-1):
    #             # Update param
    #             optimizer.step()
    #             model.zero_grad()
    #             # Record the scores
    #             logger.debug(f"Now at {i}th data in epoch {epoch}")
    #             logger.debug(f"Current loss is {running_loss}")
    #             loss_tracker.append(running_loss)
    #             running_loss = 0
    #
    #     # Report loss for this epoch in dev data
    #     with torch.no_grad():
    #         running_loss_dev = 0
    #         for j,data in enumerate(dev_data):
    #             score_matrix = model(data[0],data[1])
    #             loss = model.calc_loss(score_matrix,data[2])
    #             running_loss_dev += loss.item()
    #             if i % 1000 == 999:
    #                 logger.debug(f"Now validating model; {j}th dev data now")
    #         mean_loss_dev = running_loss_dev/len(dev_data)
    #         logger.debug(f"Current mean loss for dev data is {mean_loss_dev}")
    #
    #     # Save model
    #     result_path = result_dir_path / f"model_epoch{epoch}.pt"
    #     torch.save(model,str(result_path))

# # Check forward() and loss funtion
# data = dev_data[1]
# head_hat,score_hat,score_golden = model(data[0],data[1],data[2])
# loss = margin_based_loss(score_hat,score_golden)
# logger.debug("Data flowed through the network")
#
# # Check computational graph
# component = loss.grad_fn
# while len(component.next_functions) != 0:
#     component = component.next_functions[0][0]
#     logger.debug(component)
