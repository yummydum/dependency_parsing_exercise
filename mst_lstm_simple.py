"""
For Hydrogen;
%load_ext autoreload
%autoreload 2
"""
from typing import List,Tuple,Union,Optional
import numpy as np
import torch
import torch.nn as nn
from util import set_logger
# set logger
logger = set_logger(__name__)
# Fix seed
torch.manual_seed(1)

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

    # Train setting
    optimizer = optim.Adam(model.parameters(),config_dict["learning_rate"])
    epoch_num = config_dict["epoch_num"]
    batch_size = config_dict["batch_size"]

    # Start train
    tracker = init_tracker()
    for epoch in range(epoch_num):  # epoch = 0
        running_tracker = init_running_tracker()
        for i,data in enumerate(train_data):  # i = 4; data = train_data[i]
            score_matrix = model(data[0],data[1])
            target = torch.LongTensor(data[2])
            loss = nn.CrossEntropyLoss(score_matrix,target)  # score_matrix.shape,len(target)
            (loss / batch_size).backward()
            accuracy = 1 - (mst_lstm.compute_hamming_cost(head_hat,data[2]) / len(head_hat))
            update_running_tracker(running_tracker,loss,score_hat,score_golden,accuracy)
            # Accumualte the gradient for each batch
            if (i % batch_size == (batch_size-1)) or (i == len(train_data)-1):
                # Update param
                optimizer.step()
                model.zero_grad()
                # Record the scores
                logger.debug(f"Now at {i}th data in epoch {epoch}")
                update_tracker(tracker,batch_size,**running_tracker)
                running_tracker = init_running_tracker()

        # Report loss for this epoch in dev data
        with torch.no_grad():
            running_loss_dev = 0
            for j,data in enumerate(dev_data):
                head_hat,score_hat,score_golden = model(*data)
                loss = mst_lstm.margin_based_loss(score_hat,score_golden)
                running_loss_dev += loss.item()
                if i % 1000 == 999:
                    logger.debug(f"Now validating model; {j}th dev data now")
            mean_loss_dev = running_loss_dev/len(dev_data)
            tracker["dev_loss"].append(mean_loss_dev)
            logger.debug(f"Current mean loss for dev data is {mean_loss_dev}")

        # Save model
        result_path = result_dir_path / f"model_epoch{epoch}.pt"
        torch.save(model,str(result_path))

    ## Store the result of tracker
    tracker_path = result_dir_path / "tracker_result.json"
    with tracker_path.open(mode="w") as fp:
        json.dump(tracker, fp)
    for metric,tracks in tracker.items():
        fig,ax = plt.subplots()
        ax.plot(range(len(tracker[metric])),tracker[metric])
        plot_path = result_dir_path / f"{metric}.jpeg"
        fig.savefig(plot_path)

def init_tracker():
    """ Accumualte metrics over the whole train process """
    tracker = dict()
    tracker["train_loss"]   = []
    tracker["dev_loss"]     = []
    tracker["score_hat"]    = []
    tracker["score_golden"] = []
    tracker["accuracy"]     = []
    return tracker

def update_tracker(tracker:Dict,batch_size,loss_train,score_hat,score_golden,accuracy):
    mean_loss_train   = loss_train/batch_size
    mean_score_hat    = score_hat/batch_size
    mean_score_golden = score_golden/batch_size
    mean_accuracy     = accuracy/batch_size
    logger.debug(f"Current mean loss for train data is {mean_loss_train}")
    logger.debug(f"Current mean score_hat is {mean_score_hat}")
    logger.debug(f"Current mean score_golden is {mean_score_golden}")
    logger.debug(f"Current mean accuracy is {mean_accuracy}")
    tracker["train_loss"].append(mean_loss_train)
    tracker["score_hat"].append(mean_score_hat)
    tracker["score_golden"].append(mean_score_golden)

def init_running_tracker():
    """ Accumulate metrics for one process over batch """
    running_tracker = dict()
    running_tracker["loss_train"]   = 0
    running_tracker["score_hat"]    = 0
    running_tracker["score_golden"] = 0
    running_tracker["accuracy"]     = 0
    return running_tracker

def update_running_tracker(running_tracker,loss,score_hat,score_golden,accuracy):
    running_tracker["loss_train"]   += loss.item()
    running_tracker["score_hat"]    += score_hat.item()
    running_tracker["score_golden"] += score_golden.item()
    running_tracker["accuracy"]     += accuracy
    return running_tracker


    train(config_dict)



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
