"""
For Hydrogen;
%load_ext autoreload
%autoreload 2
"""
from typing import List,Tuple,Union,Optional
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from decoder import eisner_decode
from util import set_logger
# set logger
logger = set_logger(__name__)
# Fix seed
torch.manual_seed(1)

class BiLSTM_Parser(nn.Module):

    def __init__(self,
                 vocab_size,
                 pos_size,
                 word_embed_dim,
                 pos_embed_dim,
                 lstm_hidden_dim,
                 mlp_hidden_dim,
                 num_layers):

        super(BiLSTM_Parser,self).__init__()

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
                pos_tensor :torch.LongTensor,
                head_golden:Optional[List[int]] = None)  \
                -> Tuple[List[int],torch.tensor,torch.tensor]:

        sentence_len = len(word_tensor[0])
        # Word/POS embedding
        word_embeds = self.word_embeds(word_tensor)     # word_embeds.shape = (1,sentence_len,word_embed_dim)
        pos_embeds  = self.pos_embeds(pos_tensor)       # pos_embeds.shape = (1,sentence_len,pos_embed_dim)
        embeds = torch.cat((word_embeds,pos_embeds),2)  # embeds.shape = (1,sentence_len,(word_embed_dim+pos_embed_dim))
        embeds = embeds.view(sentence_len,1,-1)         # embeds.shape = (sentence_len,1,(word_embed_dim+pos_embed_dim))
        # Bidirectional LSTM
        lstm_out, _ = self.lstm(embeds)                 # lstm_out.shape = (sentence_len,1,lstm_hidden_dim)
        lstm_out = lstm_out.view(sentence_len, self.lstm_hidden_dim)
        # Compute score of h -> m (Hold values in float as well for decoding etc)
        ## Precompute the necessrary components
        head_features = self.Linear_head(lstm_out)  # head_features.shape(sentence_len,mlp_hidden_dim//2)
        modif_features = self.Linear_modif(lstm_out)  # modif_features.shape(sentence_len,mlp_hidden_dim//2)
        ## Compute score matrix
        score_matrix = torch.empty(size=(sentence_len,sentence_len))
        for m in range(sentence_len):
            for h in range(sentence_len):
                feature_func = torch.cat((head_features[h],modif_features[m]))
                neuron = torch.tanh(feature_func)      # neuron.shape = [mlp_hidden_dim]
                score_matrix[m][h] = self.output_layer(neuron)

        # Find the best path, given the score_matrix
        head_hat,score_hat = eisner_decode(score_matrix,head_golden)

        # Score for the golden head
        if head_golden is not None:
            score_golden = 0
            for m,h in enumerate(head_golden):
                score_golden += score_matrix[h][m]
        else:
            score_golden = None

        return head_hat,score_hat,score_golden

# util func for debug     for test : tensor = embeds
# def get_n_next_function(tensor:torch.Tensor,n:Optional[int]=None):
#     """
#     Function to check what is the n next function in the backword propagation.
#     If n is None, go back as far as possible.
#     """
#     component = tensor.grad_fn
#     logger.debug(component)
#     if n is not None:
#         for i in range(n):
#             if len(component.next_functions) == 0:
#                 raise ValueError(f"No more function back than {i}")
#             component = component.next_functions[0][0]
#             logger.debug(component)
#     else:
#         while len(component.next_functions) != 0:
#             component = component.next_functions[0][0]
#             logger.debug(component)
#     return component

if __name__ == '__main__':

    ###  Script for test

    # Load test
    from pathlib import Path
    from data_processor import ConllDataSet
    dev_path = Path("data","en-universal-dev.conll")
    dev_data = ConllDataSet(dev_path)

    # Init model  (self = model)
    model = BiLSTM_Parser(vocab_size = dev_data.vocab_size,
                          pos_size   = dev_data.pos_size,
                          word_embed_dim  = 100,
                          pos_embed_dim   = 25,
                          lstm_hidden_dim = 250,
                          mlp_hidden_dim  = 100,
                          num_layers      = 2)
    # Check forward() and loss funtion
    data = dev_data[1]
    head_hat,score_hat,score_golden = model(data[0],data[1],data[2])
    logger.debug("Data flowed through the network")

    # Check computational graph
    component = loss.grad_fn
    while len(component.next_functions) != 0:
        component = component.next_functions[0][0]
        logger.debug(component)
