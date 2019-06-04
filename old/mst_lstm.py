"""
For Hydrogen;
%load_ext autoreload
%autoreload 2
"""
from typing import List,Tuple,Union
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from decoder import eisner_decode
from data_processor import ConllDataSet
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
                            bidirectional= True,
                            dropout=0.25)
        self.Linear_head = nn.Linear(lstm_hidden_dim,mlp_hidden_dim //2)
        self.Linear_modif = nn.Linear(lstm_hidden_dim,mlp_hidden_dim //2)
        self.output_layer = nn.Linear(mlp_hidden_dim,1)  # output layer

        # Store intermediate score matrices here (values are float, not tensor)
        self.score_matrix_float = None

        # Is the model used in training or inference
        self.is_train_mode = True

    # For test : word_tensor = data[0]; pos_tensor = data[1]
    def compute_score_matrix(self,
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
        # Compute score of h -> m (Hold values in float as well for decoding etc)
        ## Precompute the necessrary components
        head_features = self.Linear_head(lstm_out)  # head_features.shape(sentence_len,mlp_hidden_dim//2)
        modif_features = self.Linear_modif(lstm_out)  # head_features.shape(sentence_len,mlp_hidden_dim//2)
        ## Compute
        score_matrix = []
        score_matrix_float = []
        for h in range(sentence_len):
            score_matrix.append([])
            score_matrix_float.append([])
            for m in range(sentence_len):
                # Words cannot depend on itself
                if h == m:
                    score_matrix[h].append(np.nan)
                    score_matrix_float[h].append(np.nan)
                else:
                    feature_func = torch.cat((head_features[h],modif_features[m]))
                    neuron = torch.tanh(feature_func)      # neuron.shape = [mlp_hidden_dim]
                    score = self.output_layer(neuron)
                    score_matrix[h].append(score)
                    score_matrix_float[h].append(score.item())

        return score_matrix,score_matrix_float

    def compute_head_score(self,
                           score_matrix:List[List[torch.Tensor]],
                           head_list:List[int]) \
                           -> torch.Tensor:
        score = 0
        for m,h in enumerate(head_list):
            score += score_matrix[h][m]
        return score

    def compute_hamming_cost(self,head_hat:List[int],head_golden:List[int]) -> int:
        # Ignore ROOT
        head_hat = np.array(head_hat[1:])
        head_golden = np.array(head_golden[1:])
        # Number of head not matching
        return int(np.sum(head_hat != head_golden))

    def forward(self,
                word_tensor:torch.LongTensor,
                pos_tensor :torch.LongTensor,
                head_golden:List[int] = None)  \
                -> Tuple[List[int],torch.Tensor,Union[torch.Tensor,None]]:

        # Check inconsistent argument and mode
        if self.is_train_mode and head_golden is None:
            raise ValueError("Pass golden for training mode")
        elif not self.is_train_mode and head_golden is not None:
            raise ValueError("Golden is not needed for inference")

        # Calculate score matrix (Hold values for convenience)
        score_matrix,score_matrix_float = self.compute_score_matrix(word_tensor,pos_tensor)
        self.score_matrix_float = score_matrix_float
        # Find the best path, given the score_matrix
        head_hat = eisner_decode(score_matrix_float,head_golden)
        # Compute the score
        score_hat = self.compute_head_score(score_matrix,head_hat)
        if head_golden is not None:
            score_golden = self.compute_head_score(score_matrix,head_golden)
            score_hat += self.compute_hamming_cost(head_hat,head_golden)
        else:
            score_golden = None
        return head_hat,score_hat,score_golden

# Loss function
def margin_based_loss(score_hat:torch.Tensor,
                      score_golden:torch.Tensor) -> torch.Tensor:
    margin = score_golden - score_hat
    return max(0,1 - margin)

if __name__ == '__main__':

    ###  Script for test

    # Load test
    from pathlib import Path
    train_path = Path("data","en-universal-train.conll")
    train_data = ConllDataSet(train_path)

    # Init model  (self = model)
    model = BiLSTM_Parser(vocab_size = train_data.vocab_size,
                          pos_size   = train_data.pos_size,
                          word_embed_dim  = 100,
                          pos_embed_dim   = 25,
                          lstm_hidden_dim = 250,
                          mlp_hidden_dim  = 100,
                          num_layers      = 2)
    # Check forward() and loss funtion
    data = train_data[1]
    head_hat,score_hat,score_golden = model(data[0],data[1],data[2])
    loss = margin_based_loss(score_hat,score_golden)
    logger.debug("Data flowed through the network!")
    # Check computational graph
    component = loss.grad_fn
    while len(component.next_functions) != 0:
        logger.debug(component)
        component = component.next_functions[0][0]
