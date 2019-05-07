from pathlib import Path
from typing import Tuple
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from decoder import eisner_decode
from data_processor import count_word_stat,data_generator
from util import set_logger
# set logger
logger = set_logger(__name__)

torch.manual_seed(1)

class BiLSTM_Parser(nn.Module):

    def __init__(self,vocab_size,pos_size,word_embed_dim,pos_embed_dim, lstm_hidden_dim,mlp_hidden_dim,num_layers):
        super(BiLSTM_Parser, self).__init__()

        # hidden dimension must be an even number for now
        assert lstm_hidden_dim % 2 == 0

        # Hyperparam
        self.vocab_size = vocab_size
        self.word_embed_dim = word_embed_dim
        self.pos_embed_dim  = pos_embed_dim
        self.lstm_hidden_dim = lstm_hidden_dim

        # Layers
        self.word_embeds = nn.Embedding(vocab_size, word_embed_dim)
        self.pos_embeds  = nn.Embedding(pos_size,   pos_embed_dim)
        self.lstm = nn.LSTM(input_size   = word_embed_dim+pos_embed_dim,
                            hidden_size  = lstm_hidden_dim // 2,
                            num_layers   = num_layers,
                            bidirectional= True)
        self.Linear = nn.Linear(2*lstm_hidden_dim,mlp_hidden_dim)
        self.output_layer = nn.Linear(mlp_hidden_dim,1)  # output layer

        # Hold the hidden state of LSTM layer
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.lstm_hidden_dim // 2),
                torch.randn(2, 1, self.lstm_hidden_dim // 2))

    def _compute_score_matrix(self, word_tensor,pos_tensor) -> np.matrix:
        """
        Compute a score matrix where
        (i,j) element is the score of ith word being the head of jth word
        """
        sentence_len = len(word_tensor)
        self.hidden = self.init_hidden()
        # Word/POS embedding
        word_embeds = self.word_embeds(word_tensor)
        pos_embeds  = self.pos_embeds(pos_tensor)
        embeds = torch.cat((word_embeds,pos_embeds),1).view(sentence_len,1,-1)
        # Bidirectional LSTM
        lstm_out, self.hidden = self.lstm(embeds)
        lstm_out = lstm_out.view(sentence_len, self.lstm_hidden_dim)  # lstm_out.shape = (sentence_len,lstm_hidden_dim)
        # Compute score of h -> m
        score_matrix = np.empty((sentence_len,sentence_len))
        for h in range(sentence_len):
            for m in range(sentence_len):    # for test : h=0;m=1
                # Words cannot depend on itself
                if h == m:
                    score_matrix[h][m] = np.nan
                else:
                    feature_func = torch.cat((lstm_out[h],lstm_out[m]))
                    neuron = torch.tanh(self.Linear(feature_func))
                    score = self.output_layer(neuron)
                    score_matrix[h][m] = score
        return score_matrix

    def forward(self, word_tensor,pos_tensor) -> Tuple[np.array,int]:
        """
        Determine the optimal dependency structure and it's score under the current param
        """
        # Extract features from lstm layer
        lstm_feats = self._compute_score_matrix(word_tensor,pos_tensor)
        # Find the best path, given the features.
        head_words_hat,max_score = eisner_decode(lstm_feats)
        return head_words_hat,max_score

if __name__ == '__main__':
    train_path = Path("data","en-universal-train.conll")
    # Get first data for test
    data_gen = data_generator(train_path)
    for data_i in data_gen:
        break
    word_tensor,pos_tensor = data_i

    # Initialize the model
    words_count,pos_count,rel_count = count_word_stat(train_path)
    model = BiLSTM_Parser(vocab_size = len(words_count.keys()),
                          pos_size   = len(pos_count.keys()),
                          word_embed_dim  = 100,
                          pos_embed_dim   = 10,
                          lstm_hidden_dim = 30,
                          mlp_hidden_dim  = 15,
                          num_layers      = 2)
    # Forward
    model.forward(word_tensor,pos_tensor)  # self = model
