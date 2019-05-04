import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from decoder import eisner_decode

torch.manual_seed(1)

class BiLSTM_Parser(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim,num_layers):
        super(BiLSTM_Parser, self).__init__()

        # Hyperparam
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Layers
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim // 2,
                            num_layers=num_layers,
                            bidirectional=True)

        # Hold the hidden state of LSTM layer
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def forward(self, sentence):
        # Extract features from lstm layer
        lstm_feats = self._get_lstm_features(sentence)
        # Find the best path, given the features.
        head_words_hat = eisner_decode(lstm_feats)
        return head_words_hat
