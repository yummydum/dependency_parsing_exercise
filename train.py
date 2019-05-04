import torch
from mst_lstm import BiLSTM_Parser
from util import read_conll,get_indexers

# Path of data
train_data_path = Path("data","en-universal-train.conll")
dev_data_path   = Path("data","en-universal-dev.conll")
test_data_path  = Path("data","en-universal-test.conll")

# # word/pos to index
# word2index,pos2index = get_indexers()

# Hyperparam
EMBEDDING_DIM = 10
HIDDEN_DIM    = 10

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)
