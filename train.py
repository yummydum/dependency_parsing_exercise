"""
%load_ext autoreload
%autoreload 2
"""
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from mst_lstm import BiLSTM_Parser
from data_processor import ConllDataSet
from util import set_logger

# set logger
logger = set_logger(__name__)
# Fix seed
torch.manual_seed(1)

# Paths for data set
train_data_path = Path("data","en-universal-train.conll")
dev_data_path   = Path("data","en-universal-dev.conll")

# Load data
train_data = ConllDataSet(train_data_path)
train_data_loader = DataLoader(train_data,shuffle=True)

# Init model
model = BiLSTM_Parser(vocab_size = train_data.vocab_size,
                      pos_size   = train_data.pos_size,
                      word_embed_dim  = 100,
                      pos_embed_dim   = 10,
                      lstm_hidden_dim = 30,
                      mlp_hidden_dim  = 15,
                      num_layers      = 2)
# Init optimizer
optimizer = optim.Adam(model.parameters(),lr=0.0001)

# Config
epoch_num = 5
record_interval = 10

for epoch in range(epoch_num):
    loss = 0
    loss_tracker = []
    # Iterate over the data
    for i,data in enumerate(train_data):
        word_tensor = data[0]
        pos_tensor  = data[1]
        head_golden = data[2]
        model.zero_grad()
        head_hat,score_hat,score_golden = model(word_tensor,
                                                pos_tensor,
                                                head_golden)
        # Loss augmented inference
        # -> Penalize if the margin is smaller than 1
        margin = score_golden - score_hat
        loss += max(0,1 - margin)
        optimizer.step()

        # Accumulate and report the mean loss for recent (record_interval) data point
        if (i % record_interval) == record_interval-1:
            logger.debug(f"Now at {i+1}th data")
            mean_loss = loss/record_interval
            logger.debug(f"Current mean loss is {mean_loss}")
            loss_tracker.append(mean_loss)
            loss = 0

    # Calc train/dev loss for this epoch
    train_loss = ""
    dev_loss   = ""

# Report the result


# Test the model
test_data_path  = Path("data","en-universal-test.conll")
