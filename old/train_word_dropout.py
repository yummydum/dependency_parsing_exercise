"""
For Hydrogen;
%load_ext autoreload
%autoreload 2
"""

from pathlib import Path
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from data_processor import ConllDataSet
from mst_lstm import BiLSTM_Parser,margin_based_loss
from util import set_logger

logger = set_logger(__name__)
torch.manual_seed(0)

# Read data
train_path = Path("data","en-universal-train.conll")
train_data = ConllDataSet(conll_path=train_path,word_dropout=True)
dev_path   = Path("data","en-universal-dev.conll")
dev_data   = ConllDataSet(conll_path=dev_path,
                          word2index=train_data.word2index,
                          pos2index=train_data.pos2index)

# Init model
model = BiLSTM_Parser(vocab_size = train_data.vocab_size,
                      pos_size   = train_data.pos_size,
                      word_embed_dim  = 100,
                      pos_embed_dim   = 25,
                      lstm_hidden_dim = 250,
                      mlp_hidden_dim  = 100,
                      num_layers      = 2)

tracker = dict()
tracker["train_loss"]   = []
tracker["dev_loss"]     = []
tracker["score_hat"]    = []
tracker["score_golden"] = []

optimizer = optim.Adam(model.parameters(),lr=0.001)
epoch_num = 10
batch_size = 32

for epoch in range(epoch_num):
    model.train()
    running_loss_train   = 0
    running_score_hat    = 0
    running_score_golden = 0
    for i,data in enumerate(train_data):  # i = 4; data = train_data[i]
        head_hat,score_hat,score_golden = model(*data)
        loss = margin_based_loss(score_hat,score_golden)
        (loss / batch_size).backward()
        running_loss_train   += loss.item()
        running_score_hat    += score_hat.item()
        running_score_golden += score_golden.item()
        # Accumualte the gradient for each batch
        if (i % batch_size == (batch_size-1)) or (i == len(train_data)-1):
            # Update param
            optimizer.step()
            model.zero_grad()
            # Record the scores
            logger.debug(f"Now at {i}th data in epoch {epoch}")
            mean_loss_train   = running_loss_train/batch_size
            mean_score_hat    = running_score_hat/batch_size
            mean_score_golden = running_score_golden/batch_size
            logger.debug(f"Current mean loss for train data is {mean_loss_train}")
            logger.debug(f"Current mean score_hat is {mean_score_hat}")
            logger.debug(f"Current mean score_golden is {mean_score_golden}")
            tracker["train_loss"].append(mean_loss_train)
            tracker["score_hat"].append(mean_score_hat)
            tracker["score_golden"].append(mean_score_golden)
            running_loss_train = 0
            running_score_hat = 0
            running_score_golden = 0

    # Report loss for this epoch in dev data
    with torch.no_grad():
        model.eval()
        running_loss_dev     = 0
        for j,data in enumerate(dev_data):
            head_hat,score_hat,score_golden = model(*data)
            loss = margin_based_loss(score_hat,score_golden)
            running_loss_dev += loss.item()
            if i % 1000 == 99:
                logger.debug(f"Now validating model; {j}th dev data now")

        mean_loss_dev = running_loss_dev/len(dev_data)
        tracker["dev_loss"].append(mean_loss_dev)
        logger.debug(f"Current mean loss for dev data is {mean_loss_dev}")

    # Save model
    result_path = Path("result","model",f"model_word_dropout_epoch{epoch}.pt")
    torch.save(model,str(result_path))

# Visualize the loss and scores
for metric,tracks in tracker.items():
    fig,ax = plt.subplots()
    ax.plot(range(len(tracker[metric])),tracker[metric])
    plot_path = Path("result",f"{__name__}_{metric}_word_dropout.jpeg")
    fig.savefig(plot_path)
