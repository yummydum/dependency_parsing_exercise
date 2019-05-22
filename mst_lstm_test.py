"""
Script to confirm model correctness by observing overfitting to small data

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
# Set logger
logger = set_logger(__name__)

# Read data
train_path = Path("data","en-universal-train.conll")
train_data = ConllDataSet(train_path)

# Init model
def init_model():
    model = BiLSTM_Parser(vocab_size = train_data.vocab_size,
                          pos_size   = train_data.pos_size,
                          word_embed_dim  = 100,
                          pos_embed_dim   = 25,
                          lstm_hidden_dim = 250,
                          mlp_hidden_dim  = 100,
                          num_layers      = 2)
    return model

def init_tracker():
    tracker = {}
    tracker["loss"] = []
    tracker["score_hat"] = []
    tracker["score_golden"] = []
    return tracker

def train_single_data_update():
    """ Update param by each data point (very noisy)"""
    # Init
    model = init_model()
    tracker = init_tracker()
    # Check if the model overfits to small data
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    epoch_num = 500
    for epoch in range(epoch_num):
        running_loss         = 0
        running_score_hat    = 0
        running_score_golden = 0
        for i,data in enumerate(train_data):  # i = 4; data = train_data[i]
            model.zero_grad()  # model.parameters()
            head_hat,score_hat,score_golden = model(*data)
            loss = margin_based_loss(score_hat,score_golden)
            loss.backward()
            optimizer.step()
            # Accumulate results
            running_loss         += loss.item()
            running_score_hat    += score_hat.item()
            running_score_golden += score_golden.item()
            # Only see the first 100 data to observe overfit
            if i > 99:
                logger.debug(f"Now finished epoch num {epoch}")
                mean_loss         = running_loss/100
                mean_score_hat    = running_score_hat/100
                mean_score_golden = running_score_golden/100
                logger.debug(f"Current mean loss is {mean_loss}")
                logger.debug(f"Current mean score_hat is {mean_score_hat}")
                logger.debug(f"Current mean score_golden is {mean_score_golden}")
                tracker["loss"].append(mean_loss)
                tracker["score_hat"].append(mean_score_hat)
                tracker["score_golden"].append(mean_score_golden)
                running_loss = 0
                running_score_hat = 0
                running_score_golden = 0
                break

    # Visualize the loss and scores
    for metric,tracks in tracker.items():
        fig,ax = plt.subplots()
        ax.plot(range(len(tracker[metric])),tracker[metric])
        result_path = Path("result",f"Single_data_update_{metric}.jpeg")
        fig.savefig(result_path)

    torch.save(model,"model_single_data_update_test.pt")

def train_mini_batch_update():
    """
    Update the param by mini batch unit
    (accumulate the gradient by batch_size)
    It is not doing mini batch training, but the gradient will be the same
    See how the estimate of the gradient will behave more well
    """
    model = init_model()
    tracker = init_tracker()
    # Check if the model overfits to small data
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    epoch_num = 200
    batch_size = 32
    for epoch in range(epoch_num):
        running_loss         = 0
        running_score_hat    = 0
        running_score_golden = 0
        for i,data in enumerate(train_data):  # i = 4; data = train_data[i]
            head_hat,score_hat,score_golden = model(*data)
            loss = margin_based_loss(score_hat,score_golden)
            (loss / batch_size).backward()
            if i % (batch_size-1) == 1:
                optimizer.step()
                model.zero_grad()
            # Accumulate results
            running_loss         += loss.item()
            running_score_hat    += score_hat.item()
            running_score_golden += score_golden.item()
            # See the scores
            # Only see the first 96 data to observe overfit
            if i > 95:
                # Update after seeing
                logger.debug(f"Now finished epoch num {epoch}")
                mean_loss         = running_loss/96
                mean_score_hat    = running_score_hat/96
                mean_score_golden = running_score_golden/96
                logger.debug(f"Current mean loss is {mean_loss}")
                logger.debug(f"Current mean score_hat is {mean_score_hat}")
                logger.debug(f"Current mean score_golden is {mean_score_golden}")
                tracker["loss"].append(mean_loss)
                tracker["score_hat"].append(mean_score_hat)
                tracker["score_golden"].append(mean_score_golden)
                running_loss = 0
                running_score_hat = 0
                running_score_golden = 0
                break
    # Visualize the loss and scores
    for metric,tracks in tracker.items():
        fig,ax = plt.subplots()
        ax.plot(range(len(tracker[metric])),tracker[metric])
        result_path = Path("result",f"Mini_batch_update_{metric}.jpeg")
        fig.savefig(result_path)

    torch.save(model,"model_mini_batch_update_test.pt")

if __name__ == '__main__':
    # model = train_single_data_update()
    train_mini_batch_update()
    # Evaluate result
    model = torch.load("model_mini_batch_update_test.pt")
    model.eval()
    for i,data in enumerate(train_data):  # i = 4; data = train_data[i]
        head_golden = data[2]
        head_hat,score_hat,_ = model(data)
        logger.debug(f"Golden_{i}",head_golden)
        logger.debug(f"Prediction_{i}",head_hat)
