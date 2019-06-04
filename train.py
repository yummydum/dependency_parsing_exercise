"""
For Hydrogen;
%load_ext autoreload
%autoreload 2
"""

import json
from datetime import datetime
from typing import Dict
from pathlib import Path
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from data_processor import ConllDataSet
from mst_lstm import BiLSTM_Parser,margin_based_loss
from util import set_logger

logger = set_logger("train.py")
torch.manual_seed(0)

def train(config_dict:Dict):

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

    # Init model and tracker
    model = BiLSTM_Parser(**config_dict["model_param"])
    tracker = init_tracker()

    # Train setting
    optimizer = optim.Adam(model.parameters(),config_dict["learning_rate"])
    epoch_num = config_dict["epoch_num"]
    batch_size = config_dict["batch_size"]

    # Start train
    for epoch in range(epoch_num):
        model.train()
        running_tracker = init_running_tracker()
        for i,data in enumerate(train_data):  # i = 4; data = train_data[i]
            head_hat,score_hat,score_golden = model(*data)
            loss = margin_based_loss(score_hat,score_golden)
            (loss / batch_size).backward()
            update_running_tracker(running_tracker,loss,score_hat,score_golden)
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
            model.eval()
            running_loss_dev = 0
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
    return tracker

def update_tracker(tracker:Dict,batch_size,loss_train,score_hat,score_golden):
    mean_loss_train   = loss_train/batch_size
    mean_score_hat    = score_hat/batch_size
    mean_score_golden = score_golden/batch_size
    logger.debug(f"Current mean loss for train data is {mean_loss_train}")
    logger.debug(f"Current mean score_hat is {mean_score_hat}")
    logger.debug(f"Current mean score_golden is {mean_score_golden}")
    tracker["train_loss"].append(mean_loss_train)
    tracker["score_hat"].append(mean_score_hat)
    tracker["score_golden"].append(mean_score_golden)

def init_running_tracker():
    """ Accumulate metrics for one process over batch """
    running_tracker = dict()
    running_tracker["loss_train"] = 0
    running_tracker["score_hat"] = 0
    running_tracker["score_golden"] = 0
    return running_tracker

def update_running_tracker(running_tracker,loss,score_hat,score_golden):
    running_tracker["loss_train"] += loss.item()
    running_tracker["score_hat"] += score_hat.item()
    running_tracker["score_golden"] += score_golden.item()
    return running_tracker


if __name__ == '__main__':
    # The configuration for the training process
    model_param = {"word_embed_dim"  : 100,
                   "pos_embed_dim"   : 25,
                   "lstm_hidden_dim" :250,
                   "mlp_hidden_dim"  :100,
                   "num_layers"      : 2}
    config_dict = {
        "model_param":model_param,
        "word_dropout":True,
        "learning_rate":0.001,
        "epoch_num":10,
        "batch_size":32
    }

    train(config_dict)
