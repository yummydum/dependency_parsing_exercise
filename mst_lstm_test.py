"""
Script to confirm model correctness by observing overfitting to small data

For Hydrogen;
%load_ext autoreload
%autoreload 2
"""

from pathlib import Path
import torch.optim as optim
import matplotlib.pyplot as plt
from data_processor import ConllDataSet
from mst_lstm import BiLSTM_Parser,margin_based_loss
from util import set_logger
# set logger
logger = set_logger(__name__)

train_path = Path("data","en-universal-train.conll")
train_data = ConllDataSet(train_path)

model = BiLSTM_Parser(vocab_size = train_data.vocab_size,
                      pos_size   = train_data.pos_size,
                      word_embed_dim  = 100,
                      pos_embed_dim   = 25,
                      lstm_hidden_dim = 250,
                      mlp_hidden_dim  = 100,
                      num_layers      = 2)

# Check if the model overfits to small data
optimizer = optim.Adam(model.parameters(),lr=0.001)
# optimizer = optim.SGD(model.parameters(),lr=0.01)
epoch_num = 500
loss_tracker = []
score_hat_tracker = []
score_golden_tracker = []
for epoch in range(epoch_num):
    running_loss         = 0
    running_score_hat    = 0
    running_score_golden = 0
    for i,data in enumerate(train_data):  # i = 4; data = train_data[i]
        model.zero_grad()  # model.parameters()
        head_hat,score_hat,score_golden = model(data[0],data[1],data[2])
        loss = margin_based_loss(score_hat,score_golden)
        loss.backward()
        optimizer.step()
        # Accumulate results
        running_loss         += loss.item()
        running_score_hat    += score_hat.item()
        running_score_golden += score_golden.item()
        # See the scores
        # Only see the first 100 data to observe overfit
        if i > 100:
            logger.debug(f"Now finished epoch num {epoch}")
            mean_loss         = running_loss/100
            mean_score_hat    = running_score_hat/100
            mean_score_golden = running_score_golden/100
            logger.debug(f"Current mean loss is {mean_loss}")
            logger.debug(f"Current mean score_hat is {mean_score_hat}")
            logger.debug(f"Current mean score_golden is {mean_score_golden}")
            loss_tracker.append(mean_loss)
            score_hat_tracker.append(mean_score_hat)
            score_golden_tracker.append(mean_score_golden)
            running_loss = 0
            running_score_hat = 0
            running_score_golden = 0
            break

# Visualize the loss and scores
fig,ax = plt.subplots()
ax.plot(range(len(loss_tracker)),loss_tracker)
result_path = Path("result","small_data_loss.jpeg")
fig.savefig(result_path)

fig,ax = plt.subplots()
ax.plot(range(len(score_hat_tracker)),score_hat_tracker)
result_path = Path("result","small_data_score_hat.jpeg")
fig.savefig(result_path)

fig,ax = plt.subplots()
ax.plot(range(len(score_golden_tracker)),score_golden_tracker)
result_path = Path("result","small_data_score_golden.jpeg")
fig.savefig(result_path)

# data = train_data[1]
# head_hat,score_hat,score_golden = model(data[0],data[1],data[2])
# loss = margin_based_loss(score_hat,score_golden)
# loss.backward()
# param_dict = {name:param for name,param in model.named_parameters()}
# ## Words contained in train_data[1]
# param_dict["word_embeds.weight"][0]
# param_dict["word_embeds.weight"].grad[0]
# param_dict["word_embeds.weight"].grad[39]
# param_dict["word_embeds.weight"].grad[40]
# param_dict["word_embeds.weight"].grad[41]
# param_dict["word_embeds.weight"].grad[42]
# param_dict["word_embeds.weight"].grad[43]
# ## Words not contained in train_data[1]
# param_dict["word_embeds.weight"].grad[1]
# param_dict["word_embeds.weight"].grad[2]
#
# # Check if the model can memorize single data point
# optimizer = optim.SGD(model.parameters(),lr=0.01)
# loss_tracker = []
# head_hat_traker = []
# head_golden_tracker = []
# for _ in range(100):
#     head_hat,score_hat,score_golden = model(data[0],data[1],data[2])
#     logger.debug(f"Predicted head is {head_hat}")
#     logger.debug(f"Score of predicted head is {score_hat.item()}")
#     logger.debug(f"Score of golden head is {score_golden.item()}")
#     loss = margin_based_loss(score_hat,score_golden)
#     logger.debug(f"Loss is {loss.item()}")
#     # Track
#     loss_tracker.append(loss.item())
#     head_hat_traker.append(score_hat.item())
#     head_golden_tracker.append(score_golden.item())
#     loss.backward()
#     logger.debug(param_dict["word_embeds.weight"][0])
#     logger.debug(param_dict["word_embeds.weight"].grad[0])
#     logger.debug(param_dict['output_layer.weight'][0])
#     logger.debug(param_dict['output_layer.weight'].grad[0])
#     optimizer.step()
# fig,ax = plt.subplots()
# ax.plot(range(len(loss_tracker)),loss_tracker)
# result_path = Path("result","single_data_loss.jpeg")
# fig.savefig(result_path)
