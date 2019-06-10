"""
For Hydrogen;
%load_ext autoreload
%autoreload 2
"""

import line_profiler
from pathlib import Path
from typing import List,Tuple,Union
import numpy as np
import torch
from data_processor import ConllDataSet
import mst_lstm
import mst_lstm_slow
from decoder import eisner_decode

# Init
train_path = Path("data","en-universal-train.conll")
train_data = ConllDataSet(train_path)
result_path = Path("log","profiler")
model_config = {"vocab_size" : train_data.vocab_size,
                "pos_size"   : train_data.pos_size,
                "word_embed_dim" : 100,
                "pos_embed_dim"   : 25,
                "lstm_hidden_dim" : 250,
                "mlp_hidden_dim"  : 100,
                "num_layers"      : 2}

# Slow
profiler_slow = line_profiler.LineProfiler()
model_slow = mst_lstm_slow.BiLSTM_Parser(**model_config)
profiler_slow.add_function(model_slow.forward)
profiler_slow.add_function(model_slow.compute_score_matrix)
profiler_slow.enable()
for i,data in enumerate(train_data):  # i = 0; data = train_data[i]
    _ = model_slow(*data)
    if i == 31:
        break
profiler_slow.disable()
profiler_slow.print_stats()
profiler_slow.dump_stats(result_path / "mst_lstm_slow.lp")

# Fast
profiler = line_profiler.LineProfiler()
model = mst_lstm.BiLSTM_Parser(**model_config)
profiler.add_function(model.forward)
profiler.add_function(model.compute_score_matrix)
profiler.enable()
for i,data in enumerate(train_data):  # i = 0; data = train_data[i]
    _ = model(*data)
    if i == 31:
        break
profiler.disable()
profiler.print_stats()
profiler.dump_stats(result_path / "mst_lstm.lp")

# Batch
