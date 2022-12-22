import PyTorchDiscreteFlows.discrete_flows.disc_utils as disc_utils
from PyTorchDiscreteFlows.discrete_flows.made import MADE
from PyTorchDiscreteFlows.discrete_flows.mlp import MLP
from PyTorchDiscreteFlows.discrete_flows.embed import EmbeddingLayer
from PyTorchDiscreteFlows.discrete_flows.disc_models import *

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import figure
from matplotlib.backends import backend_agg
from scipy.stats import multivariate_normal
import torch
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
import itertools
import math
import random
import time
import os
from preprocessing import *
from train_utils import *
import argparse

seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ['PYTHONHASHSEED'] = str(seed)
DEBUG = False

parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=int)
parse = parser.parse_args()


def find_min_test_loss(data, path, kfold, vocab_size, sequence_length, disc_layer_type, hidden_layer=0, alpha=1, beta=1, id=True):
    device = torch.device('cpu')

    total_loss = []
    total_time = []
    for kfold_idx in range(kfold):
        train_data, test_data = Mai_create_X_train_test(data, 4 / 5, kfold, kfold_idx)
        loss, time = test_disc_flow(device, test_data, vocab_size, sequence_length, disc_layer_type, hidden_layer, load_path=path + 'k' + str(kfold) + '_' + str(kfold_idx) + '.pt', alpha=alpha, beta=beta, id=id)
        total_loss.append(loss)
        total_time.append(time)
    total_loss = np.array(total_loss)
    total_time = np.array(total_time)

    print('Average Min Loss')
    print(np.average(total_loss))
    print('Std Min Loss')
    print(np.std(total_loss))
    print('Average Time')
    print(np.average(total_time))
    print('Std Time')
    print(np.std(total_time))


