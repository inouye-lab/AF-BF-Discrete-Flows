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
'''
data = preprocess_805_snp_data('805_SNP_1000G_real.hapt.zip')
path = 'Model/805/Autoregressive/'
find_min_test_loss(data, path, 3, vocab_size=2, sequence_length=805, disc_layer_type='autoreg', hidden_layer=805*2)

path = 'Model/805/Bipartite/Non_Identity/Straight_Layer/beta_1_16/'
find_min_test_loss(data, path, 3, vocab_size=2, sequence_length=805, disc_layer_type='bipartite', id=False, beta=1/16)
'''
'''
data = preprocess_binary_mnist()
path = 'Model/MNIST/Autoregressive/'
find_min_test_loss(data, path, 3, vocab_size=2, sequence_length=784, disc_layer_type='autoreg', hidden_layer=784*2)

path = 'Model/MNIST/Bipartite/Identity/Encoder_Layer/alpha_2/'
find_min_test_loss(data, path, 3, vocab_size=2, sequence_length=784, disc_layer_type='bipartite', hidden_layer=784*2, alpha=2, id=True)

path = 'Model/MNIST/Bipartite/Identity/Encoder_Layer/alpha_4/'
find_min_test_loss(data, path, 3, vocab_size=2, sequence_length=784, disc_layer_type='bipartite', hidden_layer=784*2, alpha=4)

path = 'Model/MNIST/Bipartite/Identity/Encoder_Layer/alpha_8/'
find_min_test_loss(data, path, 3, vocab_size=2, sequence_length=784, disc_layer_type='bipartite', hidden_layer=784*2, alpha=8)

path = 'Model/MNIST/Bipartite/Identity/Straight_Layer/beta_1/'
find_min_test_loss(data, path, 3, vocab_size=2, sequence_length=784, disc_layer_type='bipartite', hidden_layer=784*2, beta=1)

path = 'Model/MNIST/Bipartite/Identity/Straight_Layer/beta_1_4/'
find_min_test_loss(data, path, 3, vocab_size=2, sequence_length=784, disc_layer_type='bipartite', hidden_layer=784*2, beta=1/4)

path = 'Model/MNIST/Bipartite/Identity/Straight_Layer/beta_1_16/'
find_min_test_loss(data, path, 3, vocab_size=2, sequence_length=784, disc_layer_type='bipartite', hidden_layer=784*2, beta=1/16)

path = 'Model/MNIST/Bipartite/Non_Identity/Encoder_Layer/alpha_2/'
find_min_test_loss(data, path, 3, vocab_size=2, sequence_length=784, disc_layer_type='bipartite', hidden_layer=784*2, alpha=2)

path = 'Model/MNIST/Bipartite/Non_Identity/Encoder_Layer/alpha_4/'
find_min_test_loss(data, path, 3, vocab_size=2, sequence_length=784, disc_layer_type='bipartite', hidden_layer=784*2, alpha=4)

path = 'Model/MNIST/Bipartite/Non_Identity/Encoder_Layer/alpha_8/'
find_min_test_loss(data, path, 3, vocab_size=2, sequence_length=784, disc_layer_type='bipartite', hidden_layer=784*2, alpha=8)

path = 'Model/MNIST/Bipartite/Non_Identity/Straight_Layer/beta_1/'
find_min_test_loss(data, path, 3, vocab_size=2, sequence_length=784, disc_layer_type='bipartite', hidden_layer=784*2, beta=1)

path = 'Model/MNIST/Bipartite/Non_Identity/Straight_Layer/beta_1_4/'
find_min_test_loss(data, path, 3, vocab_size=2, sequence_length=784, disc_layer_type='bipartite', hidden_layer=784*2, beta=1/4)

path = 'Model/MNIST/Bipartite/Non_Identity/Straight_Layer/beta_1_16/'
find_min_test_loss(data, path, 3, vocab_size=2, sequence_length=784, disc_layer_type='bipartite', hidden_layer=784*2, beta=1/16)
'''

'''

#exp1
print('exp1')
orig_probs, coded_data = create_syn_data_paper(alpha=1,k=2,d=2,n_samples=10000, rnd_seed=42, manual_prob=True, orig_probs=[1/3,1/6,1/6,1/3], dirchlet_seed=0)
data = dec_to_bin(coded_data, n_features=2, k=2)

path = 'Model/Synthetic/Exp1/autoregressive/'
find_min_test_loss(data, path, 5, vocab_size=2, sequence_length=2, disc_layer_type='autoreg', hidden_layer=64)

path = 'Model/Synthetic/Exp1/bipartite/'
find_min_test_loss(data, path, 5, vocab_size=2, sequence_length=2, disc_layer_type='bipartite', hidden_layer=64, id=False, beta=16)

#exp2
print('exp2')
orig_probs, coded_data = create_syn_data_paper(alpha=1,k=2,d=2,n_samples=10000,rnd_seed=42,manual_prob=True,orig_probs=[1/8,3/8,3/8,1/8],dirchlet_seed=0)
data = dec_to_bin(coded_data,n_features=2,k=2)

path = 'Model/Synthetic/Exp2/autoregressive/'
find_min_test_loss(data, path, 5, vocab_size=2, sequence_length=2, disc_layer_type='autoreg', hidden_layer=64)

path = 'Model/Synthetic/Exp2/bipartite/'
find_min_test_loss(data, path, 5, vocab_size=2, sequence_length=2, disc_layer_type='bipartite', hidden_layer=64, id=False, beta=16)

#exp3
print('exp3')
batch_size, sequence_length, vocab_size, k_fold = 1024, 5, 5, 5
orig_probs, coded_data = create_syn_data_paper(alpha=1,k=vocab_size,d=sequence_length,n_samples=10000,rnd_seed=42,manual_prob=False,orig_probs=None,dirchlet_seed=0)
data = dec_to_bin(coded_data,n_features=sequence_length,k=vocab_size)

path = 'Model/Synthetic/Exp3/autoregressive/'
find_min_test_loss(data, path, 5, vocab_size=vocab_size, sequence_length=sequence_length, disc_layer_type='autoreg', hidden_layer=64)

path = 'Model/Synthetic/Exp3/bipartite/'
find_min_test_loss(data, path, 5, vocab_size=vocab_size, sequence_length=sequence_length, disc_layer_type='bipartite', hidden_layer=64, id=False, beta=16)

#exp4
print('exp4')
batch_size, sequence_length, vocab_size, k_fold = 1024, 10, 5, 5
orig_probs, coded_data = create_syn_data_paper(alpha=1,k=vocab_size,d=sequence_length,n_samples=10000,rnd_seed=42,manual_prob=False,orig_probs=None,dirchlet_seed=0)
data = dec_to_bin(coded_data,n_features=sequence_length,k=vocab_size)

path = 'Model/Synthetic/Exp4/autoregressive/'
find_min_test_loss(data, path, 5, vocab_size=vocab_size, sequence_length=sequence_length, disc_layer_type='autoreg', hidden_layer=64)

path = 'Model/Synthetic/Exp4/bipartite/'
find_min_test_loss(data, path, 5, vocab_size=vocab_size, sequence_length=sequence_length, disc_layer_type='bipartite', hidden_layer=64, id=False, beta=16)

#exp5
print('exp5')
batch_size, sequence_length, vocab_size, k_fold = 1024, 5, 10, 5
orig_probs, coded_data = create_syn_data_paper(alpha=1,k=vocab_size,d=sequence_length,n_samples=10000,rnd_seed=42,manual_prob=False,orig_probs=None,dirchlet_seed=0)
data = dec_to_bin(coded_data,n_features=sequence_length,k=vocab_size)

path = 'Model/Synthetic/Exp5/autoregressive/'
find_min_test_loss(data, path, 5, vocab_size=vocab_size, sequence_length=sequence_length, disc_layer_type='autoreg', hidden_layer=64)

path = 'Model/Synthetic/Exp5/bipartite/'
find_min_test_loss(data, path, 5, vocab_size=vocab_size, sequence_length=sequence_length, disc_layer_type='bipartite', hidden_layer=64, id=False, beta=16)

#exp6
print('exp6')
batch_size, sequence_length, vocab_size, k_fold = 1024, 4, 2, 5
data = np.load('coup_data_4_2_strong_corr.npy')

path = 'Model/Copula/Exp6/autoregressive/'
find_min_test_loss(data, path, 5, vocab_size=vocab_size, sequence_length=sequence_length, disc_layer_type='autoreg', hidden_layer=64)

path = 'Model/Copula/Exp6/bipartite/'
find_min_test_loss(data, path, 5, vocab_size=vocab_size, sequence_length=sequence_length, disc_layer_type='bipartite', hidden_layer=64, id=False, beta=8)

#exp7
print('exp7')
data = np.load('coup_data_4_2_moderate_corr.npy')

path = 'Model/Copula/Exp7/autoregressive/'
find_min_test_loss(data, path, 5, vocab_size=vocab_size, sequence_length=sequence_length, disc_layer_type='autoreg', hidden_layer=64)

path = 'Model/Copula/Exp7/bipartite/'
find_min_test_loss(data, path, 5, vocab_size=vocab_size, sequence_length=sequence_length, disc_layer_type='bipartite', hidden_layer=64, id=False, beta=8)

#exp8
print('exp8')
data = np.load('coup_data_4_2_weak_corr.npy')

path = 'Model/Copula/Exp8/autoregressive/'
find_min_test_loss(data, path, 5, vocab_size=vocab_size, sequence_length=sequence_length, disc_layer_type='autoreg', hidden_layer=64)

path = 'Model/Copula/Exp8/bipartite/'
find_min_test_loss(data, path, 5, vocab_size=vocab_size, sequence_length=sequence_length, disc_layer_type='bipartite', hidden_layer=64, id=False, beta=8)

#exp9
print('exp9')
data = np.load('coup_data_4_2_no_corr.npy')

path = 'Model/Copula/Exp9/autoregressive/'
find_min_test_loss(data, path, 5, vocab_size=vocab_size, sequence_length=sequence_length, disc_layer_type='autoreg', hidden_layer=64)

path = 'Model/Copula/Exp9/bipartite/'
find_min_test_loss(data, path, 5, vocab_size=vocab_size, sequence_length=sequence_length, disc_layer_type='bipartite', hidden_layer=64, id=False, beta=8)
'''

batch_size, sequence_length, vocab_size, k_fold = 1024, 22, 12, 5
data = process_mushroom_data('agaricus-lepiota.data')
path = batch_size, sequence_length, vocab_size, k_fold = 1024, 22, 12, 5
path = 'Model/Mushroom/Autoregressive/'
find_min_test_loss(data, path, 5, vocab_size=vocab_size, sequence_length=sequence_length, disc_layer_type='autoreg', hidden_layer=128)

path = 'Model/Mushroom/Bipartite/Non_Identity/Encoder_Layer/alpha_2/'
find_min_test_loss(data, path, 5, vocab_size=vocab_size, sequence_length=sequence_length, disc_layer_type='bipartite', id=False, alpha=2, beta=8)

path = 'Model/Mushroom/Bipartite/Non_Identity/Encoder_Layer/alpha_4/'
find_min_test_loss(data, path, 5, vocab_size=vocab_size, sequence_length=sequence_length, disc_layer_type='bipartite', id=False, alpha=4, beta=8)

path = 'Model/Mushroom/Bipartite/Non_Identity/Straight_Layer/beta_2/'
find_min_test_loss(data, path, 5, vocab_size=vocab_size, sequence_length=sequence_length, disc_layer_type='bipartite', id=False, beta=2, alpha=1)

path = 'Model/Mushroom/Bipartite/Non_Identity/Straight_Layer/beta_4/'
find_min_test_loss(data, path, 5, vocab_size=vocab_size, sequence_length=sequence_length, disc_layer_type='bipartite', id=False, beta=4, alpha=1)

path = 'Model/Mushroom/Bipartite/Non_Identity/Straight_Layer/beta_8/'
find_min_test_loss(data, path, 5, vocab_size=vocab_size, sequence_length=sequence_length, disc_layer_type='bipartite', id=False, beta=8, alpha=1)