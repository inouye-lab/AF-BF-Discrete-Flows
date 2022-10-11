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
from train_utils import *

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
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=int)
parse = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

    #exp1
if parse.exp == 1:
    train_toy_dataset('bipartite')
    train_toy_dataset('autoreg')
elif parse.exp == 2:
    #exp2
    Train_synth_data(k=5, d=5, n_samples=10000, disc_layer_type='bipartite', epoch=100, batch_size=1024, hidden_layer=0, path='exp2', save=True, test_per_epoch=True, af='linear', alpha=1, beta=16, id_init=False, k_fold=5, manual_prob=False)
    Train_synth_data(k=5, d=5, n_samples=10000, disc_layer_type='autoreg', epoch=100, batch_size=1024, hidden_layer=64, path='exp2', save=True, test_per_epoch=True, af='linear', alpha=1, beta=1, id_init=False, k_fold=5, manual_prob=False)
elif parse.exp == 3:
    #exp3
    Train_synth_data(k=5, d=10, n_samples=10000, disc_layer_type='bipartite', epoch=100, batch_size=1024, hidden_layer=0, path='exp3', save=True, test_per_epoch=True, af='linear', alpha=1, beta=16, id_init=False, k_fold=5, manual_prob=False)
    Train_synth_data(k=5, d=10, n_samples=10000, disc_layer_type='autoreg', epoch=100, batch_size=1024, hidden_layer=64, path='exp3', save=True, test_per_epoch=True, af='linear', alpha=1, beta=1, id_init=False, k_fold=5, manual_prob=False)
elif parse.exp == 4:
    #exp4
    Train_synth_data(k=10, d=5, n_samples=10000, disc_layer_type='bipartite', epoch=100, batch_size=1024, hidden_layer=0, path='exp4', save=True, test_per_epoch=True, af='linear', alpha=1, beta=16, id_init=False, k_fold=5, manual_prob=False)
    Train_synth_data(k=10, d=5, n_samples=10000, disc_layer_type='autoreg', epoch=100, batch_size=1024, hidden_layer=64, path='exp4', save=True, test_per_epoch=True, af='linear', alpha=1, beta=1, id_init=False, k_fold=5, manual_prob=False)
elif parse.exp == 5:
    #exp6
    Train_copula(device, copula_data_path='../data/coup_data_4_2_strong_corr.npy', disc_layer_type='bipartite', epoch=100, path='exp5', alpha=1, beta=8, id_init=False)
    Train_copula(device, copula_data_path='../data/coup_data_4_2_strong_corr.npy', disc_layer_type='autoreg', epoch=100, hidden_layer=64, path='exp5', alpha=1, beta=16, id_init=False)
elif parse.exp == 6:
    #exp7
    Train_copula(device, copula_data_path='../data/coup_data_4_2_strong_corr.npy', disc_layer_type='bipartite', epoch=100, path='exp6', alpha=1, beta=8, id_init=False)
    Train_copula(device, copula_data_path='../data/coup_data_4_2_strong_corr.npy', disc_layer_type='autoreg', epoch=100, hidden_layer=64, path='exp6', alpha=1, beta=16, id_init=False)
elif parse.exp == 7:
    #exp8
    Train_copula(device, copula_data_path='../data/coup_data_4_2_strong_corr.npy', disc_layer_type='bipartite', epoch=100, path='exp7', alpha=1, beta=8, id_init=False)
    Train_copula(device, copula_data_path='../data/coup_data_4_2_strong_corr.npy', disc_layer_type='autoreg', epoch=100, hidden_layer=64, path='exp7', alpha=1, beta=16, id_init=False)
elif parse.exp == 8:
    #exp9
    Train_copula(device, copula_data_path='../data/coup_data_4_2_strong_corr.npy', disc_layer_type='bipartite', epoch=100, path='exp8', alpha=1, beta=8, id_init=False)
    Train_copula(device, copula_data_path='../data/coup_data_4_2_strong_corr.npy', disc_layer_type='autoreg', epoch=100, hidden_layer=64, path='exp8', alpha=1, beta=16, id_init=False)
elif parse.exp == 9:
    mushroom_data_path = '../data/agaricus-lepiota.data'

    #Train_mushroom(device, mushroom_data_path=mushroom_data_path, disc_layer_type='bipartite', epoch=100, path='id_exp10_a1b8', alpha=1, beta=8, id_init=True)
    Train_mushroom(device, mushroom_data_path=mushroom_data_path, disc_layer_type='bipartite', epoch=100, path='exp9_a1b8', alpha=1, beta=8, id_init=False)
    Train_mushroom(device, mushroom_data_path=mushroom_data_path, hidden_layer=128, disc_layer_type='autoreg', epoch=100, path='exp10', alpha=1, beta=1, id_init=False)

    #Train_mushroom(device, mushroom_data_path=mushroom_data_path, disc_layer_type='bipartite', epoch=100, path='id_exp10_a1b4', alpha=1, beta=4, id_init=True)
    Train_mushroom(device, mushroom_data_path=mushroom_data_path, disc_layer_type='bipartite', epoch=100, path='exp9_a1b4', alpha=1, beta=4, id_init=False)

    #Train_mushroom(device, mushroom_data_path=mushroom_data_path, disc_layer_type='bipartite', epoch=100, path='id_exp10_a1b2', alpha=1, beta=2, id_init=True)
    Train_mushroom(device, mushroom_data_path=mushroom_data_path, disc_layer_type='bipartite', epoch=100, path='exp9_a1b2', alpha=1, beta=2, id_init=False)

    #Train_mushroom(device, mushroom_data_path=mushroom_data_path, disc_layer_type='bipartite', epoch=100, path='id_exp10_a2b8', alpha=2, beta=8, id_init=True)
    Train_mushroom(device, mushroom_data_path=mushroom_data_path, disc_layer_type='bipartite', epoch=100, path='exp9_a2b8', alpha=2, beta=8, id_init=False)

    #Train_mushroom(device, mushroom_data_path=mushroom_data_path, disc_layer_type='bipartite', epoch=100, path='id_exp10_a4b8', alpha=4, beta=8, id_init=True)
    Train_mushroom(device, mushroom_data_path=mushroom_data_path, disc_layer_type='bipartite', epoch=100, path='exp9_a4b8', alpha=4, beta=8, id_init=False)

######## MNIST ##########
elif parse.exp == 10:
    for i in range(3):
        print('kfold' + str(i))
        # beta=1
        Train_MNIST(digit='all', disc_layer_type='autoreg', batch_size=500, epoch=200, hidden_layer=784 * 2,
                    temp_decay=1, lr_decay=1, path='auto_h8', save=True, image_process=False, CNN=False,
                    test_per_epoch=True, dim=(14, 14), sample_size=15, af='linear', alpha=1, beta=1, id_init=False,
                    kfold=i)
        # beta=1/4
        Train_MNIST(digit='all', disc_layer_type='bipartite', batch_size=500, epoch=200, hidden_layer=784, temp_decay=1,
                    lr_decay=1, path='IDa1b1_4', save=True, image_process=False, CNN=False, test_per_epoch=True,
                    dim=(14, 14), sample_size=15, af='linear', alpha=1, beta=1 / 4, id_init=True, kfold=i)
        # beta=1/16
        Train_MNIST(digit='all', disc_layer_type='bipartite', batch_size=500, epoch=200, hidden_layer=784, temp_decay=1,
                    lr_decay=1, path='IDa1b1_16', save=True, image_process=False, CNN=False, test_per_epoch=True,
                    dim=(14, 14), sample_size=15, af='linear', alpha=1, beta=1 / 16, id_init=True, kfold=i)

        # alpha=2
        Train_MNIST(digit='all', disc_layer_type='bipartite', batch_size=500, epoch=200, hidden_layer=784, temp_decay=1,
                    lr_decay=1, path='IDa2b1', save=True, image_process=False, CNN=False, test_per_epoch=True,
                    dim=(14, 14), sample_size=15, af='linear', alpha=2, beta=1, id_init=True, kfold=i)
        # alpha=4
        Train_MNIST(digit='all', disc_layer_type='bipartite', batch_size=500, epoch=200, hidden_layer=784, temp_decay=1,
                    lr_decay=1, path='IDa4b1', save=True, image_process=False, CNN=False, test_per_epoch=True,
                    dim=(14, 14), sample_size=15, af='linear', alpha=4, beta=1, id_init=True, kfold=i)
        # alpha=8
        Train_MNIST(digit='all', disc_layer_type='bipartite', batch_size=500, epoch=200, hidden_layer=784, temp_decay=1,
                    lr_decay=1, path='IDa8b1', save=True, image_process=False, CNN=False, test_per_epoch=True,
                    dim=(14, 14), sample_size=15, af='linear', alpha=8, beta=1, id_init=True, kfold=i)

        # beta=1
        Train_MNIST(digit='all', disc_layer_type='bipartite', batch_size=500, epoch=200, hidden_layer=784, temp_decay=1,
                    lr_decay=1, path='a1b1', save=True, image_process=False, CNN=False, test_per_epoch=True,
                    dim=(14, 14), sample_size=15, af='linear', alpha=1, beta=1, id_init=False, kfold=i)
        # beta=1/4
        Train_MNIST(digit='all', disc_layer_type='bipartite', batch_size=500, epoch=200, hidden_layer=784, temp_decay=1,
                    lr_decay=1, path='a1b1_4', save=True, image_process=False, CNN=False, test_per_epoch=True,
                    dim=(14, 14), sample_size=15, af='linear', alpha=1, beta=1 / 4, id_init=False, kfold=i)
        # beta=1/16
        Train_MNIST(digit='all', disc_layer_type='bipartite', batch_size=500, epoch=200, hidden_layer=784, temp_decay=1,
                    lr_decay=1, path='a1b1_16', save=True, image_process=False, CNN=False, test_per_epoch=True,
                    dim=(14, 14), sample_size=15, af='linear', alpha=1, beta=1 / 16, id_init=False, kfold=i)

        # alpha=2
        Train_MNIST(digit='all', disc_layer_type='bipartite', batch_size=500, epoch=200, hidden_layer=784, temp_decay=1,
                    lr_decay=1, path='a2b1', save=True, image_process=False, CNN=False, test_per_epoch=True,
                    dim=(14, 14), sample_size=15, af='linear', alpha=2, beta=1, id_init=False, kfold=i)
        # alpha=4
        Train_MNIST(digit='all', disc_layer_type='bipartite', batch_size=500, epoch=200, hidden_layer=784, temp_decay=1,
                    lr_decay=1, path='a4b1', save=True, image_process=False, CNN=False, test_per_epoch=True,
                    dim=(14, 14), sample_size=15, af='linear', alpha=4, beta=1, id_init=False, kfold=i)
        # alpha=8
        Train_MNIST(digit='all', disc_layer_type='bipartite', batch_size=500, epoch=200, hidden_layer=784, temp_decay=1,
                    lr_decay=1, path='a8b1', save=True, image_process=False, CNN=False, test_per_epoch=True,
                    dim=(14, 14), sample_size=15, af='linear', alpha=8, beta=1, id_init=False, kfold=i)

######## Genetic Test ##########
elif parse.exp == 11:
    for i in range(3):
        print('kfold' + str(i))
        # beta=1
        Train_805(disc_layer_type='bipartite', batch_size=500, epoch=200, hidden_layer=805*2, path='IDa1b1', save=True, test_per_epoch=True, sample_size=15, af='linear', alpha=1, beta=1, id_init=True, kfold=i)
        # beta=1/4
        Train_805(disc_layer_type='bipartite', batch_size=500, epoch=200, hidden_layer=805*2, path='IDa1b1_4', save=True, test_per_epoch=True, sample_size=15, af='linear', alpha=1, beta=1/4, id_init=True, kfold=i)
        # beta=1/16
        Train_805(disc_layer_type='bipartite', batch_size=500, epoch=200, hidden_layer=805*2,  path='IDa1b1_16', save=True, test_per_epoch=True, sample_size=15, af='linear', alpha=1, beta=1/16, id_init=True, kfold=i)

        # alpha=2
        Train_805(disc_layer_type='bipartite', batch_size=500, epoch=200, hidden_layer=805*2, path='IDa2b1', save=True, test_per_epoch=True, sample_size=15, af='linear', alpha=2, beta=1, id_init=True, kfold=i)
        # alpha=4
        Train_805(disc_layer_type='bipartite', batch_size=500, epoch=200, hidden_layer=805*2, path='IDa4b1', save=True, test_per_epoch=True, sample_size=15, af='linear', alpha=4, beta=1, id_init=True, kfold=i)
        # alpha=8
        Train_805(disc_layer_type='bipartite', batch_size=500, epoch=200, hidden_layer=805*2, path='IDa8b1', save=True, test_per_epoch=True, sample_size=15, af='linear', alpha=8, beta=1, id_init=True, kfold=i)

        # beta=1
        Train_805(disc_layer_type='bipartite', batch_size=500, epoch=200, hidden_layer=805*2, path='a1b1', save=True, test_per_epoch=True, sample_size=15, af='linear', alpha=1, beta=1, id_init=False, kfold=i)
        # beta=1/4
        Train_805(disc_layer_type='bipartite', batch_size=500, epoch=200, hidden_layer=805*2, path='a1b1_4', save=True, test_per_epoch=True, sample_size=15, af='linear', alpha=1, beta=1/4, id_init=False, kfold=i)
        # beta=1/16
        Train_805(disc_layer_type='bipartite', batch_size=500, epoch=200, hidden_layer=805*2, path='a1b1_16', save=True, test_per_epoch=True, sample_size=15, af='linear', alpha=1, beta=1/16, id_init=False, kfold=i)

        # alpha=2
        Train_805(disc_layer_type='bipartite', batch_size=500, epoch=200, hidden_layer=805*2, path='a2b1', save=True, test_per_epoch=True, sample_size=15, af='linear', alpha=2, beta=1, id_init=False, kfold=i)
        # alpha=4
        Train_805(disc_layer_type='bipartite', batch_size=500, epoch=200, hidden_layer=805*2, path='a4b1', save=True, test_per_epoch=True, sample_size=15, af='linear', alpha=4, beta=1, id_init=False, kfold=i)
        # alpha=8
        Train_805(disc_layer_type='bipartite', batch_size=500, epoch=200, hidden_layer=805*2, path='a8b1', save=True, test_per_epoch=True, sample_size=15, af='linear', alpha=8, beta=1, id_init=False, kfold=i)

# City Scapes
elif parse.exp == 12:
    Train_cityscapes(disc_layer_type='autoreg', batch_size=150, epoch=200, hidden_layer=512*2, path='autoreg',
                     save=True, test_per_epoch=True, af='linear', alpha=1, beta=1, id_init=False)
    # beta=1
    Train_cityscapes(disc_layer_type='bipartite', batch_size=150, epoch=200, hidden_layer=805 * 2, path='a1b1',
                     save=True, test_per_epoch=True, af='linear', alpha=1, beta=1, id_init=False)
    # beta=1/4
    Train_cityscapes(disc_layer_type='bipartite', batch_size=150, epoch=200, hidden_layer=805 * 2, path='a1b1_4',
                     save=True, test_per_epoch=True, af='linear', alpha=1, beta=1 / 4, id_init=False)
    # beta=1/16
    Train_cityscapes(disc_layer_type='bipartite', batch_size=150, epoch=200, hidden_layer=805 * 2, path='a1b1_16',
                     save=True, test_per_epoch=True, af='linear', alpha=1, beta=1 / 16, id_init=False)

    # alpha=2
    Train_cityscapes(disc_layer_type='bipartite', batch_size=150, epoch=200, hidden_layer=805 * 2, path='a2b1',
                     save=True, test_per_epoch=True, af='linear', alpha=2, beta=1, id_init=False)
    # alpha=4
    Train_cityscapes(disc_layer_type='bipartite', batch_size=150, epoch=200, hidden_layer=805 * 2, path='a4b1',
                     save=True, test_per_epoch=True, af='linear', alpha=4, beta=1, id_init=False)
    # alpha=8
    Train_cityscapes(disc_layer_type='bipartite', batch_size=150, epoch=200, hidden_layer=805 * 2, path='a8b1',
                     save=True, test_per_epoch=True, af='linear', alpha=8, beta=1, id_init=False)