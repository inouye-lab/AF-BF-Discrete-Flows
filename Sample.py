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

import discrete_flows.disc_utils as disc_utils
from discrete_flows.made import MADE
from discrete_flows.mlp import MLP
from discrete_flows.embed import EmbeddingLayer
from discrete_flows.disc_models import *

from preprocess import *
from train_utils import *

def Sample_Model(model, base_log_probs, sample_row=2, sample_col=10, CNN=False, dim=(14, 14)):
  #Sample Pior
  plt.gray()
  prior = torch.distributions.OneHotCategorical(logits=base_log_probs)
  base = prior.sample([sample_row * sample_col])
  print(type(base))
  #Inverse model
  model.eval()
  if CNN:
    base = base.view((base.shape[0], -1, dim[0], dim[1], vocab_size))
  data = model.reverse(base)
  print(data.shape)
  #Removes one hot
  sample = torch.argmax(data, dim=-1)
  if CNN:
    sample = unsqueeze(sample, 1, 28, 28)
  print(sample.shape)

  fig, axs = plt.subplots(sample_row, sample_col, figsize=(28, 28))

  #fig.tight_layout()

  fig.subplots_adjust(hspace = 0.5, wspace = 0.005)

  sample = sample.cpu().detach().numpy()

  for i in range(sample_row):
    for j in range(sample_col):
      im = sample[sample_col*i+j].reshape(28,28)
      #im = np.hstack((im, sample[sample_col*i+j].reshape(28,28)))
      axs[i, j].imshow(im)
    if i==0:
      full_im = im
    else:
      full_im = np.vstack((full_im, im))

  #np.save('MNIST_' + str(disc_layer_type) + '_' + 'all' + '.npy', full_im)
  #plt.imshow(full_im)
  fig.savefig('MNIST_' + 'all')
  plt.gray()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sequence_length, vocab_size = 784, 2
num_flows = 6 # number of flow steps. This is different to the number of layers used inside each flow
temperature = 0.1 # used for the straight-through gradient estimator. Value taken from the paper
disc_layer_type = 'autoreg' #'autoreg' #'bipartite'
batch_size = 500
model = disc_flow_param(num_flows, temperature, vocab_size, sequence_length, batch_size, disc_layer_type, CNN=False, af='linear', beta = 1/16, hid_lay=28*28*2, id=False, )
print(model)
load_model = torch.load('MNIST/Autoregressive/k3_0.pt', map_location='cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(load_model['model_state_dict'])
base_log_probs = load_model['prior']

model.to(device)
base_log_probs.to(device)

plt.gray()
Sample_Model(model, base_log_probs, 5, 5, CNN=False)