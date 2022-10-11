import sklearn.datasets as datasets

import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd

import torch
import tensorflow_datasets as tfds

def sample_quantized_gaussian_mixture2D(batch_size):
    """Samples data from a 2D quantized mixture of Gaussians.
    This is a quantized version of the mixture of Gaussians experiment from the
    Unrolled GANS paper (Metz et al., 2017).
    Args:
        batch_size: The total number of observations.
    Returns:
        Tensor with shape `[batch_size, 2]`, where each entry is in
            `{0, 1, ..., max_quantized_value - 1}`, a rounded sample from a mixture
            of Gaussians.
    """
    clusters = np.array([[2., 0.], [np.sqrt(2), np.sqrt(2)],
                                             [0., 2.], [-np.sqrt(2), np.sqrt(2)],
                                             [-2., 0.], [-np.sqrt(2), -np.sqrt(2)],
                                             [0., -2.], [np.sqrt(2), -np.sqrt(2)]])
    assignments = torch.distributions.OneHotCategorical(
            logits=torch.zeros(8, dtype = torch.float32)).sample([batch_size])
    means = torch.matmul(assignments, torch.from_numpy(clusters).float())

    samples = torch.distributions.normal.Normal(loc=means, scale=0.1).sample()
    clipped_samples = torch.clamp(samples, -2.25, 2.25)
    quantized_samples = (torch.round(clipped_samples * 20) + 45).long()
    return quantized_samples

def binarize_digits():
    digit = datasets.load_digits(10, True)
    data = []
    for i in range(10):
        data.append(np.zeros((len(np.where(digit[1] == i)[0]), 64)))
    for i in range(10):
        data[i] = digit[0][np.where(digit[1] == i)[0]]

    # Binarize Data
    for i in range(10):
        data[i] = np.where(data[i] >= 10, 1, 0)
    return data


def all_binarize_digits():
    digit = datasets.load_digits(10, True)[0]
    data = np.where(digit >= 10, 1, 0)
    return data

def download_mnist():
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                          transform=torchvision.transforms.Compose(
                                              [torchvision.transforms.ToTensor()]))
    label = torch.zeros(60000)
    data = torch.zeros((60000, 28, 28))

    for i in range(60000):
        a, b = trainset[i]
        data[i] = a
        label[i] = b
        if (i % 1000 == 0):
            print(i)

    for i in range(10):
        print('step' + str(i))
        loc = torch.where(label == i)[0]
        store = torch.zeros((loc.size()[0], 28, 28))
        print(store.size())
        count = 0
        for j in loc:
            store[count] = data[j.numpy()]
            count += 1
        name = 'mnist' + str(i) + '.pt'
        torch.save(store, name)


def binarize_MNIST(digit):
  if digit == 'all':
    for i in range(10):
      if i == 0:
        data = torch.load('mnist' + str(i) + '.pt')
      else:
        temp = torch.load('mnist' + str(i) + '.pt')
        data = torch.vstack((data, temp))
  else:
    data = torch.load('mnist' + str(digit) + '.pt')
  data = np.where(data.view(data.shape[0], -1) >= 0.5, 1, 0)

  return data

class Digits(Dataset):

    def __init__(self, num):
        self.data = binarize_digits()[num]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class K_fold_Digits(Dataset):

    def __init__(self, k, fold_num):
        data = all_binarize_digits()
        self.data, _ = kfold_splitter(data, data.shape[0], k, fold_num)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class MNIST(Dataset):
    def __init__(self, digit):
        self.digit = digit
        self.data = binarize_MNIST(digit)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def Sample_Model(model, base_log_probs, vocab_size, disc_layer_type, sample_row=2, sample_col=10, CNN=False, dim=(28, 14)):
  #Sample Pior
  prior = torch.distributions.OneHotCategorical(logits=base_log_probs)
  base = prior.sample([sample_row * sample_col])
  #Inverse model
  if CNN:
    base = base.view((base.shape[0], -1, dim[0], dim[1], vocab_size))
  data = model.reverse(base)
  print(data.shape)
  #Removes one hot
  sample = torch.argmax(data, dim=-1)
  print(sample.shape)

  sample = sample.cpu().detach().numpy()

  for i in range(sample_row):
    for j in range(sample_col):
      if j==0:
        im = sample[10*i+j].reshape(28,28)
      else:
        im = np.hstack((im, sample[10*i+j].reshape(28,28)))
    if i==0:
      full_im = im
    else:
      full_im = np.vstack((full_im, im))

  np.save('MNIST_' + str(disc_layer_type) + '_' + 'all' + '.npy', full_im)
  plt.imshow(full_im)
  plt.savefig('MNIST_' + 'all', dpi=1000)
  plt.gray()
  plt.show()


def preprocess_binary_mnist():
  train,test = tfds.load('binarized_mnist', split=['train', 'test'])
  new_train = tfds.as_numpy(train)
  new_test = tfds.as_numpy(test)
  flattened_images = []
  for i,ex in enumerate(new_train):
    # `{'image': np.array(shape=(28, 28, 1)), 'labels': np.array(shape=())}`
    flattened_images.append(ex['image'].flatten())
  for i,ex in enumerate(new_test):
    # `{'image': np.array(shape=(28, 28, 1)), 'labels': np.array(shape=())}`
    flattened_images.append(ex['image'].flatten())
  return np.array(flattened_images).astype(int)


def Mai_kfold_splitter(n, k, fold_num, train_proportion):
    if k != 1:  # for kfold = 1 just use the train_proportion to split
        fold_size = n // k
        test_inds = list(range(fold_num * fold_size, (fold_num + 1) * fold_size))
        set1 = set(test_inds)
        set2 = set(list(range(0, n)))
        train_inds = list(set2 - set1)
    else:
        train_inds = list(range(0, round(n * train_proportion)))
        test_inds = list(range(round(n * train_proportion), n))
    return train_inds, test_inds


def Mai_create_X_train_test(X, train_portion, kfolds, fold_num):
    '''
    If number of folds is 1, you need to specify the train_portion
    '''
    n, _ = X.shape

    train_ind_perm, test_ind_perm = Mai_kfold_splitter(n, kfolds, fold_num, train_portion)

    X_train = X[train_ind_perm, :]
    X_test = X[test_ind_perm, :]

    return X_train, X_test

def preprocess_805_snp_data(snps_805_path):
    df = pd.read_csv(snps_805_path, sep = ' ', header=None, compression='infer')
    df = df.sample(frac=1, random_state = 42).reset_index(drop=True)
    df_noname = df.drop(df.columns[0:2], axis=1)
    return df_noname.values

class DATA(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def kfold_splitter(data,n,k,fold_num):
    fold_size = n//k
    test_data = data[fold_num*fold_size:(fold_num+1)*fold_size]
    train_data = np.vstack((data[0:fold_num*fold_size], data[(fold_num+1)*fold_size:n]))
    return train_data, test_data

def get_all_k_length(k_values, d, answer):
    k_len = len(k_values)
    get_all_k_length_rec(k_values, [], k_len, d, answer)


def get_all_k_length_rec(k_values, prefix, k_len, d, answer):
    if (d == 0):
        answer.append(np.array(prefix))
        return

    for i in range(k_len):
        newPrefix = prefix + [k_values[i]]
        get_all_k_length_rec(k_values, newPrefix, k_len, d - 1, answer)


def get_possible_k(k, d):
    answer = []
    k_values = list(range(0, k))
    get_all_k_length(k_values, d, answer)
    return np.array(answer)


def create_syn_data_paper(alpha, k, d, n_samples, rnd_seed, manual_prob, orig_probs, dirchlet_seed):
    if (manual_prob == False):  # if manual prob is False ignore the passed original probs,
        # and generate new ones from a dirchlet distrbuition
        np.random.seed(dirchlet_seed)
        orig_probs = np.random.dirichlet(np.ones(k ** d) * alpha)
    np.random.seed(rnd_seed)  # seed for random data
    X_train_coded = np.random.choice(k ** d, size=n_samples,
                                     p=orig_probs)  # for k=2 and d=2, X_train_coded = {0,1,2,3,2,3,0,...}
    return orig_probs, X_train_coded


def dec_to_bin(data, n_features, k):
    possible_bin_values = get_possible_k(k, n_features)
    result = possible_bin_values[data]
    return np.array(result)


mushroom_data_path = '/content/agaricus-lepiota.data'


def process_mushroom_data(mushroom_data_path):
    mushroom_data = pd.read_csv(mushroom_data_path, header=None)
    # print(mushroom_data.head())

    for col_id in mushroom_data.columns:
        # print(col_id)
        unique_col_data = np.unique(mushroom_data.iloc[:, col_id])
        # print(unique_col_data)
        mapper = {}
        for i, d in enumerate(unique_col_data):
            mapper[d] = i
        mushroom_data.iloc[:, col_id] = [mapper[x] for x in mushroom_data.iloc[:, col_id]]
        unique_col_data = np.unique(mushroom_data.iloc[:, col_id])
        # print(unique_col_data)
    mushroom_data = mushroom_data.drop([11], axis=1)  # 11 had missing data, so I dropped it
    # print(mushroom_data.head())
    # print(mushroom_data.dtypes)
    # print(len(mushroom_data))

    return mushroom_data.to_numpy()

def preprocess_city_scapes(city_scapes_path):
    return np.load(city_scapes_path).astype(int)
