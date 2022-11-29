from preprocess import *
from flow_functions import *


import numpy as np
import matplotlib.pyplot as plt

import torch


def Train_MNIST(digit, disc_layer_type, batch_size, epoch, hidden_layer=0, temp_decay=0.83, lr_decay=0.97, path='',
                save=False, image_process=False, CNN=False, test_per_epoch=True, dim=None, sample_size=None, af=None,
                alpha=1, beta=1, id_init=True, kfold=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')

    train_data, test_data = Mai_create_X_train_test(preprocess_binary_mnist(), 4 / 5, 3, kfold)
    print(train_data.shape)
    mnist = DATA(train_data)
    MNIST_loader = torch.utils.data.DataLoader(dataset=mnist,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0)
    sequence_length, vocab_size = 784, 2
    num_flows = 6  # number of flow steps. This is different to the number of layers used inside each flow
    temperature = 0.1  # used for the straight-through gradient estimator. Value taken from the paper
    disc_layer_type = disc_layer_type  # 'autoreg' #'bipartite'
    batch_size = batch_size

    train_store_time = []
    train_store_min_loss = []

    # Training
    if id_init:
        train_data = torch.from_numpy(train_data).type(torch.int64)
        base_log_probs = init_prior(train_data, vocab_size, sequence_length)
        model = disc_flow_param(num_flows, temperature, vocab_size, sequence_length, batch_size, disc_layer_type,
                                hidden_layer, CNN=CNN, af=af, alpha=alpha, beta=beta, id=True)
    else:
        base_log_probs = create_base_prior(sequence_length, vocab_size)
        model = disc_flow_param(num_flows, temperature, vocab_size, sequence_length, batch_size, disc_layer_type,
                                hidden_layer, CNN=CNN, af=af, alpha=alpha, beta=beta, id=False)
    model = model.to(device)
    print(model)

    loss, final_time, load_path, model, base_log_probs = train_disc_flow(device=device, model=model, data=MNIST_loader,
                                                  base_log_probs=base_log_probs, vocab_size=vocab_size,
                                                  test_per_epoch=test_per_epoch, temp_decay=temp_decay,
                                                  lr_decay=lr_decay, save_path=str(path) + 'MNIST',
                                                  k_fold='3', k_fold_idx=kfold, batch_size=batch_size,
                                                  epochs=epoch, learning_rate=0.01, dataloader=True, CNN=CNN, dim=dim,
                                                  save=save, test_data=test_data, update_temp=False,
                                                  data_size=train_data.shape[0])
    #train_store_min_loss.append(loss.cpu().clone().detach().numpy())
    #train_store_time.append(final_time)

    #print("Training Minimum Loss:")
    #print(train_store_min_loss)
    #print("Training Time:")
    #print(train_store_time)

    if image_process:
        # Sample Pior
        prior = torch.distributions.OneHotCategorical(logits=base_log_probs)
        base = prior.sample([sample_size]).to(device)
        # Inverse model
        model.eval()
        if CNN:
            base = base.view((base.shape[0], 4, 14, 14, vocab_size)).to(device)
            # base = F.one_hot(squeeze(torch.argmax(base, dim=-1), 4, 14, 14), -1)
        data = model.reverse(base)
        print(data.shape)
        # Removes one hot
        sample = torch.argmax(data, dim=-1)
        if CNN:
            sample = unsqueeze(sample, 1, 28, 28)
        print(sample.shape)

        sample = sample.cpu().detach().numpy()

        for i in range(sample_size):
            if i == 0:
                im = sample[i].reshape(28, 28)
            else:
                im = np.hstack((im, sample[i].reshape(28, 28)))

        np.save(str(path) + 'MNIST_' + str(disc_layer_type) + '_' + str(digit) + '.npy', im)
        plt.imshow(im)
        plt.show()


def Train_805(disc_layer_type, batch_size, epoch, hidden_layer=0, path='', save=True, test_per_epoch=True,
              alpha=1, beta=1, id_init=True, sample_size=None, af=None, kfold=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    train_data, test_data = Mai_create_X_train_test(
        preprocess_805_snp_data('805_SNP_1000G_real.hapt.zip'), 4 / 5, 3, kfold)

    snp805 = DATA(train_data)
    snp805_loader = torch.utils.data.DataLoader(dataset=snp805,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=0)
    sequence_length, vocab_size = 805, 2
    num_flows = 6  # number of flow steps. This is different to the number of layers used inside each flow
    temperature = 0.1  # used for the straight-through gradient estimator. Value taken from the paper
    disc_layer_type = disc_layer_type  # 'autoreg' #'bipartite'
    batch_size = batch_size

    train_store_time = []
    train_store_min_loss = []

    if id_init:
        train_data = torch.from_numpy(train_data).type(torch.int64)
        base_log_probs = init_prior(train_data, vocab_size, sequence_length)
        model = disc_flow_param(num_flows, temperature, vocab_size, sequence_length, batch_size, disc_layer_type,
                                hidden_layer, CNN=False, af=af, alpha=alpha, beta=beta, id=True)
    else:
        base_log_probs = create_base_prior(sequence_length, vocab_size)
        model = disc_flow_param(num_flows, temperature, vocab_size, sequence_length, batch_size, disc_layer_type,
                                hidden_layer, CNN=False, af=af, alpha=alpha, beta=beta, id=False)

    model = model.to(device)
    print(model)

    loss, final_time, load_path, model, base_log_prob = train_disc_flow(device=device, model=model, data=snp805_loader,
                                                                        base_log_probs=base_log_probs,
                                                                        vocab_size=vocab_size,
                                                                        test_per_epoch=test_per_epoch, temp_decay=1,
                                                                        lr_decay=1, save_path=str(path) + '805',
                                                                        k_fold=3, k_fold_idx=kfold,
                                                                        batch_size=batch_size, epochs=epoch,
                                                                        learning_rate=0.01, dataloader=True, CNN=False,
                                                                        save=save, test_data=test_data,
                                                                        update_temp=False,
                                                                        data_size=train_data.shape[0])
    # train_store_min_loss.append(loss.cpu().clone().detach().numpy())
    # train_store_time.append(final_time)


def Train_synth_data(k, d, n_samples, disc_layer_type, batch_size, epoch, hidden_layer=0, manual_prob=True,
                     orig_probs=[1/3, 1/6, 1/6, 1/3], path='', save=True, test_per_epoch=True, af='linear',
                     alpha=1, beta=1, id_init=True, k_fold=5):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    num_flows = 4  # number of flow steps. This is different to the number of layers used inside each flow
    temperature = 0.1  # used for the straight-through gradient estimator. Value taken from the paper

    train_store_time = []
    train_store_min_loss = []
    test_store_time = []
    test_store_loss = []

    orig_probs, coded_data = create_syn_data_paper(alpha=1, k=k, d=d, n_samples=n_samples, rnd_seed=42,
                                                   manual_prob=manual_prob, orig_probs=orig_probs,
                                                   dirchlet_seed=0)
    data = dec_to_bin(coded_data, n_features=d, k=k)
    for k_fold_idx in range(k_fold):
        print(k_fold, '-fold: ', k_fold_idx + 1)
        train_data, test_data = kfold_splitter(data=data, n=n_samples, k=k_fold, fold_num=k_fold_idx)
        print(train_data.shape)

        synth = DATA(train_data)
        synth_loader = torch.utils.data.DataLoader(dataset=synth,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=0)
        vocab_size = k
        sequence_length = d
        # Training
        if id_init:
            train_data = torch.from_numpy(train_data)
            base_log_probs = init_prior(train_data, vocab_size, sequence_length)
            model = disc_flow_param(num_flows, temperature, vocab_size, sequence_length, batch_size,
                                    disc_layer_type, hidden_layer, CNN=False, af=af, alpha=alpha, beta=beta, id=True)
        else:
            base_log_probs = create_base_prior(sequence_length, vocab_size)
            model = disc_flow_param(num_flows, temperature, vocab_size, sequence_length, batch_size,
                                    disc_layer_type, hidden_layer, CNN=False, af=af, alpha=alpha, beta=beta, id=False)
        model = model.to(device)
        print(model)

        loss, final_time, load_path, model, base_log_prob = train_disc_flow(device=device, model=model, data=synth_loader,
                                                      base_log_probs=base_log_probs, vocab_size=vocab_size,
                                                      test_per_epoch=test_per_epoch, temp_decay=1,
                                                      lr_decay=1, save_path=str(path) + '_' + str(disc_layer_type) + '_' + str(k_fold) + '_' + str(k_fold_idx),
                                                      k_fold=k_fold, k_fold_idx=k_fold_idx,
                                                      batch_size=batch_size, epochs=epoch, learning_rate=0.01,
                                                      dataloader=True, CNN=False, dim=None, save=save,
                                                      test_data=test_data, update_temp=False,
                                                      data_size=train_data.shape[0])
        #train_store_min_loss.append(loss.cpu().clone().detach().numpy())
        #train_store_time.append(final_time)
        #train_store_min_loss.append(loss.cpu().clone().detach().numpy())
        #train_store_time.append(final_time)


def Train_mushroom(device, mushroom_data_path, disc_layer_type, epoch, hidden_layer=0, path='', alpha=1, beta=1, af='linear', id_init=True):

        batch_size, sequence_length, vocab_size, k_fold = 1024, 22, 12, 5

        num_flows = 4 # number of flow steps. This is different to the number of layers used inside each flow
        temperature = 0.1 # used for the straight-through gradient estimator. Value taken from the paper

        train_store_time = []
        train_store_min_loss = []
        test_store_time = []
        test_store_loss = []
        data = process_mushroom_data(mushroom_data_path) #k=12, d=22

        sample_size = data.shape[0]
        for k_fold_idx in range(k_fold):
              print(k_fold, '-fold: ', k_fold_idx + 1)
              train_data, test_data = kfold_splitter(data=data, n=sample_size, k=k_fold, fold_num=k_fold_idx)
              print(train_data.shape)
              mush = DATA(train_data)
              mush_loader = torch.utils.data.DataLoader(dataset=mush,
                                                         batch_size=batch_size,
                                                         shuffle=True,
                                                         num_workers=0)
              # Training
              if id_init:
                  train_data = torch.from_numpy(train_data)
                  base_log_probs = init_prior(train_data, vocab_size, sequence_length)
                  model = disc_flow_param(num_flows, temperature, vocab_size, sequence_length, batch_size,
                                          disc_layer_type, hidden_layer, CNN=False, af=af, alpha=alpha, beta=beta, id=True)
              else:
                  base_log_probs = create_base_prior(sequence_length, vocab_size)
                  model = disc_flow_param(num_flows, temperature, vocab_size, sequence_length, batch_size,
                                          disc_layer_type, hidden_layer, CNN=False, af=af, alpha=alpha, beta=beta, id=False)

              model = model.to(device)

              loss, final_time, load_path, model, base_log_prob = train_disc_flow(device=device, model=model, data=mush_loader,
                                                                  base_log_probs=base_log_probs, vocab_size=vocab_size,
                                                                  test_per_epoch=True, temp_decay=1,
                                                                  lr_decay=1,
                                                                  save_path=str(path) + '_' + str(disc_layer_type) + '_' + str(k_fold) + '_' + str(k_fold_idx),
                                                                  k_fold=k_fold, k_fold_idx=k_fold_idx,
                                                                  batch_size=batch_size, epochs=epoch,
                                                                  learning_rate=0.01,
                                                                  dataloader=True, CNN=False, dim=None, save=True,
                                                                  test_data=test_data, update_temp=False,
                                                                  data_size=train_data.shape[0])

def Train_copula(device, copula_data_path, disc_layer_type, epoch, hidden_layer=0, path='', alpha=1, beta=1,
                   af='linear', id_init=True):

    batch_size, sequence_length, vocab_size, k_fold = 1024, 4, 2, 5

    num_flows = 4  # number of flow steps. This is different to the number of layers used inside each flow
    temperature = 0.1  # used for the straight-through gradient estimator. Value taken from the paper

    train_store_time = []
    train_store_min_loss = []
    test_store_time = []
    test_store_loss = []
    copula_data = np.load(copula_data_path)

    sample_size = copula_data.shape[0]
    for k_fold_idx in range(k_fold):
        print(k_fold, '-fold: ', k_fold_idx + 1)
        train_data, test_data = kfold_splitter(data=copula_data, n=sample_size, k=k_fold, fold_num=k_fold_idx)
        print(train_data.shape)
        data = DATA(train_data)
        data_loader = torch.utils.data.DataLoader(dataset=data,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=0)
        # Training
        if id_init:
            train_data = torch.from_numpy(train_data)
            base_log_probs = init_prior(train_data, vocab_size, sequence_length)
            model = disc_flow_param(num_flows, temperature, vocab_size, sequence_length, batch_size,
                                    disc_layer_type, hidden_layer, CNN=False, af=af, alpha=alpha, beta=beta,
                                    id=True)
        else:
            base_log_probs = create_base_prior(sequence_length, vocab_size)
            model = disc_flow_param(num_flows, temperature, vocab_size, sequence_length, batch_size,
                                    disc_layer_type, hidden_layer, CNN=False, af=af, alpha=alpha, beta=beta,
                                    id=False)

        model = model.to(device)
        loss, final_time, load_path, model, base_log_prob = train_disc_flow(device=device, model=model, data=data_loader,
                                                            base_log_probs=base_log_probs,
                                                            vocab_size=vocab_size,
                                                            test_per_epoch=True, temp_decay=1,
                                                            lr_decay=1,
                                                            save_path=str(path) + '_' + str(disc_layer_type) + '_' + str(k_fold) + '_' + str(k_fold_idx),
                                                            k_fold=k_fold, k_fold_idx=k_fold_idx,
                                                            batch_size=batch_size, epochs=epoch,
                                                            learning_rate=0.01,
                                                            dataloader=True, CNN=False, dim=None, save=True,
                                                            test_data=test_data, update_temp=False,
                                                            data_size=train_data.shape[0])



def train_toy_dataset(disc_layer_type):
    CNN = False
    af = 'linear'
    hidden_layer = 128
    batch_size, sequence_length, vocab_size = 250, 2, 91
    num_flows=4
    temperature = 0.1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = sample_quantized_gaussian_mixture2D(12000)
    for i in range(5):
      train_data, test_data = Mai_create_X_train_test(data, 4/5, 5, i)
      test_data = test_data.numpy()

      toy = DATA(data)
      toy_loader = torch.utils.data.DataLoader(dataset=toy,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=0)

      base_log_probs = create_base_prior(sequence_length, vocab_size)
      model = disc_flow_param(num_flows, temperature, vocab_size, sequence_length, batch_size, disc_layer_type, hidden_layer, beta=2, CNN=CNN, af=af, id=False)
      print(model)

      model = model.to(device)

      loss, final_time, load_path, model, base_log_prob = train_disc_flow(device=device, model=model, data=toy_loader,
                                                    base_log_probs=base_log_probs, vocab_size=vocab_size,
                                                    test_data=test_data, temp_decay=1, lr_decay=1,
                                                    save_path='exp1_' + str(disc_layer_type) + '_k5_' + str(i), k_fold=5, k_fold_idx=i,
                                                    batch_size=batch_size, epochs=100, learning_rate=0.01,
                                                    test_per_epoch=True, dataloader=True, CNN=False,
                                                    save=True, update_temp=False, data_size=12000*0.8)


def Train_cityscapes(disc_layer_type, batch_size, epoch, hidden_layer=0, path='', save=True, test_per_epoch=True,
                     af=None, alpha=1, beta=1, id_init=True, sample=False, sample_size=16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    for i in range(3):
        train_data, test_data = Mai_create_X_train_test(preprocess_city_scapes('city_scapes_32_64.npy'), 4 / 5, 3,
                                                        i)

        city = DATA(train_data)
        city_loader = torch.utils.data.DataLoader(dataset=city,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=0)
        sequence_length, vocab_size = 2048, 8
        num_flows = 6  # number of flow steps. This is different to the number of layers used inside each flow
        temperature = 0.1  # used for the straight-through gradient estimator. Value taken from the paper
        disc_layer_type = disc_layer_type  # 'autoreg' #'bipartite'
        batch_size = batch_size

        train_store_time = []
        train_store_min_loss = []

        # Training
        base_log_probs = create_base_prior(sequence_length, vocab_size)
        if id_init:
            train_data = torch.from_numpy(train_data)
            base_log_probs = init_prior(train_data, vocab_size, sequence_length)
            model = disc_flow_param(num_flows, temperature, vocab_size, sequence_length, batch_size,
                                    disc_layer_type, hidden_layer, CNN=False, af=af, alpha=alpha, beta=beta, id=True)
        else:
            base_log_probs = create_base_prior(sequence_length, vocab_size)
            model = disc_flow_param(num_flows, temperature, vocab_size, sequence_length, batch_size,
                                    disc_layer_type, hidden_layer, CNN=False, af=af, alpha=alpha, beta=beta, id=False)

        model = model.to(device)
        print(model)

        loss, final_time, load_path, model, base_log_prob = train_disc_flow(device=device, model=model, data=city_loader,
                                                                            base_log_probs=base_log_probs,
                                                                            vocab_size=vocab_size,
                                                                            test_per_epoch=test_per_epoch, temp_decay=1,
                                                                            lr_decay=1, save_path=str(path) + '_city_' + 'k3_' + str(i),
                                                                            k_fold=disc_layer_type, k_fold_idx='',
                                                                            batch_size=batch_size, epochs=epoch,
                                                                            learning_rate=0.01, dataloader=True, CNN=False,
                                                                            save=save, test_data=test_data,
                                                                            update_temp=False,
                                                                            data_size=train_data.shape[0])
        # train_store_min_loss.append(loss.cpu().clone().detach().numpy())
        # train_store_time.append(final_time)

        print("Training Minimum Loss:")
        print(train_store_min_loss)
        print("Training Time:")
        print(train_store_time)

        if sample:
            prior = torch.distributions.OneHotCategorical(logits=base_log_probs)
            base = prior.sample([sample_size]).to(device)
            base = base.view((base.shape[0], 1, 32, 64, vocab_size)).to(device)
            data = model.reverse(base)
            sample = torch.argmax(data, dim=-1)
            sample = sample.cpu().detach().numpy()
            for i in sample_size:
                plt.imsave('cityscapes_sample_' + str(i), sample[i][0])

