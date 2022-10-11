# Implementation_of_Discrete_Flows_Paper.ipynb

## Acknowledgements
Our implementation is based on this paper: https://arxiv.org/abs/1905.10347. 
This implementation is a slight modification of the code by Trenton Bricken taken from: https://github.com/TrentBrick/PyTorchDiscreteFlows/tree/master/discrete_flows (which was originally a modification from https://github.com/google/edward2/blob/master/edward2/tensorflow/layers/discrete_flows.py and https://github.com/google/edward2/blob/master/edward2/tensorflow/layers/utils.py)

## Modifications Done
https://github.com/TrentBrick/PyTorchDiscreteFlows/tree/master/discrete_flows code had runtime errors when applying the DiscreteBipartiteFlow class in https://github.com/TrentBrick/PyTorchDiscreteFlows/blob/master/discrete_flows/disc_models.py for dimensions larger than 2, and the code was not programmed to do odd dimensions. The embedding flow layer was removed (commonly used in NLP sequence data which is not the case for our experiments) and replaced with a single hidden layer (hidden layer with a ReLU activation function). The bipartite architecture was also adjusted to support odd dimensions for the DiscreteBipartiteFlow class.

## Key Notes
The code also had issues of using past versions of PyTorch functions (e.i. `torch.fft.fft()`) that were no longer compatible with the new version. Therefore, the version was downgraded to PyTorch 1.7.1. Moreover, the notebook had a `git` command line that directly downloaded the github repository `!git clone https://github.com/TrentBrick/PyTorchDiscreteFlows.git`.

## Running Experiment Code
To run the code, all codes in these sections must be run in these respective order (Fundamental Test section can be ignored):
- Setup
- Modified Discrete Bipartite Flow Model
  - Our modified DiscreteBipartiteFlow class.
- Synethetic Data / Functions
  - Own functions for generating synthetic data.
- Discrete Flow Functions
- Other Functions
- Synthetic Data Testing Bipartite
  - Experiments 1-5 for discrete bipartite flows.
- Synthetic Data Testing Autoregressive
  - Experiments 1-5 for discrete autoregressive flows.
- Mushroom Dataset Testing
  - Mushroom dataset experiments for both bipartite and autoregressive flows.
- Copula Data Testing
  - Experiments 6-9 for both bipartite and autoregressive flows.
- command line arguments
    - exp 1-8 : this automatically runs all the baseline experiments 1-8 (both AF and BF) on our paper.
        - each experiment saves a python model, loss, epoch test loss, average epoch training loss, epoch test time, and epoch training time.
    - exp 9 : this runs 6 different baseline configuration models for the mushroom dataset. 1 AF model a& 5 BF models (beta = 8, beta = 4, beta = 2, alpha = 2, alpha = 8).
    Again, each configuration saves a python model, loss, epoch test loss, average epoch training loss, epoch test time, and epoch training time.
    - exp 10 : this runs 12 different baseline configuration models for the binarized MNIST dataset. 1 AF model & 12 BF models (beta w/ id = 1, beta w/ id = 1/4, beta w/ id = 1/16, alpha w/ id = 2, alpha w/ id = 4, alpha w/ id = 8, beta w/o id = 1, beta w/o id = 1/4, beta w/o id = 1/16, alpha w/o id = 2, alpha w/o id = 4, alpha w/o id = 8)
    - exp 11 : this runs 12 different baseline configuration models for the 805 genetic dataset. 1 AF model & 12 BF models (beta w/ id = 1, beta w/ id = 1/4, beta w/ id = 1/16, alpha w/ id = 2, alpha w/ id = 4, alpha w/ id = 8, beta w/o id = 1, beta w/o id = 1/4, beta w/o id = 1/16, alpha w/o id = 2, alpha w/o id = 4, alpha w/o id = 8)
```console
# Below is an example of how the user would run an experiment on the terminal. 
# In this case this would run the 805 genetic datset.
python experiments.py --exp 11
```

## Output
The experment sections will output an average and standard deviation of the negative log-likelihood values and its computation time over a k-fold cross validation.

# Code Structure (Baseline models)
- experiments.py : main function for running all the baseline experiments.
- flow_functions.py : contains basic functions for the creating, testing, and training the experiments.
- preprocess.py : contains functions that loads in or creates data for the experiments. This file also includes a dataloader
and data processing functions.
- train_utils.py : contains all the experiment training functions.
- PyTorchDiscreteFlows: This was a downloaded discrete flow folder in github (https://github.com/TrentBrick/PyTorchDiscreteFlows.git). 
  - Modified files
      - PyTorchDiscreteFlows/discrete_flows/disc_models.py: a linear layer network class was added and the DiscreteBipartiteFlow class has been        completely revamped for our experiment.
      - PyTorchDiscreteFlows/discrete_flows/disc_utils.py: this python file was modified to run on cuda.
      - PyTorchDiscreteFlows/discrete_flows/made.py: this python file was modified to run on cuda.


