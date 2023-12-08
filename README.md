# MucLiPred 

Welcome to the MucLiPred repository.

## Directory Structure

The repository is organized into several directories, each serving a distinct purpose in the model development and deployment process.

### `/MucLiPred/train`

- `protBert_main.py`: Contains the code for training the model. This script is the entry point for initiating the training process.
- `/other`: This directory includes various experimental code snippets that have been used during the development and testing of the model.

### `/MucLiPred/model`

- `prot_bert.py`: Houses the code defining the architecture of the model. This script details the layers and structure of the neural network used for predictions.

### `/MucLiPred/data`

- This directory contains datasets utilized for training and testing the model. Ensure to follow any licensing or usage restrictions associated with the data.

### `/MucLiPred/configuration`

- `config.py`: Stores various configuration parameters used during training. Parameters such as learning rate, number of epochs, and other hyperparameters can be adjusted here.

### `/MucLiPred/prot_bert_bfd`

- This directory contains the pretrained models. 

## Getting Started

To get started with MucLiPred, clone the repository and install the required dependencies:

```bash
git clone https://github.com/sethzhangjs/MucLiPred
