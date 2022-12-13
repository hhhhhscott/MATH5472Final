import torch
from torch import nn
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from torchvision import datasets,transforms
from torch.utils.data import Dataset, DataLoader
class config:
    input_dim = 784
    hidden_dim = 20
    trainable_simga = True
    sigma_init = None
    stochastic_ELBO = False

class LinearVAE(nn.Module):
    def __init__(self,config):
        super(LinearVAE,self).__init__()





if __name__ == '__main__':
    train_dataset= datasets.MNIST('./data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
    ]))
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

