import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 224

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 25

# Size of feature maps in generator
ngf = 20

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(nz, 7040),
            nn.BatchNorm2d(7040),
            nn.ReLU(True),
            transforms.Resize((4,11,ngf * 8)),
            nn.ConvTranspose2d(ngf * 8, 80, 5, 2, 2),
            nn.BatchNorm2d(80),
            nn.ReLU(True),
            nn.Conv2d(80, 40, 5, 2, 2),
            nn.BatchNorm2d(40),
            nn.ReLU(True),
            nn.Conv2d(40, 20, 5, 2, 2),
            nn.BatchNorm2d(20),
            nn.ReLU(True),
            nn.Conv2d(20, 3, 5, 2, 2),
            nn.Tanh(True)
        )