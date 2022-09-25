import torch
import torch.nn as nn
from reshape import Reshape

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
            nn.BatchNorm1d(7040),
            nn.ReLU(True),
            Reshape(ngf * 8, 4,11),
            nn.ConvTranspose2d(ngf * 8, 80, 5, 2, 2, bias=False),
            nn.BatchNorm2d(80),
            nn.ReLU(True),
            nn.ConvTranspose2d(80, 40, 5, 2, 2, bias=False),
            nn.BatchNorm2d(40),
            nn.ReLU(True),
            nn.ConvTranspose2d(40, 20, 5, 2, 2, bias=False),
            nn.BatchNorm2d(20),
            nn.ReLU(True),
            nn.ConvTranspose2d(20, 3, 5, 2, 2, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)
