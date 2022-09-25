import torch
import torch.nn as nn
from reshape import Reshape

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 25

# Size of feature maps in discriminator
ndf = 20

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 20, 5, 2, 2, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(20, 40, 5, 2, 2, bias=False),
            nn.BatchNorm2d(40),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(40, 80 , 5, 2, 2, bias=False),
            nn.BatchNorm2d(80),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(80, 160, 5, 2, 2, bias=False),
            nn.BatchNorm2d(160),
            nn.LeakyReLU(inplace=True),
            Reshape(1, 7040),
            nn.Linear(7040, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
