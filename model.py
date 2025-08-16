import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Annotated


import torch
import torch.nn as nn

class Generator(nn.Module):
    """Generate new 128x128 RGB image from latent vector z."""
    def __init__(self, latent_dim, device='cuda'):
        super().__init__()
        self.latent_dim = latent_dim
        self.device = device

        # Start from a 4x4 feature map
        self.seq_pipe = nn.Sequential(
            # Input: z latent vector -> (latent_dim, 1, 1)
            self.block(latent_dim, 1024, 4, 1, 0),   # (1024, 4, 4)
            self.block(1024, 512, 4, 2, 1),          # (512, 8, 8)
            self.block(512, 256, 4, 2, 1),           # (256, 16, 16)
            self.block(256, 128, 4, 2, 1),           # (128, 32, 32)
            self.block(128, 64, 4, 2, 1),            # (64, 64, 64)
            self.block(64, 32, 4, 2, 1),             # (32, 128, 128)
            nn.ConvTranspose2d(32, 3, 3, 1, 1),      # (3, 128, 128)
            nn.Tanh()                                # output range [-1,1]
        )

    @staticmethod
    def block(inp, out, kernel, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(inp, out, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out),
            nn.ReLU(True)  # ReLU works better than LeakyReLU in G
        )

    def forward(self, x):
        return self.seq_pipe(x)
        
        
        
    
class Critic(nn.Module):
    """
    Critic/Discriminator for 128x128 RGB images.
    Outputs a single score (real vs fake).
    """
    def __init__(self):
        super().__init__()
        self.critic = nn.Sequential(
            # Input: (3, 128, 128)
            self.block(3, 32, 4, 2, 1, batch_norm=False),   # (32, 64, 64)
            self.block(32, 64, 4, 2, 1),                   # (64, 32, 32)
            self.block(64, 128, 4, 2, 1),                  # (128, 16, 16)
            self.block(128, 256, 4, 2, 1),                 # (256, 8, 8)
            self.block(256, 512, 4, 2, 1),                 # (512, 4, 4)
            nn.Conv2d(512, 1, 4, 1, 0),                    # (1, 1, 1)
            nn.Flatten()                                   # -> (batch, 1)
        )

    @staticmethod
    def block(inp, out, kernel, stride, padding, batch_norm=True):
        layers = [
            nn.Conv2d(inp, out, kernel, stride, padding, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        if batch_norm:
            layers.insert(1, nn.BatchNorm2d(out))
        return nn.Sequential(*layers)

    def forward(self, img_batch):
        return self.critic(img_batch)