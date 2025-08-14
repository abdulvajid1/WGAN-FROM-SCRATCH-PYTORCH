import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Annotated


class Generator(nn.Module):
    """Generate new image from input sampled from normal distribution.
    """
    def __init__(self, latent_dim, device='cuda'):
        super().__init__()
        self.latent_dim = latent_dim
        self.device = device
        
        # (input - 1) * stride - 2*padding + kernal size 
        self.seq_pipe = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.latent_dim, out_channels=512, stride=1, kernel_size=4), # (128, 4, 4)
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2), #  (64, 10, 10)
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2), # (32, 22, 22)
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=1), # (16, 25, 25)
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4, stride=1), # (1, 28, 28)
            nn.Tanh()
        )
        
    
    def forward(self, batch_size):
        latent_inp = torch.randn(size=(batch_size, self.latent_dim)).to(self.device) # (b, h)
        x = torch.unsqueeze(torch.unsqueeze(latent_inp, dim=-1), dim=-1) # (b, h, 1, 1)
        return self.seq_pipe(x)
        
        
        
    
class Descriminator(nn.Module):
    """
    Classify image => real or fake
    """
    def __init__(self):
        super().__init__()
        self.descriminator = nn.Sequential(
            
            # (inp + 1)/stride + 2*padding - kernal
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding='same'), # (3, 28, 28) -> (64, 28, 28)
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2), # (128, 17, 17)
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, stride=2), # (32, 6, 6)
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=1152, out_features=64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1),
        )
    
    def forward(self, img_batch):
        assert img_batch.shape[1:] == torch.Size([1, 28, 28]), 'the size of the image should (b, 1, 28, 28)' 
        predictions = self.descriminator(img_batch)
        return predictions
        
    
class DCGAN(nn.Module):
    """
    GAN model
    """
    def __init__(self,generator: Generator, descriminator: Descriminator, batch_size, device):
        super().__init__()
        self.generator = generator
        self.descriminator = descriminator
        self.batch_size = batch_size
        self.device = device
    
    def generate_image_batch(self, batch_size):
        return self.generator(batch_size)
    
    def descriminate_batch(self, img_batch):
        return self.descriminator(img_batch)
    
    def forward(self, x):
        fake_img_batch = self.generate_image_batch(self.batch_size) # (B, C, W, H)
        real_img_batch = x # real image comes from the dataset
        
        real_labels = torch.ones(size=(self.batch_size, ), device=self.device)
        fake_labels = torch.zeros(size=(self.batch_size, ), device=self.device)
        
        # Descriminate
        real_desc_pred = self.descriminator(real_img_batch).squeeze() # (5, 1)
        fake_desc_pred = self.descriminator(fake_img_batch).squeeze() # Descrimating Generated image
        
        # Descriminator
        real_desc_loss = F.binary_cross_entropy(real_desc_pred, real_labels)
        fake_desc_loss = F.binary_cross_entropy(fake_desc_pred, fake_labels)
        descriminator_loss = (real_desc_loss + fake_desc_loss) / 2.0 # average
        # Generator loss
        generator_loss = F.binary_cross_entropy(fake_desc_pred, real_labels)
        return generator_loss, descriminator_loss
        
        
def build_model(latent_dim, batch_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    descriminator = Descriminator()
    generator = Generator(latent_dim=latent_dim)
    
    model = DCGAN(generator, descriminator, batch_size, device=device)
    
    return model