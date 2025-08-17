from torchvision.transforms import transforms
from torchvision.datasets.mnist import FashionMNIST
from torch.utils.data import DataLoader, Subset
from datasets import load_dataset
import torch


def load_mnist_data(train_batch_size=8, test_batch_size=2, sample_size=60000):
    
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
    ])
    
    train_dataset = FashionMNIST(root='data/', train=True, transform=transform, download=True)
    test_dataset = FashionMNIST(root='data/', train=False, transform=transform, download=True)
    
    train_dataset = Subset(train_dataset, range(sample_size))
    train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=False, drop_last=True)
    
    return train_loader, test_loader


def load_car_data(train_batch_size=16, test_batch_size=2, sample_size=60000):
    
    from torchvision import datasets
    from torchvision.transforms import transforms
    from torch.utils.data import random_split

    transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    
    ])
    
    car_data = datasets.ImageFolder(root='data/cars', transform=transforms)
    
    train_dataset, test_dataset = random_split(car_data, lengths=[.97, .03])
    train_dataset = Subset(train_dataset, range(sample_size))
    train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=False, drop_last=True)
    
    return train_loader, test_loader

def collate_fn(batch):
    from torchvision.transforms import transforms
    
    transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    
    ])
    
    image_batch = []
    
    for image in batch:
        trf_image = torch.tensor(transforms(image['image']))
        image_batch.append(trf_image)
    
    return torch.stack(image_batch)

def load_bedroom_data(train_batch_size=16, test_batch_size=2, sample_size=60000):
    
    dataset = load_dataset("pcuenq/lsun-bedrooms")
    
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    
    train_dataset = train_dataset.shuffle().select(range(sample_size))
    train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)
    test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=False, drop_last=True, collate_fn=collate_fn)
    
    return train_loader, test_loader
    

        
        

