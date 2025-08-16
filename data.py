from torchvision.transforms import transforms
from torchvision.datasets.mnist import FashionMNIST
from torch.utils.data import DataLoader, Subset

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

