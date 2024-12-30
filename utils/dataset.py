from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
from .transforms import get_transforms

class AlbumentationsDataset:
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = np.array(img)
        
        if self.transform:
            transformed = self.transform(image=img)
            img = transformed["image"]
            
        return img, label

    def __len__(self):
        return len(self.dataset)

def get_dataloaders(batch_size, num_workers):
    train_transform = get_transforms(train=True)
    test_transform = get_transforms(train=False)

    train_dataset = datasets.CIFAR10(root='../../data', train=True, download=True)
    test_dataset = datasets.CIFAR10(root='../../data', train=False, download=True)

    train_dataset = AlbumentationsDataset(train_dataset, train_transform)
    test_dataset = AlbumentationsDataset(test_dataset, test_transform)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader 