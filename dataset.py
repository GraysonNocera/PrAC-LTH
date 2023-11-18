import os 
import numpy as np 
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder, MNIST
from torch.utils.data import DataLoader, Subset

__all__ = ['cifar10_dataloaders', 'cifar100_dataloaders', 'tiny_imagenet_dataloaders', 'mnist_dataloaders',
            'cifar10_dataloaders_val', 'cifar100_dataloaders_val', 'tiny_imagenet_dataloaders_val', 'mnist_dataloaders_val']


def cifar10_dataloaders(batch_size=128, data_dir='datasets/cifar10', dataset=False):

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = Subset(CIFAR10(data_dir, train=True, transform=train_transform, download=True), list(range(45000)))
    val_set = Subset(CIFAR10(data_dir, train=True, transform=test_transform, download=True), list(range(45000, 50000)))
    test_set = CIFAR10(data_dir, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    if dataset:
        train_dataset = CIFAR10(data_dir, train=True, transform=train_transform, download=True)
        print('return train dataset: ', len(train_dataset))
        return train_dataset, val_loader, test_loader
    else:
        return train_loader, val_loader, test_loader

def cifar100_dataloaders(batch_size=128, data_dir='datasets/cifar100', dataset=False):

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = Subset(CIFAR100(data_dir, train=True, transform=train_transform, download=True), list(range(45000)))
    val_set = Subset(CIFAR100(data_dir, train=True, transform=test_transform, download=True), list(range(45000, 50000)))
    test_set = CIFAR100(data_dir, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    if dataset:
        train_dataset = CIFAR100(data_dir, train=True, transform=train_transform, download=True)
        print('return train dataset: ', len(train_dataset))
        return train_dataset, val_loader, test_loader
    else:
        return train_loader, val_loader, test_loader

def mnist_dataloaders(batch_size=128, data_dir='datasets/mnist', dataset=False):

    train_transform = transforms.Compose([
        transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Split into train, evaluation, and test sets
    train_set = Subset(MNIST(data_dir, train=True, transform=train_transform, download=True), list(range(50000)))
    val_set = Subset(MNIST(data_dir, train=True, transform=test_transform, download=True), list(range(50000, 60000)))
    test_set = MNIST(data_dir, train=False, transform=test_transform, download=True)

    # num_workers - how many subprocesses to use for data loading
    # drop_last - drop the last batch if it is incomplete
    # pin_memory - the data loader will copy tensors into CUDA pinned memory before returning them
    # test
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    if dataset:
        train_dataset = MNIST(data_dir, train=True, transform=train_transform, download=True)
        print('return train dataset: ', len(train_dataset))
        return train_dataset, val_loader, test_loader
    else:
        return train_loader, val_loader, test_loader

def tiny_imagenet_dataloaders(batch_size=64, data_dir='datasets/tiny-imagenet-200', dataset=False, split_file=None):

    train_transform = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_path = os.path.join(data_dir, 'train')
    val_path = os.path.join(data_dir, 'val')

    if not split_file:
        split_file = 'npy_files/tiny-imagenet-train-val.npy'
    split_permutation = list(np.load(split_file))

    train_set = Subset(ImageFolder(train_path, transform=train_transform), split_permutation[:90000])
    val_set = Subset(ImageFolder(train_path, transform=test_transform), split_permutation[90000:])
    test_set = ImageFolder(val_path, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    if dataset:
        train_dataset = ImageFolder(train_path, transform=train_transform)
        print('return train dataset: ', len(train_dataset))
        return train_dataset, val_loader, test_loader
    else:
        return train_loader, val_loader, test_loader



def cifar10_dataloaders_val(batch_size=128, data_dir='datasets/cifar10'):

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = Subset(CIFAR10(data_dir, train=True, transform=test_transform, download=True), list(range(45000)))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader

def cifar100_dataloaders_val(batch_size=128, data_dir='datasets/cifar100'):

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = Subset(CIFAR100(data_dir, train=True, transform=test_transform, download=True), list(range(45000)))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader

def mnist_dataloaders_val(batch_size=128, data_dir='datasets/mnist'):

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = Subset(MNIST(data_dir, train=True, transform=test_transform, download=True), list(range(50_000)))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader

def tiny_imagenet_dataloaders_val(batch_size=64, data_dir='datasets/tiny-imagenet-200', split_file=None):

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_path = os.path.join(data_dir, 'train')

    if not split_file:
        split_file = 'npy_files/tiny-imagenet-train-val.npy'
    split_permutation = list(np.load(split_file))

    train_set = Subset(ImageFolder(train_path, transform=test_transform), split_permutation[:90000])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader



