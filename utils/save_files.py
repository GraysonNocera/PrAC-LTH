from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder, MNIST
from torch.utils.data import DataLoader, Subset
import numpy as np

def write():
  # train_transform = transforms.Compose([
  #     transforms.RandomCrop(32, padding=4),
  #     transforms.RandomHorizontalFlip(),
  #     transforms.ToTensor(),
  # ])

  # test_transform = transforms.Compose([
  #     transforms.ToTensor(),
  # ])

  # train_set = Subset(CIFAR10("./data", train=True, transform=train_transform, download=True), list(range(45000)))
  # val_set = Subset(CIFAR10("./data", train=True, transform=test_transform, download=True), list(range(45000, 50000)))
  # test_set = CIFAR10("./data", train=False, transform=test_transform, download=True)

  # train = train_set.dataset.data
  print(np.arange(0,60000))
  np.save("npy_files/mnist-train-val.npy", np.arange(0,50000))


def read():
  arr = np.load("npy_files/tiny-imagenet-train-val.npy")
  print(arr)
  print(arr.shape)

write()