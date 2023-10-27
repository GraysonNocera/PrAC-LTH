from torchvision.datasets import MNIST
import numpy as np
from matplotlib import pyplot as plt
from torchvision import transforms

# MNIST has a training set of 60,000 and a test set of 10,000 (28x28)
# CIFAR-10 (CIFAR-100) has a training set of 50,000 and a testing set of 10,000 (32x32)
# Tiny-ImageNet has 100,000 images total

# For MNIST, I will use the same batch size as CIFAR-10 because the datasets are 
# similar in terms of size and number of classifications (10)

train_transform = transforms.Compose([

    # Randomly crops the original 36x36 image of MNIST to be the first argument
    # Essentially just moves the number around to different parts of the screen
    transforms.RandomCrop(32, padding=4),

    # Randomly flips the image horizontally (mirrors it)
    transforms.RandomHorizontalFlip(),

    # Converts PIL Image to a tensor
    transforms.ToTensor(),
])

mnist = MNIST(root='./data_test', train=True, download=True, transform=train_transform)
print(type(mnist))
print(len(mnist))
print(type(mnist[0]), mnist[0]) # Prints (image, target), image is 28x28

first_image = mnist[0][0]
plt.imshow(first_image.squeeze(), cmap='gray')
plt.savefig('figures/mnist.png')