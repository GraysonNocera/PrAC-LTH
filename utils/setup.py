import copy 
import torch
import numpy as np 

from dataset import *

from models.vgg import vgg16_bn
from models.resnet import resnet18, resnet50
from models.resnets import resnet20, resnet56

from advertorch.utils import NormalizeByChannelMeanStd



__all__ = ['setup_model_dataset', 'setup_model_dataset_PIE',
            'forget_times', 'sorted_examples', 'split_class_sequence', 'blance_dataset_sequence']



def setup_model_dataset(args, if_train_set=False):

    if args.dataset == 'cifar10':
        # 60,000 32x32 color images in 10 classes (6000 per class)
        # 50,000 training images and 10,000 test images
 
        classes = 10
        train_number = 45000
        normalization = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        train_set_loader, val_loader, test_loader = cifar10_dataloaders(batch_size = args.batch_size, data_dir = args.data, dataset = if_train_set)

    elif args.dataset == 'cifar100':
        classes = 100
        train_number = 45000
        normalization = NormalizeByChannelMeanStd(
            mean=[0.5071, 0.4866, 0.4409], std=[0.2673,	0.2564,	0.2762])
        train_set_loader, val_loader, test_loader = cifar100_dataloaders(batch_size = args.batch_size, data_dir = args.data, dataset = if_train_set)

    elif args.dataset == 'tiny-imagenet':
        classes = 200
        train_number = 90000
        normalization = NormalizeByChannelMeanStd(
            mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262])
        train_set_loader, val_loader, test_loader = tiny_imagenet_dataloaders(batch_size = args.batch_size, data_dir = args.data, dataset = if_train_set, split_file = args.split_file)

    elif args.dataset == 'mnist':
        # train size - 50,000
        # val size - 10,000
        # test size - 10,000
        # total - 70,000

        classes = 10
        train_number = 50000

        # Should only have one channel for MNIST because black and white images, thus only one mean and one std
        mnist_mean = 0.1307
        mnist_std = 0.3081
        normalization = NormalizeByChannelMeanStd(
            mean=[mnist_mean], std=[mnist_std])
        
        train_set_loader, val_loader, test_loader = mnist_dataloaders(batch_size = args.batch_size, data_dir = args.data, dataset = if_train_set, split_file = args.split_file)

    else:
        raise ValueError('unknown dataset')

    if args.arch == 'res18':
        print('build model resnet18')
        model = resnet18(num_classes=classes, imagenet=True)
    elif args.arch == 'res50':
        print('build model resnet50')
        model = resnet50(num_classes=classes, imagenet=True)
    elif args.arch == 'res20s':
        print('build model: resnet20')
        model = resnet20(number_class=classes)
    elif args.arch == 'res56s':
        print('build model: resnet56')
        model = resnet56(number_class=classes)
    elif args.arch == 'vgg16_bn':
        print('build model: vgg16_bn')
        model = vgg16_bn(num_classes=classes)
    else:
        raise ValueError('unknow model')

    model.normalize = normalization

    if if_train_set:
        return model, train_set_loader, val_loader, test_loader, train_number
    else:
        return model, train_set_loader, val_loader, test_loader


def setup_model_dataset_PIE(args):

    if args.dataset == 'cifar10':
        classes = 10
        normalization = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        train_loader = cifar10_dataloaders_val(batch_size = args.batch_size, data_dir = args.data)

    elif args.dataset == 'cifar100':
        classes = 100
        normalization = NormalizeByChannelMeanStd(
            mean=[0.5071, 0.4866, 0.4409], std=[0.2673,	0.2564,	0.2762])
        train_loader = cifar100_dataloaders_val(batch_size = args.batch_size, data_dir = args.data)

    elif args.dataset == 'tiny-imagenet':
        classes = 200
        normalization = NormalizeByChannelMeanStd(
            mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262])
        train_loader = tiny_imagenet_dataloaders_val(batch_size = args.batch_size, data_dir = args.data, split_file=args.split_file)

    else:
        raise ValueError('unknow dataset')

    if args.arch == 'res18':
        print('build model resnet18')
        model = resnet18(num_classes=classes, imagenet=True)
    elif args.arch == 'res50':
        print('build model resnet50')
        model = resnet50(num_classes=classes, imagenet=True)
    elif args.arch == 'res20s':
        print('build model: resnet20')
        model = resnet20(number_class=classes)
    elif args.arch == 'res56s':
        print('build model: resnet56')
        model = resnet56(number_class=classes)
    elif args.arch == 'vgg16_bn':
        print('build model: vgg16_bn')
        model = vgg16_bn(num_classes=classes)
    else:
        raise ValueError('unknow model')

    model.normalize = normalization

    return model, train_loader




# other function (this is the most useless comennt ever)
def forget_times(record_list):

    # Going through the epochs and seeing how many times the model swapped its bet
    # I.E. if it guessed the first time right then wrong second time, +1 
    
    offset = 200000
    number = offset
    learned = False

    for i in range(record_list.shape[0]):
        
        if not learned:
            if record_list[i] == 1:
                learned = True 
                if number == offset:
                    number = 0

        else:
            if record_list[i] == 0:
                learned = False
                number+=1 

    return number

# Is this where the actual pruning happens?
def sorted_examples(example_wise_prediction, data_prune, data_rate, state, threshold, train_number):

    # Threshold for reamining forgetting events
    # Looks like this is the first data slimming

    offset = 200_000

    forgetting_events_number = np.zeros(example_wise_prediction.shape[0])
    for j in range(example_wise_prediction.shape[0]):
        tmp_data = example_wise_prediction[j,:]
        if tmp_data[0] < 0:
            forgetting_events_number[j] = -1 
        else:
            forgetting_events_number[j] = forget_times(tmp_data)

    # forgetting_events_number holds 50,000 items of how much each sample is forgotten

    # print('* never learned image number = {}'.format(np.where(forgetting_events_number==offset)[0].shape[0]))

    if data_prune == 'constent':
        print('* pruning {} data'.format(data_rate))
        rest_number = int(train_number*(1-data_rate)**state)
    elif data_prune == 'zero_out':
        rest_number = np.where(forgetting_events_number > threshold)[0].shape[0]
        print('zero all unforgettable images out, rest number = ', rest_number)
    else:
        print('error data_prune type')
        assert False

    # print('max forgetting times = {}'.format(np.max(forgetting_events_number)))
    selected_index = np.argsort(forgetting_events_number)[-rest_number:]

    return selected_index

def split_class_sequence(sequence, all_labels, num_class):
    
    class_wise_sequence = {}
    for i in range(num_class):
        class_wise_sequence[i] = []
    
    for index in range(sequence.shape[0]):
        class_wise_sequence[all_labels[sequence[index]]].append(sequence[index])
    
    for i in range(num_class):
        class_wise_sequence[i] = np.array(class_wise_sequence[i])
        print('class = {0}, number = {1}'.format(i, class_wise_sequence[i].shape[0]))

    return class_wise_sequence

def blance_dataset_sequence(class_wise_sequence, num_class):

    class_wise_number = np.zeros(num_class, dtype=np.int)
    for i in range(num_class):
        class_wise_number[i] = class_wise_sequence[i].shape[0]
    
    max_length = np.max(class_wise_number)
    print('max class number = {}'.format(max_length))

    balance_sequence = []
    arange_max = np.arange(max_length)
    for i in range(num_class):

        shuffle_index = np.random.permutation(class_wise_number[i])
        shuffle_class_sequence = class_wise_sequence[i][shuffle_index]
        balance_sequence.append(shuffle_class_sequence[arange_max%class_wise_number[i]])

    balance_sequence = np.concatenate(balance_sequence)
    print(balance_sequence.shape)
    return balance_sequence



