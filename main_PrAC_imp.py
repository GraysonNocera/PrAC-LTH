'''
iterative magnitude pruning with Early Bird Tickets and Pruning-Aware Critical Subset
'''

import os
import pdb
import time 
import pickle
import random
import shutil
import argparse
import numpy as np  
from copy import deepcopy
import matplotlib.pyplot as plt

import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torchvision.models as models
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler
from advertorch.utils import NormalizeByChannelMeanStd

from utils.pruner import *
from utils.setup import *

parser = argparse.ArgumentParser(description='PyTorch Iterative Pruning')

##################################### general setting #################################################
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
parser.add_argument('--arch', type=str, default='res20s', help='model architecture')
parser.add_argument('--split_file', type=str, default=None, help='dataset index', required=True)
parser.add_argument('--seed', default=None, type=int, help='random seed')
parser.add_argument('--save_dir', help='The directory used to save the trained models', default=None, type=str)
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--resume', action="store_true", help="resume from checkpoint")
parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint file')

##################################### training setting #################################################
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--epochs', default=182, type=int, help='number of total epochs to run')
parser.add_argument('--warmup', default=0, type=int, help='warm up epochs')
parser.add_argument('--print_freq', default=50, type=int, help='print frequency')
parser.add_argument('--decreasing_lr', default='91,136', help='decreasing strategy')

##################################### Model Pruning setting ############################################
parser.add_argument('--pruning_times', default=16, type=int, help='overall times of pruning')
parser.add_argument('--rate', default=0.2, type=float, help='pruning rate')
parser.add_argument('--eb_eps', default=0.08, type=float, help='epsilon for mask distance')
parser.add_argument('--queue_length', default=5, type=int, help='distance queue length')
parser.add_argument('--prune_type', default='lt', type=str, help='IMP type (lt, rewind_lt)')
parser.add_argument('--rewind_epoch', default=2, type=int, help='rewind checkpoint ')
parser.add_argument('--core_set_method', default="both", type=str, help="How to form the pruning aware critical set (both, easy_to_forget, hard_to_memorize)")

##################################### Data pruning setting ##############################################
parser.add_argument('--data_prune', default='zero_out', type=str, help='data_prune type (zero_out, constent)')
parser.add_argument('--threshold', default=0, type=int, help='threshold for remaining forgetting events')
parser.add_argument('--data_rate', default=0.2, type=float, help='data pruning rate')

best_sa = 0

def main():
    global args, best_sa
    args = parser.parse_args()
    print(args)

    torch.cuda.set_device(int(args.gpu))
    os.makedirs(args.save_dir, exist_ok=True)
    if args.seed:
        setup_seed(args.seed)

    # prepare dataset 
    # this gets the model and dataset based on the user input in CLI
    model, train_dataset, val_loader, test_loader, train_number = setup_model_dataset(args, if_train_set=True)
    print(f"Train number for dataset: {train_number}")
    model.cuda()

    criterion = nn.CrossEntropyLoss()
    decreasing_lr = list(map(int, args.decreasing_lr.split(',')))

    origin_decreasing_lr = deepcopy(decreasing_lr)
    origin_epoch = deepcopy(args.epochs)
    origin_warmup = deepcopy(args.warmup)

    if args.prune_type == 'lt':
        print('lottery tickets setting (rewind to random init')
        initalization = deepcopy(model.state_dict())
    elif args.prune_type == 'rewind_lt':
        initalization = None
    else:
        assert False

    # Use SGD for gradient descent
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # Decay the learning rate after reaching certain training iterations
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)
    
    if args.resume:
        print('resume from checkpoint')

        # Reload some data about the model from a file
        checkpoint = torch.load(args.checkpoint, map_location = torch.device('cuda:'+str(args.gpu)))
        best_sa = checkpoint['best_sa']
        start_epoch = checkpoint['epoch']
        all_result = checkpoint['result']
        start_state = checkpoint['state']
        example_wise_prediction = checkpoint['prediction']
        sequence = checkpoint['sequence']
        remain_para = checkpoint['remain_para']
        distance_queue = checkpoint['distance_queue']
        last_mask = checkpoint['last_mask']
        start_record = checkpoint['start_record']

        if start_state>0:
            current_mask = extract_mask(checkpoint['state_dict'])
            prune_model_custom(model, current_mask)
            check_sparsity(model)
            optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)
    
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        initalization = checkpoint['init_weight']
        print('loading state:', start_state)
        print('loading from epoch: ',start_epoch, 'best_sa=', best_sa)

    else:
        all_result = {}
        all_result['train'] = []
        all_result['test_ta'] = []
        all_result['ta'] = []

        example_wise_prediction = []
        remain_para = cnt_model_para(model)
        distance_queue = torch.ones(args.queue_length)
        last_mask = None
        start_record = False

        start_epoch = 0
        start_state = 0
        sequence = np.load(args.split_file)[:train_number]


    ## MY CODE ##
    forgetting_num = [] # forgetting statistics data points
    pie_num = [] # pruned model vs. entire model data points
    both_num = []
    model_sparsities = []
    dataset = args.dataset
    ## END MY CODE ##

    startTime = time.time()
    print('######################################## Start Standard Training Iterative Pruning ########################################')

    for state in range(start_state, args.pruning_times):

        print('******************************************')
        print('pruning state', state)
        print('* remain parameters = {}'.format(remain_para))
        print('******************************************')

        check_sparsity(model)        
        for epoch in range(start_epoch, args.epochs):

            print(optimizer.state_dict()['param_groups'][0]['lr'])

            acc, epoch_acc_train = train(train_dataset, model, criterion, optimizer, epoch, sequence)
            example_wise_prediction.append(epoch_acc_train)

            if state == 0:
                if epoch == args.rewind_epoch:
                    torch.save(model.state_dict(), os.path.join(args.save_dir, 'epoch_{}_rewind_weight.pt'.format(epoch+1)))
                    if args.prune_type == 'rewind_lt':
                        initalization = deepcopy(model.state_dict())

            # evaluate on validation set
            tacc = validate(val_loader, model, criterion)
            # evaluate on test set
            test_tacc = validate(test_loader, model, criterion)

            # Decays the learning rate after certain milestones
            scheduler.step()

            all_result['train'].append(acc)
            all_result['ta'].append(tacc)
            all_result['test_ta'].append(test_tacc)

            print(f"Epoch #{epoch}: Train Accuracy: {acc}, Validation Accuracy: {tacc}, Test Accuracy: {test_tacc}")

            # remember best prec@1 and save checkpoint
            is_best_sa = tacc  > best_sa
            best_sa = max(tacc, best_sa)

            # Early Bird Tickets, check whether current mask is good enough
            # early breaking, maybe we don't have to go up to the highest epoch but can move to the next pruning rate earlier
            epoch_mask = return_current_mask(model, args.rate, pruned=state) # rate is the pruning rate
            if epoch:
                hamming_dis = calculate_hamming_distance(last_mask, epoch_mask, remain_para)
                start_record = start_record or (hamming_dis > args.eb_eps) #eb_eps - epsilon for mask distance
                if epoch > args.warmup and start_record:
                    distance_queue = FIFO(distance_queue, hamming_dis)
                print('* current-mask-distance is = {}'.format(hamming_dis))

                if distance_queue.max() < args.eb_eps:
                    flag_break = True
                else:
                    flag_break = False
                    last_mask = deepcopy(epoch_mask)
            else:
                flag_break = False
                last_mask = deepcopy(epoch_mask)

            save_checkpoint({
                'state': state,
                'result': all_result,
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_sa': best_sa,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'init_weight': initalization,
                'prediction': example_wise_prediction,
                'sequence': sequence,
                'remain_para': remain_para,
                'distance_queue': distance_queue,
                'last_mask': last_mask,
                'start_record': start_record
            }, is_SA_best=is_best_sa, pruning=state, save_path=args.save_dir)
        
            plt.plot(all_result['train'], label='train_acc')
            plt.plot(all_result['ta'], label='val_acc')
            plt.plot(all_result['test_ta'], label='test_acc')
            plt.legend()
            plt.yticks(np.arange(0, 101, 5))
            plt.ylabel("Accuracy (%)")
            plt.xlabel("Epoch")
            plt.title(f"Accuracy at Pruning Stage {state}")
            plt.savefig(os.path.join(args.save_dir, str(state)+'net_train.png'))
            plt.close()

            if flag_break:
                break 

        #report result
        check_sparsity(model)
        print('* best SA={}'.format(all_result['test_ta'][np.argmax(np.array(all_result['ta']))]))

        all_result = {}
        all_result['train'] = []
        all_result['test_ta'] = []
        all_result['ta'] = []

        best_sa = 0
        start_epoch = 0
        start_record = False
        remain_para = cnt_remain_para(epoch_mask)
        distance_queue = torch.ones(args.queue_length)
        last_mask = None

        print('* find early bird tickets at epoch {}'.format(epoch+1))
        if state:
            remove_prune(model)
        model.load_state_dict(initalization)

        # Actual pruning? Based on L1 norm
        prune_model_custom(model, epoch_mask)   
 
        model_sparsities.append(100 - check_sparsity(model))               

        # construct PrAC sets
        # example_wise_prediction holds the predictions of all 50,000 trained images for each epoch
        # hence, size is (50,000, num_epochs_in_this_state) - I think these are 0 or 1s based on if it was correct
        example_wise_prediction = np.concatenate(example_wise_prediction, axis=1)
        print('* record size = {}'.format(example_wise_prediction.shape))

        # This may be where the actual pruning happens?
        sequence = sorted_examples(example_wise_prediction, args.data_prune, args.data_rate, state+1, args.threshold, train_number, args.core_set_method)
        forgetting_num.append(len(list(sequence)))

        if state:
            # This is the second round of pruning
            pie_index = prune_aware_example(os.path.join(args.save_dir, '0model_SA_best.pth.tar'), 
                os.path.join(args.save_dir, str(state)+'model_SA_best.pth.tar'), criterion, args)
            
            # Loading all of them but only picking the ones that don't match pruned vs full
            pie_sequence = np.load(args.split_file)[pie_index]

            try:
                if len(sequence) > 0 and len(pie_sequence) > 0:
                    tmp_list = list(sequence).extend(list(pie_sequence))
                    tmp_set_list = list(set(list(sequence).extend(list(pie_sequence))))
                    print("Number of samples in both types of data: ", len(tmp_list) - len(tmp_set_list))
                    pie_num.append(len(list(pie_sequence)))
                    both_num.append(len(tmp_list) - len(tmp_set_list))
                else:
                    pie_num.append(len(list(pie_sequence)))
                    both_num.append(0)
            except:
                print(sequence)
                print(pie_sequence)
                pie_num.append(len(list(pie_sequence)))
                both_num.append(0)

            # Combining the two types of data from the two prunings
            # pie_sequence - Pruning where check between subnetwork and full model prediction
            # sequence (main_sequence) - Pruning where we check the forgetting statistics
            sequence = concate_sequence(pie_sequence, sequence)

            if len(sequence) == 0:
                sequence = np.load(args.split_file)[:train_number]
        else:
            pie_num.append(0)
            both_num.append(0)

            if len(sequence) == 0:
                # We have no pie_sequence and no sequence (i.e. no PrAC set), so we must call the sequence all the data
                sequence = np.load(args.split_file)[:train_number]

        # dynamic training iterations
        # Just gaining training efficiency
        args.epochs = int(origin_epoch*sequence.shape[0]/train_number)
        decreasing_lr = [int(d*sequence.shape[0]/train_number) for d in origin_decreasing_lr]
        args.warmup = int(origin_warmup*sequence.shape[0]/train_number)

        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)

        example_wise_prediction = []

    print(f"Algorithm finished in {time.time() - startTime} seconds")

    print("forgetting numbers: ", forgetting_num)
    print("pie num: ", pie_num)
    print("model sparsities: ", model_sparsities)

    plot_data_types_for_dataset(forgetting_num, pie_num, model_sparsities, dataset, args.save_dir)

def plot_data_types_for_dataset(forgetting_num, pie_num, model_sparsities, dataset, save_dir):

    width = 2
    model_sparsities = [x - width/2 for x in model_sparsities]
    plt.bar(model_sparsities, forgetting_num, width=width)
    model_sparsities = [x + width for x in model_sparsities]
    plt.bar(model_sparsities, pie_num, width=width)
    plt.title(f"Comparison of Data Types for {dataset}")
    plt.xlabel("Model Sparsities (%)")
    plt.ylabel("Number of Data Points")
    plt.legend(["Hard to Memorize", "Easy to Forget"])
    plt.savefig(os.path.join(save_dir, f"data_types_sparsities_{dataset}.png"))


def train(trainset, model, criterion, optimizer, epoch, sequence):
    
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    image_number = sequence.shape[0]
    if epoch == 0:
        print('* training images number = {}'.format(image_number))

    number_of_all_dataset = len(trainset)
    all_acc_train = -np.ones((number_of_all_dataset, 1))
    # Get permutation to shuffle trainset
    trainset_permutation_inds = sequence[np.random.permutation(
        np.arange(image_number))]

    batch_size = args.batch_size
    iteration_steps = int(image_number/batch_size)
    if (image_number/batch_size) > iteration_steps:
        iteration_steps+=1

    start = time.time()
    for batch_idx, batch_start_ind in enumerate(
            range(0, image_number, batch_size)):

        if epoch < args.warmup:
            warmup_lr(epoch, batch_idx+1, optimizer, one_epoch_step=iteration_steps)

        # get batch inputs and targets, transform them appropriately
        batch_inds = trainset_permutation_inds[batch_start_ind:
                                            batch_start_ind + batch_size]

        transformed_trainset = []
        for ind in batch_inds:
            transformed_trainset.append(trainset.__getitem__(ind)[0])
        image = torch.stack(transformed_trainset)
        target = torch.LongTensor(
            np.array(trainset.targets)[batch_inds].tolist())
        
        image = image.cuda()
        target = target.cuda()

        # compute output
        output_clean = model(image)
        loss = criterion(output_clean, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update statistic
        predicted = torch.argmax(output_clean.data, 1)
        acc = (predicted == target)
        for j, index in enumerate(batch_inds):
            all_acc_train[index,0] = acc[j]

        output = output_clean.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]

        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

        if batch_idx % args.print_freq == 0:
            end = time.time()
            print('Epoch: [{0}][{1}/{2}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                'Time {3:.2f}'.format(
                    epoch, batch_idx, iteration_steps, end-start, loss=losses, top1=top1))
            start = time.time()

    print('train_accuracy {top1.avg:.3f}'.format(top1=top1))

    return top1.avg, all_acc_train

def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (image, target) in enumerate(val_loader):
        
        image = image.cuda()
        target = target.cuda()

        # compute output
        with torch.no_grad():
            output = model(image)
            loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i, len(val_loader), loss=losses, top1=top1))

    print('valid_accuracy {top1.avg:.3f}'
        .format(top1=top1))

    return top1.avg

def validate_PIE(val_loader, model, criterion):
    """
    Run evaluation
    """
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    prediction_record = []

    for i, (image, target) in enumerate(val_loader):
        
        image = image.cuda()
        target = target.cuda()

        # compute output
        with torch.no_grad():
            output = model(image)
            loss = criterion(output, target)

        prediction_record.append(torch.argmax(output, dim=1).cpu())

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i, len(val_loader), loss=losses, top1=top1))

    print('valid_accuracy {top1.avg:.3f}'
        .format(top1=top1))

    prediction_record = torch.cat(prediction_record, dim=0)
    
    return prediction_record

def prune_aware_example(full_model_path, prune_model_path, criterion, args):

    # Other pruning for critical set here

    # re-define model and loader
    test_model, test_train_loader = setup_model_dataset_PIE(args)
    test_model.cuda()
    full_model_path_checkpoint = torch.load(full_model_path, map_location = torch.device('cuda:'+str(args.gpu)))['state_dict']
    test_model.load_state_dict(full_model_path_checkpoint)
    full_model_train_predict = validate_PIE(test_train_loader, test_model, criterion)

    # pruning model 
    prune_model_path_checkpoint = torch.load(prune_model_path, map_location = torch.device('cuda:'+str(args.gpu)))['state_dict']
    current_mask = extract_mask(prune_model_path_checkpoint)
    prune_model_custom(test_model, current_mask)
    test_model.load_state_dict(prune_model_path_checkpoint)
    check_sparsity(test_model)
    prune_model_train_predict = validate_PIE(test_train_loader, test_model, criterion)

    pie_example = (full_model_train_predict == prune_model_train_predict).float()
    if (args.core_set_method == "hard_to_memorize"):
        # we don't rule out any if they don't want to use this pruning method
        pie_example = (full_model_train_predict == full_model_train_predict).float()

    pie_example = 1 - pie_example
    pie_example = pie_example.nonzero().reshape(-1)
    pie_example = pie_example.numpy()

    return pie_example

def concate_sequence(pie_sequence, main_sequence):

    main_sequence = list(main_sequence)
    pie_sequence = list(pie_sequence)

    print('* Critical examples for prunings = {}'.format(len(pie_sequence)))
    print('* Critical examples for training = {}'.format(len(main_sequence)))

    main_sequence.extend(pie_sequence)

    # Removes duplicates
    main_sequence = np.array(list(set(main_sequence)))

    print('* PrAC images = {}'.format(main_sequence.shape[0]))

    return main_sequence

def save_checkpoint(state, is_SA_best, save_path, pruning, filename='checkpoint.pth.tar'):
    filepath = os.path.join(save_path, str(pruning)+filename)
    torch.save(state, filepath)
    if is_SA_best:
        shutil.copyfile(filepath, os.path.join(save_path, str(pruning)+'model_SA_best.pth.tar'))

def warmup_lr(epoch, step, optimizer, one_epoch_step):

    overall_steps = args.warmup*one_epoch_step
    current_steps = epoch*one_epoch_step + step 

    lr = args.lr * current_steps/overall_steps
    lr = min(lr, args.lr)

    for p in optimizer.param_groups:
        p['lr']=lr

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def setup_seed(seed): 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed) 
    random.seed(seed) 
    torch.backends.cudnn.deterministic = True 

if __name__ == '__main__':
    main()


