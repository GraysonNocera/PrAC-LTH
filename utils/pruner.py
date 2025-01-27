import copy 
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


__all__  = ['pruning_model', 'pruning_model_random', 'prune_model_custom', 'remove_prune',
            'extract_mask', 'reverse_mask', 'check_sparsity',
            'return_current_mask', 'calculate_hamming_distance', 'cnt_remain_para', 'cnt_model_para', 'FIFO']



# Pruning operation
def pruning_model(model, px):

    print('Apply Unstructured L1 Pruning')
    parameters_to_prune =[]
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            parameters_to_prune.append((m,'weight'))

    parameters_to_prune = tuple(parameters_to_prune)
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=px,
    )

def pruning_model_random(model, px):

    print('Apply Unstructured Random Pruning')
    parameters_to_prune =[]
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            parameters_to_prune.append((m,'weight'))

    parameters_to_prune = tuple(parameters_to_prune)
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.RandomUnstructured,
        amount=px,
    )

def prune_model_custom(model, mask_dict):

    print('Pruning with custom mask')
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            prune.CustomFromMask.apply(m, 'weight', mask=mask_dict[name+'.weight_mask'])

def remove_prune(model):
    
    print('Remove Pruning')
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            prune.remove(m,'weight')



# Mask operation function
def extract_mask(model_dict):

    new_dict = {}

    for key in model_dict.keys():
        if 'mask' in key:
            new_dict[key] = copy.deepcopy(model_dict[key])

    return new_dict

def reverse_mask(mask_dict):
    new_dict = {}

    for key in mask_dict.keys():

        new_dict[key] = 1 - mask_dict[key]

    return new_dict

# Mask statistic function
def check_sparsity(model):
    
    sum_list = 0
    zero_sum = 0

    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            sum_list = sum_list+float(m.weight.nelement())
            zero_sum = zero_sum+float(torch.sum(m.weight == 0))  

    # zero_sum / sum_list = percentage of weights that are 0
    # 1 - that gives us the sparsity
    print('* remain weight = ', 100*(1-zero_sum/sum_list),'%')
    return 100*(1-zero_sum/sum_list)


# Function for Early Bird Tickets
# Early stopping: once the distance between sparsity masks (using Hamming distance)
# reaches a certain threshold, then we interrupt training, prune the network, update sparsity mask, and use it for further retraining
def return_current_mask(model, px, pruned=False):
    
    # save current state_dict
    checkpoint = copy.deepcopy(model.state_dict())
    if pruned:
        checkpoint_mask = extract_mask(checkpoint)

    # pruning and then unpruning to get the mask?

    # get current mask 
    pruning_model(model, px)
    epoch_mask = extract_mask(model.state_dict())

    # recover model
    remove_prune(model)
    if pruned:
        prune_model_custom(model, checkpoint_mask)
    model.load_state_dict(checkpoint)

    return epoch_mask

def calculate_hamming_distance(last_mask, current_mask, remain_parameters):

    cnt_diff = 0
    for key in last_mask.keys():
        same_number = (last_mask[key] == current_mask[key]).float().sum()
        mask_size = last_mask[key].nelement()
        cnt_diff += (mask_size - same_number)

    hamming_distance = cnt_diff/remain_parameters

    return hamming_distance

# Count remaining parameters
def cnt_remain_para(mask_dict):

    remain_para = 0
    for key in mask_dict.keys():
        remain_para += mask_dict[key].sum().float()

    return remain_para.item()

def cnt_model_para(model):

    para_number = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            para_number += m.weight.nelement()
    
    return para_number

def FIFO(dis_queue, ele):
    new_queue = torch.ones_like(dis_queue)
    new_queue[1:] = dis_queue[:-1]
    new_queue[0] = ele 
    return new_queue



