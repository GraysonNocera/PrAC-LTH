import matplotlib.pyplot as plt
import os
import numpy as np
import torch

def plot_data_types_for_dataset(forgetting_num, pie_num, model_sparsities, dataset):

    width = 2
    model_sparsities = [x - width/2 for x in model_sparsities]
    plt.bar(model_sparsities, forgetting_num, width=width)
    model_sparsities = [x + width for x in model_sparsities]
    plt.bar(model_sparsities, pie_num, width=width)
    plt.title(f"Comparison of Data Types for {dataset}")
    plt.xlabel("Model Sparsities (%)")
    plt.ylabel("Number of Data Points")
    plt.legend(["Hard to Memorize", "Easy to Forget"])
    plt.savefig(os.path.join('.', "data_types_sparsities.png"))


# forgetting_num = [29000, 18000, 20000]
# pie_num = [10, 4000, 2000]
# model_sparsities = [20, 40, 60]
# dataset = "Cifar10"
# plot_data_types_for_dataset(forgetting_num, pie_num=pie_num, model_sparsities=model_sparsities, dataset=dataset)

# arr1 = torch.tensor([0, 1, 2])
# arr2 = torch.tensor([3, 1, 2])
# print(1 - (arr1 == arr2).float())

# 
ndarray = np.arange(10)
print(ndarray)
print(len(ndarray))

print(list(list([]).extend([1])))