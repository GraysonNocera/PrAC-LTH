import os
import matplotlib.pyplot as plt

def getListAccuracy(file):
    lines = []
    with open(file, 'r') as f:
      lines = f.readlines()

    acc_list = []
    for line in lines:
        if line.startswith("* best SA="):
            acc = float(line.split('=')[-1])
            acc_list.append(acc)
    return acc_list


def plotAcc(acc_list1, acc_list2, acc_list3, sparsities, title, legend, file_name):
    plt.plot(sparsities, acc_list1, label=title)
    plt.plot(sparsities, acc_list2, label=title)
    plt.plot(sparsities, acc_list3, label=title)
    plt.legend(legend)
    plt.xlabel("Model Sparsities (%)")
    plt.ylabel("Accuracy (%)")
    plt.savefig(file_name)
    plt.close()
    plt.clf()



cifar_with_bars_exec_time = 7121.601646184921
cifar10_easy_to_forget_exec_time = 7121.601646184921
cifar10_hard_to_memorize_exec_time = 8579.314288377762
cifar10_acc = getListAccuracy("cifar10_with_bars/output.log")
cifar10_easy_to_forget_acc = getListAccuracy("cifar10_easy_to_forget/output.log")
cifar10_hard_to_memorize_acc = getListAccuracy("cifar10_hard_to_memorize/output.log")

# cifar100_acc = getListAccuracy("cifar100_with_bars/output.log")
# cifar100_easy_to_forget_acc = getListAccuracy("cifar100_easy_to_forget_attempt_2/output.log")
# cifar100_hard_to_memorize_acc = getListAccuracy("cifar100_hard_to_memorize/output.log")

sparsities = [19.999925288386834, 35.99979080748312, 48.799757934373325, 59.03973163588548, 67.23186002032156, 73.78556272787043, 79.02845018229634, 83.22276014583707, 86.57805869344331, 89.26244695475465, 91.41003227541688, 93.12802582033352, 94.50234594465363, 95.60172733249657, 96.48145657761043, 97.18523997370151]

plotAcc(cifar10_acc, cifar10_easy_to_forget_acc, cifar10_hard_to_memorize_acc, sparsities, "Accuracies", ["CIFAR-10", "CIFAR-10 Easy to Forget", "CIFAR-10 Hard to Memorize"], "cifar10_acc.png")
# plotAcc(cifar100_acc, cifar100_easy_to_forget_acc, cifar100_hard_to_memorize_acc, sparsities, "Accuracies", ["CIFAR-100", "CIFAR-100 Easy to Forget", "CIFAR-100 Hard to Memorize"], "cifar100_acc.png")