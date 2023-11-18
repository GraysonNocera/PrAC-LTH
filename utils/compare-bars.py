import matplotlib.pyplot as plt
import os

def plot_data_types_for_dataset(forgetting_num, pie_num, model_sparsities, dataset, save_dir):

    width = 2
    model_sparsities = [x - width/2 for x in model_sparsities]
    plt.bar(model_sparsities, forgetting_num, width=width)
    model_sparsities = [x + width for x in model_sparsities]
    plt.bar(model_sparsities, pie_num, width=width)
    plt.title(f"Comparison of Data Types for {dataset}")
    plt.xlabel("Model Sparsities (%)")
    plt.ylabel("Ratio of Data Points to Total Data Points")
    plt.legend(["Hard to Memorize", "Easy to Forget"])
    plt.savefig(os.path.join(save_dir, f"data_types_sparsities_{dataset}.png"))
    plt.clf()

save_dir = "compare_bars"

cifar10_forgetting_stats = [29618, 24291, 21848, 19737, 17805, 16362, 15274, 14584, 14249, 14158, 14096, 14361, 14634, 14949, 15378, 15857]
cifar10_different_prune_full = [0, 1259, 1266, 3436, 3605, 4028, 4432, 4494, 4302, 4234, 4689, 5413, 4816, 5335, 6002, 6651]
cifar10_model_sparsities = [19.999925288386834, 35.99979080748312, 48.799757934373325, 59.03973163588548, 67.23186002032156, 73.78556272787043, 79.02845018229634, 83.22276014583707, 86.57805869344331, 89.26244695475465, 91.41003227541688, 93.12802582033352, 94.50234594465363, 95.60172733249657, 96.48145657761043, 97.18523997370151]
cifar10_training_num = 50_000
cifar10_forgetting_stats_normalized = [x / cifar10_training_num for x in cifar10_forgetting_stats]
cifar10_different_prune_full_normalized = [x / cifar10_training_num for x in cifar10_different_prune_full]
plot_data_types_for_dataset(cifar10_forgetting_stats_normalized, cifar10_different_prune_full_normalized, cifar10_model_sparsities, "CIFAR-10", save_dir)

cifar100_forgetting_stats = [42340, 40669, 39697, 38479, 36893, 35618, 35056, 34765, 34636, 34674, 34560, 33554, 33673, 34205, 34715, 34253]
cifar100_different_prune_full = [0, 7898, 7620, 13050, 13561, 13872, 13196, 12258, 12836, 9075, 9876, 18604, 19819, 19396, 19679, 22446]
cifar100_model_sparsities = [19.999925288386834, 35.99979080748312, 48.799757934373325, 59.03973163588548, 67.23186002032156, 73.78556272787043, 79.02845018229634, 83.22276014583707, 86.57805869344331, 89.26244695475465, 91.41003227541688, 93.12802582033352, 94.50234594465363, 95.60172733249657, 96.48145657761043, 97.18523997370151]
cifar100_training_num = 50_000
cifar100_forgetting_stats_normalized = [x / cifar100_training_num for x in cifar100_forgetting_stats]
cifar100_different_prune_full_normalized = [x / cifar100_training_num for x in cifar100_different_prune_full]
plot_data_types_for_dataset(cifar100_forgetting_stats_normalized, cifar100_different_prune_full_normalized, cifar100_model_sparsities, "CIFAR-100", save_dir)

mnist_forgetting_stats = [2775, 1477, 916, 709, 677, 896, 1194, 825, 658, 696, 781, 941, 837, 1025, 1120, 1220]
mnist_different_prune_full = [0, 419, 539, 707, 1132, 2793, 580, 588, 1046, 1083, 1664, 926, 1914, 1647, 1424, 1406]
mnist_model_sparsities = [20.00014958415605, 36.000044875246815, 48.799961108119426, 59.04011847065159, 67.23209477652127, 73.78575061329505, 79.02867528271406, 83.22301501824927, 86.57856159875546, 89.26284927900437, 91.41012983904744, 93.12810387123795, 94.50240830491235, 95.60185185185185, 96.48140668940346, 97.1852001436008]
mnist_training_num = 60_000
mnist_forgetting_stats_normalized = [x / mnist_training_num for x in mnist_forgetting_stats]
mnist_different_prune_full_normalized = [x / mnist_training_num for x in mnist_different_prune_full]
plot_data_types_for_dataset(mnist_forgetting_stats_normalized, mnist_different_prune_full_normalized, mnist_model_sparsities, "MNIST", save_dir)

