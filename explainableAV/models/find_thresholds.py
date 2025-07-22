import numpy as np
import argparse
from explainableAV.models.test import evaluate_model, binarize
from explainableAV.utils.utils import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

def find_optimal_threshold(y_trues, y_preds, thresholds=np.arange(0.01, 1.0, 0.01)):
    """
    Finds the threshold that minimizes the standard deviation of accuracies across different splits
    Inputs:
        y_trues: y labels
        y_preds: predicted y scores
        thresholds: which thresholds to test for
    Output:
        threshold that minimizes the standard deviation of split accuracies
    """
    best_threshold = None
    min_std_dev = float('inf')
    stds = []
    total_accuracies = []

    for threshold in thresholds: # compute accuracies per threshold
        accuracies = []
        
        for true, pred in zip(y_trues, y_preds):
            accuracy = accuracy_score(true, binarize(pred, threshold))
            accuracies.append(accuracy)

        std_dev = np.std(accuracies)
        stds.append(std_dev)
        total_accuracies.append(accuracies)

        if std_dev < min_std_dev:
            min_std_dev = std_dev
            best_threshold = threshold

    return best_threshold, thresholds, stds, total_accuracies

def thresholds_plot(best_threshold, thresholds, stds, total_accuracies, model_name, dataset_name):
    '''
    Plot results for all thresholds
    Inputs:
        best_threshold: optimal found threshold
        threshold: list of all thresholds
        stds: list of stds over thresholds
        total_accuracies: total accuracies per thresholds
        model_name: name of the model
        dataset_name: name of the dataset
    '''
    SS_acc = [accuracy[0] for accuracy in total_accuracies]
    SD_acc = [accuracy[1] for accuracy in total_accuracies]
    DS_acc = [accuracy[2] for accuracy in total_accuracies]
    DD_acc = [accuracy[3] for accuracy in total_accuracies]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('Threshold', fontsize=22)
    ax1.set_ylabel('Accuracy', color='black', fontsize=22)
    custom_cmap = LinearSegmentedColormap.from_list("strong_puor", ["#5E00B5", "#FF8500"])

    ax1.plot(thresholds, SS_acc, color=custom_cmap(0.05), label='SS')
    ax1.plot(thresholds, SD_acc, color=custom_cmap(0.25), label='SD')
    ax1.plot(thresholds, DS_acc, color=custom_cmap(0.75), label='DS')
    ax1.plot(thresholds, DD_acc, color=custom_cmap(0.95), label='DD')
    ax1.tick_params(axis='y', labelcolor='black', labelsize=20)
    ax1.tick_params(axis='x', labelsize=20)

    ax2 = ax1.twinx()

    ax2.set_ylabel('Standard Deviation', color='black', fontsize=22)
    ax2.plot(thresholds, stds, color='black', label='Std. Dev.')
    ax2.tick_params(axis='y', labelcolor='black', labelsize=20)
    ax2.axvline(x=best_threshold, color='gray', linestyle='--', label='Optimal $\gamma$')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    fig.suptitle(f'Optimal $\gamma$ ({model_name})', fontsize=24)
    
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center left', fontsize=18)
    fig.tight_layout()

    plt.savefig(f"explainableAV/models/results/thresholds_{model_name}_{dataset_name}.pdf")
    plt.show()

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--SS_val_path', type=str, default="explainableAV/Amazon/SS_val.json", help="Path to validation set of SS split")
    parser.add_argument('--SD_val_path', type=str, default="explainableAV/Amazon/SD_val.json", help="Path to validation set of SD split")
    parser.add_argument('--DS_val_path', type=str, default="explainableAV/Amazon/DS_val.json", help="Path to validation set of DS split")
    parser.add_argument('--DD_val_path', type=str, default="explainableAV/Amazon/DD_val.json", help="Path to validation set of DD split")
    parser.add_argument('--model_name', type=str, default="LUAR", help="Model to use, one of: 'LUAR', 'StyleDistance', 'ModernBERT'")
    parser.add_argument('--dataset_name', type=str, default="amazon", help="Dataset to use, one of: 'amazon', 'pan20'")
    parser.add_argument('--title', type=str, help="Anything that will be printed to recognise different runs")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = argument_parser()
    SS_data = load_dataset(args.SS_val_path)
    SD_data = load_dataset(args.SD_val_path)
    DS_data = load_dataset(args.DS_val_path)
    DD_data = load_dataset(args.DD_val_path)
    print()
    print(args.model_name)
    # load models
    if args.model_name == "LUAR":
        model = SentenceTransformer("gabrielloiseau/LUAR-MUD-sentence-transformers")
    elif args.model_name == "StyleDistance":
        model = SentenceTransformer("StyleDistance/styledistance") 
    elif args.model_name == "ModernBERT":
        model = SentenceTransformer('gabrielloiseau/ModernBERT-base-authorship-verification')
    else:
        print("Model name not recognised, choose one of: 'LUAR', 'StyleDistance', 'ModernBERT'")
 
    y_trues = []
    y_preds = []
    if args.model_name == 'ModernBERT':
        print(args.title)
        for data in [SS_data, SD_data, DS_data, DD_data]:
            y_true, y_pred = evaluate_model(data, model, modern_bert=True)
            y_trues.append(y_true)
            y_preds.append(y_pred)
    else: 
        print(args.title)
        for data in [SS_data, SD_data, DS_data, DD_data]:
            y_true, y_pred = evaluate_model(data, model)
            y_trues.append(y_true)
            y_preds.append(y_pred)
    best_threshold, thresholds, stds, accuracies = find_optimal_threshold(y_trues, y_preds)
    thresholds_plot(best_threshold, thresholds, stds, accuracies, args.model_name, args.dataset_name)
    print(f"The best threshold for {args.model_name} is {best_threshold}") # print optimal threshold
