import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import argparse
from collections import defaultdict
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from explainableAV.utils.utils import load_dataset
from explainableAV.utils.plot_utils import data_path

def create_np_arrays(data, mask_types, baseline=False):
    '''
    Create np arrays with difference between original and masked accuracy
    Inputs:
        data: model performance results
        mask_types: perturbation techniques
        baseline: whether to compute it for the baselines or not
    Output:
        differences as np arrays per pair type
    '''
    results = defaultdict(list) 
    pair_types = ['SS', 'SD', 'DS', 'DD']

    for pair_type in pair_types:
        if baseline:
            new_array = np.array([(data[pair_type][mask_type]['accuracy'] - data[pair_type][f'{mask_type}_baseline']['accuracy']) if mask_type != 'llm' else (data[pair_type][mask_type]['accuracy'] - data[pair_type]['original']['accuracy']) for mask_type in mask_types])
        else:
            new_array = np.array([(data[pair_type][mask_type]['accuracy'] - data[pair_type]['original']['accuracy']) for mask_type in mask_types])
        results[pair_type] = new_array

    return results

def prediction_swap_plot(model_name, results, ax, vmin, vmax, experiment, dataset_name, baseline):
    '''
    Plot heatmap per model
    Inputs:
        model_name: model name
        results: corresponding result values
        ax: which ax of the plot
        vmin: minimum value of the plot
        vmax: maximum value of the plot
        experiment: dual or single-sided perturbation
        dataset_name: dataset name
        baseline: whether to compare with the baseline or not
    '''

    pert_techs = ["Asterisk", "POS tag", "One word"]
    if experiment == 'first':
        pert_techs.append("Swap")
        if dataset_name == 'amazon':
            pert_techs.append("LLM")

    results["Perturbation Technique"] = pert_techs

    df = pd.DataFrame(results)
    df.set_index("Perturbation Technique", inplace=True)

    custom_cmap = LinearSegmentedColormap.from_list("strong_redblue", ["#0000ff", "#ff0000"])

    custom_cmap = LinearSegmentedColormap.from_list(
        "jump_puor",
        [
            (0.0, "#0000ff"),   # strong purple
            (0.5, "#b6cbfc"),  # still strong purple near 0
            (0.500000000000001, "#fcb6b6"),  # neutral continues
            (1.0, "#ff0000")    # strong orange continues
        ],
        N=256
    )
    max_abs = np.max([abs(vmin), abs(vmax)])
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)


    annot_data = df.applymap(lambda x: f"+{x:.2f}" if x >= 0 else (f"{x:.2f}" if x < 0 else f"{x:.2f}"))

    sns.heatmap(df, annot=annot_data, fmt='', cmap=custom_cmap, linewidths=0.5, ax=ax, 
                annot_kws={"size": 18, "color": "black"}, cbar=False, norm=norm)

    if dataset_name == 'amazon':
            name = "Amazon"
    elif dataset_name == 'pan20':
        name = 'PAN20'
    
    ax.set_title(f"{model_name} ({name})", fontsize=22)
    ax.set_xlabel("Pair Type", fontsize=20)
    ax.set_ylabel("Perturbation Technique", fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=14)

def plot_all_models(luar_results, modernbert_results, styledistance_results, experiment, dataset_name, baseline):
    '''
    Plot heatmap for behavioral experiment
    Inputs:
        luar_results: result values for LUAR
        modernbert_results: result values for ModernBERT
        styledistance_results: result values for StyleDistance
        experiment: dual or single-sided perturbation
        dataset_name: dataset name
        baseline: whether to compare with the baseline or not
    '''
    if experiment == 'both':
        fig, axs = plt.subplots(1, 3, figsize=(20, 4), sharey=False)
    else:
        fig, axs = plt.subplots(1, 3, figsize=(20, 6), sharey=False)

    min_luar = min(accuracy for accuracies in luar_results.values() for accuracy in accuracies)
    min_modernbert = min(accuracy for accuracies in modernbert_results.values() for accuracy in accuracies)
    min_styledistance = min(accuracy for accuracies in styledistance_results.values() for accuracy in accuracies)

    max_luar = max(accuracy for accuracies in luar_results.values() for accuracy in accuracies)
    max_modernbert = max(accuracy for accuracies in modernbert_results.values() for accuracy in accuracies)
    max_styledistance = max(accuracy for accuracies in styledistance_results.values() for accuracy in accuracies)

    global_vmin = min(min_luar, min_modernbert, min_styledistance)
    global_vmax = max(max_luar, max_modernbert, max_styledistance)

    prediction_swap_plot("LUAR", luar_results, axs[0], global_vmin, global_vmax, experiment, dataset_name, baseline)
    prediction_swap_plot("ModernBERT", modernbert_results, axs[1], global_vmin, global_vmax, experiment, dataset_name, baseline)
    prediction_swap_plot("StyleDistance", styledistance_results, axs[2], global_vmin, global_vmax, experiment, dataset_name, baseline)

    axs[0].set_ylabel("Perturbation Technique", fontsize=20, labelpad=10)
    
    axs[1].get_yaxis().set_visible(False)
    axs[2].get_yaxis().set_visible(False)

    fig.subplots_adjust(wspace=0.1)
    
    custom_cmap = LinearSegmentedColormap.from_list("strong_puor", ["#5E00B5", "#FF8500"])
    cbar = fig.colorbar(axs[1].collections[0], ax=axs, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label('Accuracy Difference', fontsize=20)
    if baseline:
        plt.savefig(f"explainableAV/results/predictions/heatmaps_mask_{experiment}_{dataset_name}_baseline.pdf", dpi=300, bbox_inches='tight')
    else:
        plt.savefig(f"explainableAV/results/predictions/heatmaps_mask_{experiment}_{dataset_name}.pdf", dpi=300, bbox_inches='tight')

def true_positives(data, model, dictionary, mask_types, num_elems):
    '''
    Compute TPs
    Inputs:
        data: prediction accuracy
        model: model name
        dictionary: dictionary to store results
        mask_types: perturbation techniques
        num_elems: number of text pairs
    Output:
        dictionary
    '''
    dictionary[model] = [(data['SS'][mask_type]['accuracy'] * num_elems + data['SD'][mask_type]['accuracy'] * num_elems) for mask_type in mask_types]
    return dictionary

def false_positives(data, model, dictionary, mask_types, num_elems):
    '''
    Compute FPs
    Inputs:
        data: prediction accuracy
        model: model name
        dictionary: dictionary to store results
        mask_types: perturbation techniques
        num_elems: number of text pairs
    Output:
        dictionary
    '''
    dictionary[model] = [((1 - data['DS'][mask_type]['accuracy']) * num_elems + (1 - data['DD'][mask_type]['accuracy']) * num_elems) for mask_type in mask_types]
    return dictionary

def true_negatives(data, model, dictionary, mask_types, num_elems):
    '''
    Compute TNs
    Inputs:
        data: prediction accuracy
        model: model name
        dictionary: dictionary to store results
        mask_types: perturbation techniques
        num_elems: number of text pairs
    Output:
        dictionary
    '''
    dictionary[model] = [(data['DS'][mask_type]['accuracy'] * num_elems + data['DD'][mask_type]['accuracy'] * num_elems) for mask_type in mask_types]
    return dictionary

def false_negatives(data, model, dictionary, mask_types, num_elems):
    '''
    Compute FNs
    Inputs:
        data: prediction accuracy
        model: model name
        dictionary: dictionary to store results
        mask_types: perturbation techniques
        num_elems: number of text pairs
    Output:
        dictionary
    '''
    dictionary[model] = [((1 - data['SS'][mask_type]['accuracy']) * num_elems + (1 - data['SD'][mask_type]['accuracy']) * num_elems) for mask_type in mask_types]
    return dictionary

def confusion_matrix(models_dict, data_name, experiment, baseline, type='TP'):
    '''
    Compute elements of the confusion matrix
    Inputs:
        models_dict: dict with predictions
        data_name: dataset name
        experiment: dual or single sided experiment
        baseline: whether to compute it with regards to the baseline or not
        type: which element of the confusion matrix to compute
    Output:
        dictionary with results
    '''
    
    num_elems = len(models_dict['LUAR']['SS']['original']['predictions'])
    mask_types = ['asterisk', 'pos tag', 'one word']
    if experiment == 'first':
        mask_types.append('change topic')
        if data_name == 'amazon':
            mask_types.append('llm')
    mask_types_advanced = []
    if baseline:
        for mask_type in mask_types:
            if mask_type != 'llm':
                mask_types_advanced.append(f"{mask_type}_baseline")
            else:
                mask_types_advanced.append("original")
        mask_types_advanced = ['original'] + mask_types_advanced
    else:
        for mask_type in mask_types:
            mask_types_advanced.append(f"original")
    mask_types_advanced += mask_types
    
    dictionary = defaultdict(list)
    if type == 'TP':
        for model in models_dict.keys():
            dictionary = true_positives(models_dict[model], model, dictionary, mask_types_advanced, num_elems)
    elif type == 'FP':
        for model in models_dict.keys():
            dictionary = false_positives(models_dict[model], model, dictionary, mask_types_advanced, num_elems)
    elif type == 'TN':
        for model in models_dict.keys():
            dictionary = true_negatives(models_dict[model], model, dictionary, mask_types_advanced, num_elems)
    elif type == 'FN':
        for model in models_dict.keys():
            dictionary = false_negatives(models_dict[model], model, dictionary, mask_types_advanced, num_elems)
    return dictionary

def confusion(TP, FP, TN, FN, dataset_name, experiment, baseline):
    '''
    Plot confusion matrix elements
    Inputs:
        TP: true positives
        FP: false positives
        TN: true negatives
        FN: false negatives
        dataset_name: dataset name
        experiment: dual or single-sided perturbation
        baseline: whether to plot new baseline
    '''
    conditions = ["Asterisk", "POS tag", "One word"]
    if experiment == 'first':
        conditions.append("Swap")
        if dataset_name == 'amazon':
            conditions.append('LLM')

    models = list(TP.keys())
    n_models = len(models)

    bar_width = 0.2
    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 3), squeeze=False, sharey=True)
    axes = axes[0]

    custom_cmap = LinearSegmentedColormap.from_list("strong_puor", ["#5E00B5", "#FF8500"])

    colors = {
        "TP": custom_cmap(0.05),   
        "FP": custom_cmap(0.25), 
        "TN": custom_cmap(0.75), 
        "FN": custom_cmap(0.95),
    }

    for col in range(n_models):
        ax = axes[col]
        model = models[col]

        num_vals = int(len(TP[model])/2)
        x = np.arange(num_vals)

        ax.bar(x - 1.5 * bar_width, TP[model][num_vals+1:], width=bar_width, label="TP", color=colors["TP"], alpha=0.6)
        ax.bar(x - 0.5 * bar_width, FP[model][num_vals+1:], width=bar_width, label="FP", color=colors["FP"], alpha=0.6)
        ax.bar(x + 0.5 * bar_width, TN[model][num_vals+1:], width=bar_width, label="TN", color=colors["TN"], alpha=0.6)
        ax.bar(x + 1.5 * bar_width, FN[model][num_vals+1:], width=bar_width, label="FN", color=colors["FN"], alpha=0.6)

        baseline_len = bar_width
        for i, xi in enumerate(x):
            ax.hlines(y=TP[model][i+1],
                      xmin=xi - 1.5 * bar_width - baseline_len / 2,
                      xmax=xi - 1.5 * bar_width,
                      colors="red", linestyles='-', linewidth=1.5, alpha=0.9)
            ax.hlines(y=FP[model][i+1],
                      xmin=xi - 0.5 * bar_width - baseline_len / 2,
                      xmax=xi - 0.5 * bar_width,
                      colors="red", linestyles='-', linewidth=1.5, alpha=0.9)
            ax.hlines(y=TN[model][i+1],
                      xmin=xi + 0.5 * bar_width - baseline_len / 2,
                      xmax=xi + 0.5 * bar_width,
                      colors="red", linestyles='-', linewidth=1.5, alpha=0.9)
            ax.hlines(y=FN[model][i+1],
                      xmin=xi + 1.5 * bar_width - baseline_len / 2,
                      xmax=xi + 1.5 * bar_width,
                      colors="red", linestyles='-', linewidth=1.5, alpha=0.9)
            ax.hlines(y=TP[model][0],
                        xmin=xi - 1.5 * bar_width - baseline_len / 2,
                        xmax=xi - 1.5 * bar_width,
                        colors="black", linestyles='-', linewidth=1.5, alpha=0.9)
            ax.hlines(y=FP[model][0],
                        xmin=xi - 0.5 * bar_width - baseline_len / 2,
                        xmax=xi - 0.5 * bar_width,
                        colors="black", linestyles='-', linewidth=1.5, alpha=0.9)
            ax.hlines(y=TN[model][0],
                        xmin=xi + 0.5 * bar_width - baseline_len / 2,
                        xmax=xi + 0.5 * bar_width,
                        colors="black", linestyles='-', linewidth=1.5, alpha=0.9)
            ax.hlines(y=FN[model][0],
                        xmin=xi + 1.5 * bar_width - baseline_len / 2,
                        xmax=xi + 1.5 * bar_width,
                        colors="black", linestyles='-', linewidth=1.5, alpha=0.9)

        ax.set_xticks(x)
        ax.set_xticklabels(conditions, rotation=0, ha='center', fontsize=10)
        if dataset_name == 'amazon':
            name = "Amazon"
        elif dataset_name == 'pan20':
            name = 'PAN20'
        ax.set_title(f"{model} ({name})", fontsize=16)
        ax.set_xlabel("Perturbation Technique", fontsize=14)
        ax.tick_params(axis='y', labelsize=10)

        if col == 0:
            ax.set_ylabel("Number of Pairs", fontsize=14)
            if dataset_name == 'amazon':
                ax.set_yticks(range(0, int(max(TP[model] + FP[model] + TN[model] + FN[model])) + 5000, 5000))
            elif dataset_name == 'pan20':
                ax.set_yticks(range(0, int(max(TP[model] + FP[model] + TN[model] + FN[model])) + 1000, 1000))

    baseline_line = Line2D([0], [0], color='black', lw=1.5, label='Original Baseline')
    baseline_line2 = Line2D([0], [0], color='red', lw=1.5, label='Perturbation-Specific Baseline')

    handles, labels = ax.get_legend_handles_labels()
    handles.append(baseline_line)
    labels.append('Original Baseline')
    handles.append(baseline_line2)
    labels.append('Perturbation-Specific Baseline')
    fig.legend(
        handles, labels,
        loc='lower center',
        ncol=8,
        fontsize=14,
        bbox_to_anchor=(0.5, -0.02),
        borderaxespad=0.
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    if baseline:
        plt.savefig(f"explainableAV/results/predictions/confusion_{experiment}_plot_{dataset_name}_baseline.pdf", dpi=300, bbox_inches='tight')
    else:
        plt.savefig(f"explainableAV/results/predictions/confusion_{experiment}_plot_{dataset_name}.pdf", dpi=300, bbox_inches='tight')


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot_type', default='heatmaps', help='Choose from "heatmaps", "confusion"')
    parser.add_argument('--experiment', default='both', help='Choose from "both", "first"')
    parser.add_argument('--dataset_name', default='amazon', help='Choose from "amazon", "pan20"')
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--model_name', default='LUAR')
    parser.add_argument('--luar_results_path', default=None, help="Set results path if results named differently otherwise assume default save name")
    parser.add_argument('--modernbert_results_path', default=None, help="Set results path if results named differently otherwise assume default save name")
    parser.add_argument('--styledistance_results_path', default=None, help="Set results path if results named differently otherwise assume default save name")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = argument_parser()
    luar = load_dataset(data_path(f"explainableAV/results/predictions/{args.dataset_name}_LUAR_predictions_mask_{args.experiment}.json", args.luar_results_path))
    modernbert = load_dataset(data_path(f"explainableAV/results/predictions/{args.dataset_name}_ModernBERT_predictions_mask_{args.experiment}.json", args.modernbert_results_path))
    styledistance = load_dataset(data_path(f"explainableAV/results/predictions/{args.dataset_name}_StyleDistance_predictions_mask_{args.experiment}.json", args.styledistance_results_path))

    if args.plot_type == 'confusion':
        models = {"LUAR": luar, "ModernBERT": modernbert, "StyleDistance": styledistance}
        TP = confusion_matrix(models, args.dataset_name, args.experiment, args.baseline, type='TP')
        FP = confusion_matrix(models, args.dataset_name, args.experiment, args.baseline, type='FP')
        TN = confusion_matrix(models, args.dataset_name, args.experiment, args.baseline, type='TN')
        FN = confusion_matrix(models, args.dataset_name, args.experiment, args.baseline, type='FN')
        confusion(TP, FP, TN, FN, args.dataset_name, args.experiment, args.baseline)
    elif args.plot_type == 'heatmaps':
        mask_types = ['asterisk', 'pos tag', 'one word'] 
        if args.experiment == 'first':
            mask_types.append('change topic')
            if args.dataset_name == 'amazon':
                mask_types.append('llm')

        luar_results = create_np_arrays(luar, mask_types, args.baseline)
        modernbert_results = create_np_arrays(modernbert, mask_types, args.baseline)
        styledistance_results = create_np_arrays(styledistance, mask_types, args.baseline)
        plot_all_models(luar_results, modernbert_results, styledistance_results, args.experiment, args.dataset_name, args.baseline)

