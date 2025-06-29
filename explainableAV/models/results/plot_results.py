import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import argparse
from tabulate import tabulate
from scipy.stats import wilcoxon
from collections import defaultdict, Counter
from matplotlib.colors import rgb2hex
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from explainableAV.utils.utils import load_dataset
import math
from matplotlib.ticker import FuncFormatter
import re

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

def extract_metric_stats(mask_quality, metric, dataset):
    '''
    Extract masking quality values
    Inputs:
        mask_quality: dataset with mask quality results
        metric: metric to compute it for 'semantic similarity etc'
        dataset: dataset name
    Output:
        mean and standard deviation of requested metric
    '''
    strategies = ['asterisk', 'pos tag', 'one word', 'change topic']
    if dataset == 'amazon':
        strategies.append('llm')
    means = [mask_quality[strategy][metric]['mean'] for strategy in strategies]
    std_devs = [mask_quality[strategy][metric]['std_dev'] for strategy in strategies]
    return means, std_devs

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
        plt.savefig(f"explainableAV/models/results/images/heatmaps_mask_{experiment}_{dataset_name}_baseline.pdf", dpi=300, bbox_inches='tight')
    else:
        plt.savefig(f"explainableAV/models/results/images/heatmaps_mask_{experiment}_{dataset_name}.pdf", dpi=300, bbox_inches='tight')

def text_perturbation_plot(syntax_mean, syntax_std, semantic_mean, semantic_std, perplexity_mean, perplexity_std, dataset_name):
    '''
    Plot mask quality plots
    Inputs:
        syntax_mean: mean of syntactical similarity
        syntax_std: standard deviation of syntactical similarity
        semantic_mean: mean of semantic similarity
        semantic_std: standard deviation of semantic similarity
        perplexity_mean: mean of perplexity
        perplexity_std: standard deviation of perplexity
        dataset_name: dataset name
    '''
    mask_types = ["Asterisk", "POS tag", "One word", "Swap", "LLM"]
    if dataset_name == 'amazon':
        data_name = 'Amazon'
    else:
        data_name = 'PAN20'
        syntax_mean.append(0)
        syntax_std.append(0)
        semantic_mean.append(0)
        semantic_std.append(0)
        perplexity_mean.append(0)
        perplexity_std.append(0)


    plt.figure(figsize=(8, 5))
    x = np.arange(len(mask_types))
    width = 0.4
    custom_cmap = LinearSegmentedColormap.from_list("strong_puor", ["#5E00B5", "#FF8500"])

    plt.bar(x - width/2, syntax_mean, width, yerr=syntax_std, capsize=8, label=r'$\uparrow$Syntactical Similarity', color=custom_cmap(0.25), alpha=0.6, edgecolor='black', linewidth=1)
    plt.bar(x + width/2, semantic_mean, width, yerr=semantic_std, capsize=8, label=r'$\downarrow$Semantic Similarity', color=custom_cmap(0.95), alpha=0.6, edgecolor='black', linewidth=1)
    plt.xticks(x, mask_types)

    plt.ylabel("Similarity Score", fontsize=18)
    plt.title(f"{data_name}", fontsize=20)
    plt.xticks(fontsize=16)
    plt.xlabel('Perturbation Technique', fontsize=18)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16, loc='lower left')
    plt.savefig(f"explainableAV/models/results/images/similarity_plot_{dataset_name}.pdf", dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(8, 5))
    x = np.arange(len(mask_types))
    plt.xlim(min(x) - 0.5, max(x) + 0.5)

    if data_name == "PAN20":
        plt.errorbar(x[:4], perplexity_mean[:4], yerr=perplexity_std[:4], fmt='o', mfc=custom_cmap(0.50), mec='black', ecolor='black', capsize=5, elinewidth=2, markersize=14, capthick=2)
    else:
        plt.errorbar(x, perplexity_mean, yerr=perplexity_std, fmt='o', mfc=custom_cmap(0.50), mec='black', ecolor='black', capsize=5, elinewidth=2, markersize=14, capthick=2)
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=2)

    plt.xticks(x, mask_types)
    plt.xticks(fontsize=16)
    plt.xlabel('Perturbation Technique', fontsize=18)
    plt.yticks(fontsize=16)
    plt.ylabel('Standard Deviation', fontsize=18)
    plt.title(f"{data_name}", fontsize=20)
    plt.savefig(f"explainableAV/models/results/images/perplexity_plot_{dataset_name}.pdf", dpi=300, bbox_inches='tight')
    plt.show()

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
        mask_types.append('swap')
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
        plt.savefig(f"explainableAV/models/results/images/confusion_{experiment}_plot_{dataset_name}_baseline.pdf", dpi=300, bbox_inches='tight')
    else:
        plt.savefig(f"explainableAV/models/results/images/confusion_{experiment}_plot_{dataset_name}.pdf", dpi=300, bbox_inches='tight')


def statistical_test(data, mask_types):
    '''
    Compute and print statistical tests for predictions
    Inputs:
        data: prediction results
        mask_types: perturbation techniques
    '''
    pair_types = ['SS', 'SD', 'DS', 'DD']

    for mask_type in mask_types:
        results = []
        for pair_type in pair_types:
            original = data[pair_type]['original']['predictions']
            perturbed = data[pair_type][mask_type]['predictions']

            stat, p_value = wilcoxon(original, perturbed)
            
            if p_value > 0.05:
                p_str = "ns"
            elif p_value <= 0.0001:
                p_str = "****"
            elif p_value <= 0.001:
                p_str = "***"
            elif p_value <= 0.01:
                p_str = "**"
            elif p_value <= 0.05:
                p_str = "*"

            results.append([pair_type, f"{stat:.2f}", p_str])

        print(f"\n### Perturbation: {mask_type}")
        print(tabulate(results, headers=["Pair Type", "Wilcoxon Stat", "p-value"], tablefmt="github"))


def plot_lda_results(data_name):
    '''
    Plot LDA evaluation
    Input:
        data_name: dataset name
    '''
    if data_name == 'amazon':
        inter_topic = [0.8221040474690441, 0.8071133977948014, 0.8049904249225951, 0.8094604941912703, 0.8165027216994806,
                       0.8261435980722225, 0.8377352854322951, 0.8486925835088309, 0.8592311947540534, 0.8697164422721183, 
                       0.877915753935596, 0.8816306520056674, 0.8793846515488714,  0.8742942907531, 0.8648754808591309,
                       0.8526694173333215, 0.8390757279521652, 0.8268401651821644, 0.8153585848303524, 0.8054527801709418]
        mask_percentage = [0.11564134611873401, 0.13171815242559182, 0.13933694196683577, 0.14329933789725915, 0.14561782377761942,
                           0.14717656879663668, 0.14826446804737986, 0.1491264525709453, 0.14961600808945794, 0.14992453611647258,
                           0.15018848538620902, 0.15038461482133053, 0.15062214619925493, 0.15081035301869436, 0.15109898171947325,
                           0.15131531757551808, 0.1515160602460532, 0.1518189669432677, 0.1519962953863353, 0.15219829026567805]
        topic_related_size = [700, 1400, 2100, 2800, 3500, 4200, 4900, 5600, 6300, 7000, 7700, 8400, 9100, 9800, 10500, 11200, 11900, 12600, 13300, 14000]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    custom_cmap = LinearSegmentedColormap.from_list("strong_puor", ["#5E00B5", "#FF8500"])

    ax1.plot(topic_related_size, inter_topic, 'o-', color=custom_cmap(0.25), label='Average Inter-Topic Distance')
    ax1.set_xlabel("Number of Topic Words", fontsize=18)
    ax1.set_ylabel("Average Inter-Topic Distance", color=custom_cmap(0.25), fontsize=18)
    ax1.tick_params(axis='y', labelcolor=custom_cmap(0.25), labelsize=16)
    ax1.set_xticks(topic_related_size)
    ax1.tick_params(axis='x', labelrotation=45, labelsize=14) 

    ax2 = ax1.twinx()
    ax2.plot(topic_related_size, mask_percentage, 's-', color=custom_cmap(0.95), label='Average Mask Percentage')
    ax2.set_ylabel("Average Percentage of Altered Text", color=custom_cmap(0.95), fontsize=18)
    ax2.tick_params(axis='y', labelcolor=custom_cmap(0.95), labelsize=16)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='lower center', bbox_to_anchor=(0.57, 0), fontsize=14)

    if data_name == 'amazon':
        plt.title('Guided LDA (Amazon)', fontsize=24)

    plt.savefig(f"explainableAV/models/results/images/lda_{data_name}.pdf", dpi=300, bbox_inches='tight')


def probing_accuracy(metric_results_LUAR, metric_results_ModernBERT, metric_results_StyleDistance):
    '''
    Plot line plot with probing accuracy
    Inputs:
        metric_results_LUAR: probing results dictionary for LUAR
        metric_results_ModernBERT: probing results dictionary for ModernBERT
        metric_results_StyleDistance: probing results dictionary for StyleDistance
    '''
    fig, axs = plt.subplots(1, 3, figsize=(18, 4), sharey=True)
    custom_cmap = LinearSegmentedColormap.from_list("strong_puor", ["#5E00B5", "#FF8500"])

    colors = {
        "Test_accuracy": custom_cmap(0.05),
        "Pre-trained Accuracy": custom_cmap(0.5),
        "Test_accuracy_masked": custom_cmap(0.95),
    }

    model_names = ['LUAR', 'ModernBERT', 'StyleDistance']
    handles = []
    labels = ["Fine-Tuned Test", "Pre-trained Test", "Fine-Tuned Test Masked"]

    for i, (model_results, model_name) in enumerate(zip([metric_results_LUAR, metric_results_ModernBERT, metric_results_StyleDistance], model_names)):
        layers = list(range(1, len(model_results["Test_accuracy"])+1))

        for j, (key, color) in enumerate(colors.items()):
            metric_dict = model_results[key]
            y_values = [metric_dict[str(layer)] for layer in layers]
            line, = axs[i].plot(
                layers,
                y_values,
                label=key.replace("_", " ").title() if i == 0 else "",
                color=color,
                marker='o',
                linewidth=2.5,
                markersize=8
            )
            if i == 0:
                handles.append(line)

        if i == 1:
            axs[i].set_xticks([layer for layer in layers if layer % 2 == 0])
        else:
            axs[i].set_xticks(layers)
        axs[i].tick_params(axis='x', labelsize=16)
        axs[i].set_xlabel("Layer", fontsize=18)
        axs[i].tick_params(axis='y', labelsize=16)
        if i == 0:
            axs[i].set_ylabel("Accuracy", fontsize=18)
        axs[i].set_title(f"{model_name}", fontsize=20)

    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.08), fontsize=22)

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(f"explainableAV/models/results/images/probing_accuracy.pdf", dpi=300, bbox_inches='tight')
    plt.show()


def average_layers(metric, n_groups):
    '''
    For probing results average over consecutive layers
    Inputs:
        metric: which metric
        n_groups: how groups to have
    Output:
        grouped results
    '''
    total_layers = len(metric)
    group_size = total_layers // n_groups
    remainder = total_layers % n_groups
    
    new_metric = {}
    start = 1
    for i in range(n_groups):
        end = start + group_size + (1 if i < remainder else 0)
        values = []
        for layer in range(start, end):
            values.append(metric[str(layer)])
        if start == end - 1:
            new_metric[f"{start}"] = np.mean(values, axis=0)
        else:
            new_metric[f"{start}-{end - 1}"] = np.mean(values, axis=0)
        start = end
    
    return new_metric

def single_probing_heatmap(metric_name, metric_values, y_labels, ax, vmin, vmax):
    '''
    Probing heatmap for one metric
    Inputs:
        metric_name: which metric
        metric_values: results for that metric
        y_labels: topic names
        ax: which ax
        vmin: minimum value of heatmap
        vmax: maximum value of heatmap
    '''
    metric_values["Topic Names"] = y_labels

    df = pd.DataFrame(metric_values)
    df.set_index("Topic Names", inplace=True)

    custom_cmap = LinearSegmentedColormap.from_list("strong_puor", ["#5E00B5", "#FF8500"])
    annot_data = df.applymap(lambda x: f"{x:.2f}" if x >= 0 else (f"{x:.2f}" if x < 0 else f"{x:.2f}"))

    sns.heatmap(df, annot=annot_data, fmt='', cmap=custom_cmap, linewidths=0.5, ax=ax, 
                annot_kws={"size": 24, "color": "white"}, cbar=False, vmin=vmin, vmax=vmax)
    
    ax.set_title(f"{metric_name}", fontsize=32)
    ax.set_xlabel("Layer", fontsize=28)
    ax.set_ylabel("Topic Names", fontsize=28)
    ax.tick_params(axis='both', which='major', labelsize=24)

def probing_per_layer_heatmap(metric_results, model_name):
    '''
    Plot probing results for all metrics per topic
    Inputs:
        metric_results: all probing results
        model_name: name of the model
    '''
    fig, axs = plt.subplots(1, 3, figsize=(30, 12), sharey=False)

    precisions = metric_results['Precision']
    recalls = metric_results['Recall']
    f1_scores = metric_results['F1-scores']
    precisions = average_layers(precisions, 6)
    recalls = average_layers(recalls, 6)
    f1_scores = average_layers(f1_scores, 6)

    min_precision = min(prec for prec_list in precisions.values() for prec in prec_list)
    min_recall = min(rec for rec_list in recalls.values() for rec in rec_list)
    min_f1 = min(f1 for f1_list in f1_scores.values() for f1 in f1_list)

    max_precision = max(prec for prec_list in precisions.values() for prec in prec_list)
    max_recall = max(rec for rec_list in recalls.values() for rec in rec_list)
    max_f1 = max(f1 for f1_list in f1_scores.values() for f1 in f1_list)

    global_vmin = min(min_precision, min_recall, min_f1)
    global_vmax = max(max_precision, max_recall, max_f1)

    single_probing_heatmap(f"Precision ({model_name})", precisions, metric_results['Label_names'], axs[0], global_vmin, global_vmax)
    single_probing_heatmap(f"Recall ({model_name})", recalls, metric_results['Label_names'], axs[1], global_vmin, global_vmax)
    single_probing_heatmap(f"F1-score ({model_name})", f1_scores, metric_results['Label_names'], axs[2], global_vmin, global_vmax)

    axs[0].set_ylabel("Topic Names", fontsize=28, labelpad=10)
    axs[1].get_yaxis().set_visible(False)
    axs[2].get_yaxis().set_visible(False)

    fig.subplots_adjust(wspace=0.1)
    
    custom_cmap = LinearSegmentedColormap.from_list("strong_puor", ["#5E00B5", "#FF8500"])
    cbar = fig.colorbar(axs[1].collections[0], ax=axs, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.ax.tick_params(labelsize=24)
    plt.savefig(f"explainableAV/models/results/images/heatmaps_probing_{model_name}.pdf", dpi=300, bbox_inches='tight')

def probing_f1_only(metric_results_LUAR, metric_results_ModernBERT, metric_results_StyleDistance):
    '''
    Plot probing results for F1 scores per topic
    Inputs:
        metric_results_LUAR: probing results for LUAR
        metric_results_ModernBERT: probing results for ModernBERT
        metric_results_StyleDistance: probing results for StyleDistance
    '''
    fig, axs = plt.subplots(1, 3, figsize=(30, 12), sharey=False)

    LUAR = metric_results_LUAR['F1-scores_masked']
    ModernBERT = metric_results_ModernBERT['F1-scores_masked']
    StyleDistance = metric_results_StyleDistance['F1-scores_masked']
    LUAR = average_layers(LUAR, 6)
    ModernBERT = average_layers(ModernBERT, 6)
    StyleDistance = average_layers(StyleDistance, 6)

    min_LUAR = min(luar for luar_list in LUAR.values() for luar in luar_list)
    min_ModernBERT = min(modernbert for modernbert_list in ModernBERT.values() for modernbert in modernbert_list)
    min_StyleDistance = min(styledistance for styledistance_list in StyleDistance.values() for styledistance in styledistance_list)

    max_LUAR = max(luar for luar_list in LUAR.values() for luar in luar_list)
    max_ModernBERT = max(modernbert for modernbert_list in ModernBERT.values() for modernbert in modernbert_list)
    max_StyleDistance = max(styledistance for styledistance_list in StyleDistance.values() for styledistance in styledistance_list)

    global_vmin = min(min_LUAR, min_ModernBERT, min_StyleDistance)
    global_vmax = max(max_LUAR, max_ModernBERT, max_StyleDistance)

    single_probing_heatmap("LUAR", LUAR, metric_results_LUAR['Label_names'], axs[0], global_vmin, global_vmax)
    single_probing_heatmap("ModernBERT", ModernBERT, metric_results_ModernBERT['Label_names'], axs[1], global_vmin, global_vmax)
    single_probing_heatmap("StyleDistance", StyleDistance, metric_results_StyleDistance['Label_names'], axs[2], global_vmin, global_vmax)

    axs[0].set_ylabel("Topic Names", fontsize=28, labelpad=10)
    axs[1].get_yaxis().set_visible(False)
    axs[2].get_yaxis().set_visible(False)

    fig.subplots_adjust(wspace=0.1)
    
    custom_cmap = LinearSegmentedColormap.from_list("strong_puor", ["#5E00B5", "#FF8500"])
    cbar = fig.colorbar(axs[1].collections[0], ax=axs, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.ax.tick_params(labelsize=24)
    plt.savefig(f"explainableAV/models/results/images/heatmaps_probing_f1_only.pdf", dpi=300, bbox_inches='tight')

def probing_learning_curve(metric_results, model_name):
    '''
    Plot probing learning curves for one model
    Inputs:
        metric_results: probing results
        model_name: name of the model
    '''
    custom_cmap = LinearSegmentedColormap.from_list("strong_puor", ["#5E00B5", "#FF8500"])

    colors = {
        "Train_loss": custom_cmap(0.05),
        "Val_loss": custom_cmap(0.95),
    }

    layer_names = list(range(1, len(list(metric_results.keys())) + 1))
    num_layers = len(layer_names)

    max_cols = 6
    num_cols = min(num_layers, max_cols)
    num_rows = math.ceil(num_layers / max_cols)

    subplot_width = 3.5
    subplot_height = 3.5
    fig_width = subplot_width * num_cols
    fig_height = subplot_height * num_rows

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height), sharey=True)
    axs = axs.flatten()

    handles = []
    labels = []

    for i, layer in enumerate(layer_names):
        ax = axs[i]
        for j, (key, color) in enumerate(colors.items()):
            metric_dict = metric_results[str(layer)][key]
            epochs = list(range(len(metric_dict)))
            line, = ax.plot(
                epochs,
                metric_dict,
                label=key.replace("_", " ").title() if i == 0 else "",
                color=color,
            )
            if i == 0:
                handles.append(line)
                labels.append(key.replace("_", " ").title())

        ax.set_xlabel("Epoch", fontsize=18)
        ax.tick_params(axis='y', labelsize=18)
        ax.tick_params(axis='x', labelsize=18)
        if i % max_cols == 0:
            ax.set_ylabel("Loss", fontsize=18)
        ax.set_title(f"Layer {layer}", fontsize=18)

    for j in range(len(layer_names), len(axs)):
        fig.delaxes(axs[j])

    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08), fontsize=24)

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(f"explainableAV/models/results/images/probing_learning_curve_{model_name}.pdf", dpi=300, bbox_inches='tight')
    plt.show()

def top_k_words_plot(data_LUAR, data_ModernBERT, data_StyleDistance):
    '''
    Plot topic coverage results
    Inputs:
        data_LUAR: results for LUAR
        data_ModernBERT: results for ModernBERT
        data_StyleDistance: results for StyleDistance
    '''
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    custom_cmap = LinearSegmentedColormap.from_list("strong_puor", ["#5E00B5", "#FF8500"])
    colors = [custom_cmap(0.05), custom_cmap(0.50), custom_cmap(0.95)]

    for i, (model, color) in enumerate(zip(['LUAR', 'ModernBERT', 'StyleDistance'], colors)):
        ax = axes[i]
        if model == 'LUAR':
            data_specific = data_LUAR['top-k']
        elif model == 'ModernBERT':
            data_specific = data_ModernBERT['top-k']
        elif model == 'StyleDistance':
            data_specific = data_StyleDistance['top-k']
        top_words = np.mean(data_specific, axis=0)
        top_words_std = np.std(data_specific, axis=0)

        layers = np.arange(1, len(top_words) + 1)
        lower_error = np.minimum(top_words, top_words_std)
        upper_error = top_words_std

        asymmetric_error = np.array([lower_error, upper_error])

        ax.errorbar(layers, top_words, yerr=asymmetric_error, fmt='-o', color=color,
                    ecolor='gray', capsize=4, elinewidth=1.5, markerfacecolor='white', markersize=10, linewidth=3, markeredgewidth=2.5, markeredgecolor=color)

        if model == 'ModernBERT':
            even_indices = [idx for idx, layer in enumerate(layers) if layer % 2 == 0]
            even_labels = [layer for layer in layers if layer % 2 == 0]
            ax.set_xticks(even_labels)
            ax.set_xticklabels(even_labels, fontsize=18)
        else:
            ax.set_xticks(layers)
            ax.set_xticklabels(layers, fontsize=18)

        ax.set_xlabel('Layer', fontsize=20)
        ax.set_title(f'{model}', fontsize=22)
        ax.tick_params(axis='y', labelsize=18)
        ax.set_yticks(np.arange(0, 0.81, 0.1))

        if i == 0:
            ax.set_ylabel('Topic Coverage', fontsize=20)

    plt.tight_layout()
    plt.savefig('explainableAV/models/results/images/top_k.pdf')
    plt.show()


def top_k_ratio_plot(data_LUAR, data_ModernBERT, data_StyleDistance):
    '''
    Plot relative topic-attention ratio results
    Inputs:
        data_LUAR: results for LUAR
        data_ModernBERT: results for ModernBERT
        data_StyleDistance: results for StyleDistance
    '''
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    custom_cmap = LinearSegmentedColormap.from_list("strong_puor", ["#5E00B5", "#FF8500"])
    colors = [custom_cmap(0.05), custom_cmap(0.50), custom_cmap(0.95)]

    i = 0
    for model, color in zip(['LUAR', 'ModernBERT', 'StyleDistance'], colors):
        if model == 'LUAR':
            data_specific = data_LUAR['ratio']
        elif model == 'ModernBERT':
            data_specific = data_ModernBERT['ratio']
        elif model == 'StyleDistance':
            data_specific = data_StyleDistance['ratio']
        data_len = len(data_specific[0])
        x_pos = np.arange(data_len)
        layers = range(1, data_len+1)
        df = pd.DataFrame(data_specific, columns=[f'Layer {i}' for i in layers])
        df_long = df.melt(var_name='Layer', value_name='')

        ax = axes[i]
        sns.boxplot(data=df_long, x='Layer', y='', ax=ax, color=color, showfliers=False)

        if i == 1:
            even_indices = [idx for idx, layer in enumerate(layers) if layer % 2 == 0]
            even_labels = [layer for layer in layers if layer % 2 == 0]

            ax.set_xticks(even_indices)
            ax.set_xticklabels(even_labels, fontsize=18)
        else:
            ax.set_xticks(x_pos)
            ax.set_xticklabels(layers, fontsize=18)
        
        ax.set_xlabel('Layer', fontsize=20)
        ax.set_title(f'{model}', fontsize=22)
        if i == 0:
            ax.set_ylabel('Relative Topic-Attention Ratio', fontsize=20)
        ax.tick_params(axis='y', labelsize=18)
        ax.set_ylim(-0.15, 0.15)
        i += 1

    plt.tight_layout()
    plt.savefig('explainableAV/models/results/images/top_ratio.pdf')
    plt.show()

def thousands_formatter(x):
    '''
    Clip results per thousand
    Inputs:
        x: x value to clip
    '''
    if x >= 1000:
        return f'{x/1000:.0f}k'
    return f'{int(x)}'

def topic_distribution(data, dataset, split='test'):
    '''
    Plot distribution of documents over the topics
    Inputs:
        data: text pairs
        dataset: dataset name
        split: 'test' or 'train'
    '''
    custom_cmap = LinearSegmentedColormap.from_list("strong_puor", ["#5E00B5", "#FF8500"])
    if dataset == 'amazon':
        dataset_name = 'Amazon'
        color = custom_cmap(0.05)
    elif dataset == 'pan20':
        dataset_name = 'PAN20'
        color = custom_cmap(0.95)
    topic_dict = defaultdict(int)
    for line in data:
        topic_dict[line["Topics"][0]] += 1
        topic_dict[line["Topics"][1]] += 1

    sorted_topics = dict(sorted(topic_dict.items(), key=lambda item: item[1], reverse=True))
    x = sorted_topics.keys()
    y = sorted_topics.values()

    plt.figure(figsize=(10, 10))

    ax = plt.gca()
    ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))

    if dataset == "pan20" and split=='test':
        ax.set_yticks(range(0, max(y) + 10, 10))
    elif dataset == 'amazon' and split=='test':
        ax.set_yticks(range(0, max(y) + 5000, 5000))
    elif dataset == "pan20" and split=='train':
        ax.set_yticks(range(0, max(y) + 100, 100))
    elif dataset == 'amazon' and split=='train':
        ax.set_yticks(range(0, max(y) + 50000, 50000))

    plt.bar(x, y, color=color, alpha=0.6)
    if dataset == "pan20" and split=='test':
        num_ticks = 12
        x = list(x)
        tick_positions = np.linspace(0, len(x) - 1, num=num_ticks, dtype=int)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([x[i] for i in tick_positions], rotation=90, fontsize=22)
        ax.set_xlabel("Topics", fontsize=26, labelpad=100)
    elif dataset == 'amazon' and split=='test':
        plt.xticks(np.arange(len(x)), x, rotation=90, fontsize=22)
        ax.set_xlabel("Topics", fontsize=26, labelpad=13)
    elif dataset == "pan20" and split=='train':
        num_ticks = 12
        x = list(x)
        tick_positions = list(np.linspace(1, len(x) - 1, num=num_ticks, dtype=int))
        tick_positions = np.array(tick_positions[:-1] + [tick_positions[-1] -1])
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([x[i] for i in tick_positions], rotation=90, fontsize=22)
        ax.set_xlabel("Topics", fontsize=26, labelpad=10)
    elif dataset == "amazon" and split=='train':
        plt.xticks(np.arange(len(x)), x, rotation=90, fontsize=22)
        ax.set_xlabel("Topics", fontsize=26, labelpad=13)
    plt.ylabel("Number of Documents", fontsize=26)
    plt.tick_params(axis='y', labelsize=22)
    plt.title(f"Topic Distribution ({dataset_name})", fontsize=28)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.5)
    plt.savefig(f'explainableAV/models/results/images/topic_distribution_{dataset}_{split}.pdf')    

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot_type', default='heatmaps', help='Choose from "heatmaps", "text_similarity", "topic_distribution", "confusion", "top_attention_words", "lda", "probing_heatmap", "probing_line_plot", "probing_learning_curve"')
    parser.add_argument('--experiment', default='both', help='Choose from "both", "first", "raw", "rollout", "value_zeroing"')
    parser.add_argument('--dataset_name', default='amazon', help='Choose from "amazon", "pan20"')
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--model_name', default='LUAR')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = argument_parser()
    if args.plot_type == 'lda':
        plot_lda_results(args.dataset_name)
    elif args.plot_type == 'text_similarity':
        mask_quality = load_dataset(f"explainableAV/change_topic/mask_quality_results_{args.dataset_name}.json") 
        syntax_mean, syntax_std = extract_metric_stats(mask_quality, 'syntax', args.dataset_name)
        semantic_mean, semantic_std = extract_metric_stats(mask_quality, 'semantic', args.dataset_name)
        perplexity_mean, perplexity_std = extract_metric_stats(mask_quality, 'perplexity', args.dataset_name)
        text_perturbation_plot(syntax_mean, syntax_std, semantic_mean, semantic_std, perplexity_mean, perplexity_std, args.dataset_name)
    elif args.plot_type == 'topic_distribution':
        if args.dataset_name == 'amazon':
            data = load_dataset("explainableAV/Amazon/test_set_15000x4.json")
            # data = load_dataset("explainableAV/Amazon/train_set_15000x4.json")
        elif args.dataset_name == 'pan20':
            data = load_dataset("explainableAV/PAN20/test_set_2500x4.json")
            # data = load_dataset("explainableAV/PAN20/train_set_2500x4.json")
        topic_distribution(data, args.dataset_name)#, split='train')
    elif args.plot_type == 'top_attention_words':
        data_LUAR = load_dataset(f"explainableAV/attention_top_LUAR_rollout_non_topic.json")
        data_ModernBERT = load_dataset(f"explainableAV/attention_top_ModernBERT_value_zeroing_non_topic.json")
        data_StyleDistance = load_dataset(f"explainableAV/attention_top_StyleDistance_raw_non_topic.json")
        top_k_words_plot(data_LUAR, data_ModernBERT, data_StyleDistance)
        top_k_ratio_plot(data_LUAR, data_ModernBERT, data_StyleDistance)
    elif args.plot_type == 'attention_faithfulness':
        raw = load_dataset(f"explainableAV/attention_faithfulness_raw.json") 
        rollout = load_dataset(f"explainableAV/attention_faithfulness_rollout.json") 
        globenc = load_dataset(f"explainableAV/attention_faithfulness_globenc.json") 
        value_zeroing = load_dataset(f"explainableAV/attention_faithfulness_value_zeroing_rollout_updated.json") 
        print_attention_faithfulness([raw, rollout, globenc, value_zeroing])
    elif args.plot_type == 'probing_heatmap':
        metric_results = load_dataset(f"explainableAV/probes/probing_metrics_{args.model_name}.json")
        probing_per_layer_heatmap(metric_results, args.model_name)
        # probing_f1_only_heatmap(metric_results, args.model_name)
    elif args.plot_type == 'probing_heatmap_f1':
        metric_results_LUAR = load_dataset(f"explainableAV/probes/probing_metrics_LUAR.json")
        metric_results_ModernBERT = load_dataset(f"explainableAV/probes/probing_metrics_ModernBERT.json")
        metric_results_StyleDistance = load_dataset(f"explainableAV/probes/probing_metrics_StyleDistance.json")
        probing_f1_only(metric_results_LUAR, metric_results_ModernBERT, metric_results_StyleDistance)
    elif args.plot_type == 'probing_line_plot':
        metric_results_LUAR = load_dataset(f"explainableAV/probes/probing_metrics_LUAR.json")
        metric_results_ModernBERT = load_dataset(f"explainableAV/probes/probing_metrics_ModernBERT.json")
        metric_results_StyleDistance = load_dataset(f"explainableAV/probes/probing_metrics_StyleDistance.json")
        probing_accuracy(metric_results_LUAR, metric_results_ModernBERT, metric_results_StyleDistance)
    elif args.plot_type == 'probing_learning_curve':
        metric_results = load_dataset(f"explainableAV/probes/probing_losses_{args.model_name}.json")
        probing_learning_curve(metric_results, args.model_name)
    else:
        luar = load_dataset(f"explainableAV/models/results/{args.dataset_name}_LUAR_predictions_mask_{args.experiment}.json")
        modernbert = load_dataset(f"explainableAV/models/results/{args.dataset_name}_ModernBERT_predictions_mask_{args.experiment}.json")
        styledistance = load_dataset(f"explainableAV/models/results/{args.dataset_name}_StyleDistance_predictions_mask_{args.experiment}.json")

            # luar_pan20 = load_dataset(f"explainableAV/models/results/pan20_LUAR_predictions_mask_{experiment}.json")
            # modernbert_pan20 = load_dataset(f"explainableAV/models/results/pan20_ModernBERT_predictions_mask_{experiment}.json")
            # styledistance_pan20 = load_dataset(f"explainableAV/models/results/pan20_StyleDistance_predictions_mask_{experiment}.json")

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
                mask_types.append('swap')
                if args.dataset_name == 'amazon':
                    mask_types.append('llm')
            print("LUAR")
            luar_results = create_np_arrays(luar, mask_types, args.baseline)
            print("ModernBERT")
            modernbert_results = create_np_arrays(modernbert, mask_types, args.baseline)
            print("StyleDistance")
            styledistance_results = create_np_arrays(styledistance, mask_types, args.baseline)

            print("LUAR")
            statistical_test(luar, mask_types)
            print("ModernBERT")
            statistical_test(modernbert, mask_types)
            print("StyleDistance")
            statistical_test(styledistance, mask_types)

            plot_all_models(luar_results, modernbert_results, styledistance_results, args.experiment, args.dataset_name, args.baseline)
