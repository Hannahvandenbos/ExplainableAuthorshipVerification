import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import argparse
from matplotlib.colors import LinearSegmentedColormap
from explainableAV.utils.utils import load_dataset
from explainableAV.utils.plot_utils import data_path

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
    plt.savefig(f"explainableAV/results/probing_metrics/probing_accuracy.pdf", dpi=300, bbox_inches='tight')
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
    plt.savefig(f"explainableAV/results/probing_metrics/heatmaps_probing_{model_name}.pdf", dpi=300, bbox_inches='tight')

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
    plt.savefig(f"explainableAV/results/probing_metrics/heatmaps_probing_f1_only.pdf", dpi=300, bbox_inches='tight')

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot_type', default='heatmap', help='Choose from "heatmap", "heatmap_f1", "probing_line_plot"')
    parser.add_argument('--model_name', default='LUAR')
    parser.add_argument('--results_path', default=None, help="Set results path if results named differently otherwise assume default save name for heatmap")
    parser.add_argument('--luar_results_path', default=None, help="Set results path if results named differently otherwise assume default save name")
    parser.add_argument('--modernbert_results_path', default=None, help="Set results path if results named differently otherwise assume default save name")
    parser.add_argument('--styledistance_results_path', default=None, help="Set results path if results named differently otherwise assume default save name")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = argument_parser()

    if args.plot_type == 'heatmap':
        metric_results = load_dataset(data_path(f"explainableAV/results/probing_metrics/probing_metrics_{args.model_name}.json", args.results_path))
        probing_per_layer_heatmap(metric_results, args.model_name)
    else:
        metric_results_LUAR = load_dataset(data_path("explainableAV/results/probing_metrics/probing_metrics_LUAR.json", args.luar_results_path))
        metric_results_ModernBERT = load_dataset(data_path("explainableAV/results/probing_metrics/probing_metrics_ModernBERT.json", args.modernbert_results_path))
        metric_results_StyleDistance = load_dataset(data_path("explainableAV/results/probing_metrics/probing_metrics_StyleDistance.json", args.styledistance_results_path))
        if args.plot_type == 'heatmap_f1':
            probing_f1_only(metric_results_LUAR, metric_results_ModernBERT, metric_results_StyleDistance)
        elif args.plot_type == 'probing_line_plot':
            probing_accuracy(metric_results_LUAR, metric_results_ModernBERT, metric_results_StyleDistance)
