import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import argparse
from matplotlib.colors import LinearSegmentedColormap
from explainableAV.utils.utils import load_dataset
from explainableAV.utils.plot_utils import data_path

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
    plt.savefig('explainableAV/results/attention/top_k.pdf')
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
    plt.savefig('explainableAV/results/attention/top_ratio.pdf')
    plt.show()


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--luar_results_path', default=None, help="Set results path if results named differently otherwise assume default save name")
    parser.add_argument('--modernbert_results_path', default=None, help="Set results path if results named differently otherwise assume default save name")
    parser.add_argument('--styledistance_results_path', default=None, help="Set results path if results named differently otherwise assume default save name")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = argument_parser()
    data_LUAR = load_dataset(data_path(f"explainableAV/results/attention/attention_top_LUAR_rollout.json", args.luar_results_path))
    data_ModernBERT = load_dataset(data_path(f"explainableAV/results/attention/attention_top_ModernBERT_value_zeroing.json", args.modernbert_results_path))
    data_StyleDistance = load_dataset(data_path(f"explainableAV/results/attention/attention_top_StyleDistance_raw.json", args.styledistance_results_path))
    top_k_words_plot(data_LUAR, data_ModernBERT, data_StyleDistance)
    top_k_ratio_plot(data_LUAR, data_ModernBERT, data_StyleDistance)
