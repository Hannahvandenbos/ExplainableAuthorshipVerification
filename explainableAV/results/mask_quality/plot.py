import matplotlib.pyplot as plt
import numpy as np
import argparse
from matplotlib.colors import LinearSegmentedColormap
from explainableAV.utils.utils import load_dataset

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
    plt.savefig(f"explainableAV/results/mask_quality/similarity_plot_{dataset_name}.pdf", dpi=300, bbox_inches='tight')
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
    plt.savefig(f"explainableAV/results/mask_quality/perplexity_plot_{dataset_name}.pdf", dpi=300, bbox_inches='tight')
    plt.show()

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='amazon', help='Choose from "amazon", "pan20"')
    parser.add_argument('--model_name', default='LUAR')
    parser.add_argument('--results_path', default=None, help="Set results path if results named differently otherwise assume default save name")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = argument_parser()

    if args.results_path is None:
        mask_quality = load_dataset(f"explainableAV/results/mask_quality/mask_quality_results_{args.dataset_name}.json") 
    else:
        mask_quality = load_dataset(args.results_path)
    syntax_mean, syntax_std = extract_metric_stats(mask_quality, 'syntax', args.dataset_name)
    semantic_mean, semantic_std = extract_metric_stats(mask_quality, 'semantic', args.dataset_name)
    perplexity_mean, perplexity_std = extract_metric_stats(mask_quality, 'perplexity', args.dataset_name)
    text_perturbation_plot(syntax_mean, syntax_std, semantic_mean, semantic_std, perplexity_mean, perplexity_std, args.dataset_name)
