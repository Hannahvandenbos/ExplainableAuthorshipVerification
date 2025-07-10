import matplotlib.pyplot as plt
import argparse
from matplotlib.colors import LinearSegmentedColormap
from explainableAV.utils.utils import load_dataset
from explainableAV.utils.plot_utils import data_path
import math

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
    plt.savefig(f"explainableAV/results/probing_losses/probing_learning_curve_{model_name}.pdf", dpi=300, bbox_inches='tight')
    plt.show()

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='LUAR')
    parser.add_argument('--results_path', default=None, help="Set results path if results named differently otherwise assume default save name")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = argument_parser()
    metric_results = load_dataset(data_path(f"explainableAV/results/probing_losses/probing_losses_{args.model_name}.json", args.results_path))
    probing_learning_curve(metric_results, args.model_name)
