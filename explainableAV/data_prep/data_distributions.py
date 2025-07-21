import argparse
from explainableAV.utils.utils import load_dataset
from prettytable import PrettyTable
import sys
from collections import defaultdict
import spacy
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

def number_of_topics(data):
    '''
    Find number of unique topics in the data.
    Input: 
        data: dataset
    Output:
        number of topics
    '''
    topics = []
    for line in data:
        topics += line["Topics"]
    return len(list(set(topics)))

def text_size(data):
    '''
    Find number the average number of words in the data.
    Input: 
        data: dataset
    Output:
        average number of words
    '''
    nlp = spacy.load('en_core_web_sm')
    words = []
    for line in data:
        doc = len(nlp(line["Pair"][0]))
        words.append(doc)
        doc2 = len(nlp(line["Pair"][1]))
        words.append(doc2)
    return sum(words) / len(words)

def thousands_formatter(x, pos):
    '''
    Clip results per thousand
    Input:
        x: x value to clip
    Output:
        clipped results
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
    if dataset == 'Amazon':
        color = custom_cmap(0.05)
    elif dataset == 'PAN20':
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

    if dataset == "PAN20" and split=='test':
        ax.set_yticks(range(0, max(y) + 10, 10))
    elif dataset == 'Amazon' and split=='test':
        ax.set_yticks(range(0, max(y) + 5000, 5000))
    elif dataset == "PAN20" and split=='train':
        ax.set_yticks(range(0, max(y) + 100, 100))
    elif dataset == 'Amazon' and split=='train':
        ax.set_yticks(range(0, max(y) + 50000, 50000))

    plt.bar(x, y, color=color, alpha=0.6)
    if dataset == "PAN20" and split=='test':
        num_ticks = 12
        x = list(x)
        tick_positions = np.linspace(0, len(x) - 1, num=num_ticks, dtype=int)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([x[i] for i in tick_positions], rotation=90, fontsize=22)
        ax.set_xlabel("Topics", fontsize=26, labelpad=100)
    elif dataset == 'Amazon' and split=='test':
        plt.xticks(np.arange(len(x)), x, rotation=90, fontsize=22)
        ax.set_xlabel("Topics", fontsize=26, labelpad=13)
    elif dataset == "PAN20" and split=='train':
        num_ticks = 12
        x = list(x)
        tick_positions = list(np.linspace(1, len(x) - 1, num=num_ticks, dtype=int))
        tick_positions = np.array(tick_positions[:-1] + [tick_positions[-1] -1])
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([x[i] for i in tick_positions], rotation=90, fontsize=22)
        ax.set_xlabel("Topics", fontsize=26, labelpad=10)
    elif dataset == "Amazon" and split=='train':
        plt.xticks(np.arange(len(x)), x, rotation=90, fontsize=22)
        ax.set_xlabel("Topics", fontsize=26, labelpad=13)

    plt.ylabel("Number of Documents", fontsize=26)
    plt.tick_params(axis='y', labelsize=22)
    plt.title(f"Topic Distribution ({dataset})", fontsize=28)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.5)
    plt.savefig(f'explainableAV/data_prep//Topic_distribution_{dataset}_{split}.pdf')

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='Amazon', help="Name of the dataset: 'Amazon' or 'PAN20'")  
    parser.add_argument('--statistic', type=str, default='pairs', help="Which statistics to print: 'pairs' or 'splits' or 'topic_distribution'")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = argument_parser()

    # check if data name is valid
    if args.data_name not in ['Amazon', 'PAN20']:
        sys.exit("Dataname not found, choose 'Amazon' or 'PAN20'")

    if args.statistic not in ['pairs', 'splits', 'topic_distribution']:
        sys.exit("Statistic not found, choose 'pairs' or 'splits' or 'topic_distribution'")

    if args.statistic == 'pairs':
        SS = len(load_dataset(f"explainableAV/{args.data_name}/SS.json"))
        SD = len(load_dataset(f"explainableAV/{args.data_name}/SD.json"))
        DS = len(load_dataset(f"explainableAV/{args.data_name}/DS.json"))
        DD = len(load_dataset(f"explainableAV/{args.data_name}/DD.json"))
        pairs = SS + SD + DS + DD

        # print statistics
        table = PrettyTable(["Dataset", "#Pairs", "#SS", "#SD", "#DS", "#DD"])
        table.add_row([args.data_name, pairs, SS, SD, DS, DD])
        print(table)
    elif args.statistic == 'splits':
        if args.data_name == 'Amazon':
            split_size = 15000
        else:
            split_size = 2500

        test = load_dataset(f"explainableAV/{args.data_name}/test_set_{split_size}x4.json")
        validation = load_dataset(f"explainableAV/{args.data_name}/val_set_{split_size}x4.json")
        train = load_dataset(f"explainableAV/{args.data_name}/train_set_{split_size}x4.json")
        total = test + validation + train
        topics = number_of_topics(total)
        text_size = text_size(test)

        # print statistics
        table = PrettyTable(["Dataset", "#Test", "#Validation", "#Train", "#Topics", "Avg. Text Size (Test)"])
        table.add_row([args.data_name, len(test), len(validation), len(train), topics, text_size])
        print(table)
    else:
        if args.data_name == 'Amazon':
            split_size = 15000
        else:
            split_size = 2500

        test = load_dataset(f"explainableAV/{args.data_name}/test_set_{split_size}x4.json")
        train = load_dataset(f"explainableAV/{args.data_name}/train_set_{split_size}x4.json")
        topic_distribution(test, args.data_name, split='test')
        topic_distribution(train, args.data_name, split='train')
