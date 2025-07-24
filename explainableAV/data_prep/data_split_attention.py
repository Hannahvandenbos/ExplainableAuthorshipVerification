import argparse
from explainableAV.utils.utils import load_dataset, create_dataset
import numpy as np

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', default='amazon', help='Choose from "amazon", "pan20"')
    parser.add_argument('--experiment', default='both', help='Choose from "both", "first"')
    parser.add_argument('--mask_type', default='asterisk', help='Choose from "asterisk", "pos tag", "one word", "swap"')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = argument_parser()

    pred_LUAR = load_dataset(f'explainableAV/results/predictions/{args.data_name}_LUAR_predictions_mask_{args.experiment}.json')
    pred_ModernBERT = load_dataset(f'explainableAV/results/predictions/{args.data_name}_ModernBERT_predictions_mask_{args.experiment}.json')
    pred_StyleDistance = load_dataset(f'explainableAV/results/predictions/{args.data_name}_StyleDistance_predictions_mask_{args.experiment}.json')
  
    dataset_most_influence = []
    dataset_least_influence = []
    dataset_most_influence_asterisk = []
    dataset_least_influence_asterisk = []
    pair_types = ['SS', 'SD', 'DS', 'DD']
    for pair_type in pair_types:
        LUAR_baseline_confidence = pred_LUAR[pair_type][f"{args.mask_type}_baseline"]['confidences']
        LUAR_confidence = pred_LUAR[pair_type][args.mask_type]['confidences']
        ModernBERT_baseline_confidence = pred_ModernBERT[pair_type][f"{args.mask_type}_baseline"]['confidences']
        ModernBERT_confidence = pred_ModernBERT[pair_type][args.mask_type]['confidences']
        StyleDistance_baseline_confidence = pred_StyleDistance[pair_type][f"{args.mask_type}_baseline"]['confidences']
        StyleDistance_confidence = pred_StyleDistance[pair_type][args.mask_type]['confidences']
        luar_diffs = [abs(a - b) for a, b in zip(LUAR_baseline_confidence, LUAR_confidence)]
        modernbert_diffs = [abs(a - b) for a, b in zip(ModernBERT_baseline_confidence, ModernBERT_confidence)]
        styledistance_diffs = [abs(a - b) for a, b in zip(StyleDistance_baseline_confidence, StyleDistance_confidence)]
        averages = []
        for luar_diff, modernbert_diff, styledistance_diff in zip(luar_diffs, modernbert_diffs, styledistance_diffs):
            averages.append(np.mean([luar_diff, modernbert_diff, styledistance_diff]))
        averages = np.array(averages)
        idxs_ordered = np.argsort(averages)
        idxs_most = idxs_ordered[-100:]
        idxs_least = idxs_ordered[:100]
        if args.data_name == 'amazon':
            data_name = 'Amazon'
        else:
            data_name = 'PAN20'
        dataset = load_dataset(f"explainableAV/{data_name}/{pair_type}_test.json")
        dataset_most_influence.extend([dataset[i] for i in idxs_most])
        dataset_least_influence.extend([dataset[i] for i in idxs_least])

        dataset_asterisk = load_dataset(f"explainableAV/change_topic/{data_name}/amazon_lda_{pair_type}_asterisk_False_False.json")
        dataset_most_influence_asterisk.extend([dataset_asterisk[i] for i in idxs_most])
        dataset_least_influence_asterisk.extend([dataset_asterisk[i] for i in idxs_least])
    
    create_dataset(f'explainableAV/{data_name}/attention_most_influence.json', dataset_most_influence)
    create_dataset(f'explainableAV/{data_name}/attention_least_influence.json', dataset_least_influence)
    create_dataset(f'explainableAV/{data_name}/attention_most_influence_asterisk.json', dataset_most_influence_asterisk)
    create_dataset(f'explainableAV/{data_name}/attention_least_influence_asterisk.json', dataset_least_influence_asterisk)
