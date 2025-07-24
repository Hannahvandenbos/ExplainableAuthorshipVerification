#################################################################################################################
# Metrics are adapted from Valla repository: https://github.com/JacobTyo/Valla/blob/main/valla/dsets/Amazon.py
#################################################################################################################

from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from explainableAV.utils.utils import load_dataset, load_multiple_datasets, create_dataset
from explainableAV.utils.perturb_utils import mask_first_text
from sentence_transformers.util import cos_sim
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from statsmodels.stats.contingency_tables import mcnemar
from collections import defaultdict
import scipy.stats as stats
import torch
import argparse
import time
import sys
import numpy as np
import os
import torch.nn.functional as F

def evaluate_model(data, model, batch_size=32, modern_bert=False):
    """
    Inference of AV
    Inputs:
        data: text pairs
        model: SentenceTransformer model
        batch_size: batch size
        modern_bert: whether the model is ModernBERT or not
    Output:
        true y labels
        predicted y scores (similarity)
    """
    y_true = [int(line['Label']) for line in data]

    if modern_bert:
        pairs = [line['Pair'] for line in data]
        flat_pairs = [item for pair in pairs for item in pair]
        all_embeddings = model.encode(flat_pairs, batch_size=batch_size)
        all_embeddings = all_embeddings.reshape(len(pairs), 2, 768)
        similarities = [F.cosine_similarity(torch.tensor(emb[0]).unsqueeze(0), torch.tensor(emb[1]).unsqueeze(0)).item() for emb in all_embeddings]
    else:
        all_sentences = [line['Pair'][0] for line in data] + [line['Pair'][1] for line in data]
        all_embeddings = model.encode(all_sentences, batch_size=batch_size)

        embeddings_1 = all_embeddings[:len(data)]
        embeddings_2 = all_embeddings[len(data):]

        similarities = [F.cosine_similarity(torch.tensor(e1).unsqueeze(0), torch.tensor(e2).unsqueeze(0)).item() for e1, e2 in zip(embeddings_1, embeddings_2)]

    return y_true, similarities

def binarize(y, threshold=0.5):
    '''
    Binarize predictions to get classifications
    Inputs:
        y: predictions by the model
        threshold: model threshold
    Output:
        Binary array of classifications
    '''
    y = np.array(y)
    y[y >= threshold] = 1
    y[y < threshold] = 0
    return y

def confidence(probability, threshold):
    '''
    Transform embedding similarity to confidence score with softmax function
    Inputs:
        similarity: similarity between embeddings of text pair
        threshold: threshold for model
    Output:
        confidence score
    '''
    return 1 / (1 + np.exp(-10 * (probability - threshold))) 

def get_confidence(y, threshold=0.5):
    '''
    Get confidence scores per class and in total
    Inputs:
        y: predicted y classifications
        threshold: model thresholds
    '''
    y = np.array(y)
    confidences_0 = []
    confidences_1 = []
    confidences = []
    for y_value in y:
        confidences_1.append(confidence(y_value, threshold))
        confidences_0.append(1 - confidence(y_value, threshold))
        confidences.append(confidence(y_value, threshold))
    return confidences_0, confidences_1, confidences


def print_results(y_true, y_pred, threshold):
    '''
    Print and compute prediction results for AV
    Inputs:
        y_true: y labels
        y_pred: predicted y scores
        threshold: model classification threshold
    Output:
        dictionary with accuracy, predictions, and confidences
    '''
    print('Accuracy: ', accuracy_score(y_true, binarize(y_pred, threshold)))
    print(binarize(y_pred, threshold))
    confidences_0, confidences_1, confidences = get_confidence(y_pred, threshold)

    return {
        "accuracy": accuracy_score(y_true, binarize(y_pred, threshold)),
        "predictions": binarize(y_pred, threshold), 
        "confidences": confidences
    }

def dictify(obj):
    '''
    Ensure that d is a dictionary so it can be easily saved
    '''
    if isinstance(obj, defaultdict):
        return {k: dictify(v) for k, v in obj.items()}
    elif isinstance(obj, dict):
        return {k: dictify(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [dictify(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    else:
        return obj

def set_threshold(manual_threshold, default_threshold):
    if manual_threshold != 0:
        return manual_threshold
    return default_threshold

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="explainableAV/Amazon/SS_test.json", help="Path to test dataset")
    parser.add_argument('--model_name', type=str, default="LUAR", help="Model to use, one of: 'LUAR', 'StyleDistance', 'ModernBERT'")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--mask_type', default='original')
    parser.add_argument('--data_split', default='SS')
    parser.add_argument('--dataset_name', default='amazon')
    parser.add_argument('--extra_data_path', type=str, help="Path to masked data that should replace one of the texts in a pair of the data_path data")
    parser.add_argument('--perturb_second', action='store_true', help="Perturbs first text if False, perturbs both texts if True, if extra_data_path provided")
    parser.add_argument('--threshold', type=int, default=0, help="If your thresholds differ from ours, set them manually")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = argument_parser()
    data = load_dataset(args.data_path)

    # check dataset name
    if args.dataset_name not in ["amazon", "pan20"]:
        print("Dataset name not recognised, choose one of: 'amazon', 'pan20'")
        exit()

    if args.extra_data_path is not None:
        masked_data = load_dataset(args.extra_data_path)
        data = mask_first_text(data, masked_data, args.perturb_second)

    print()
    # load models and set threshold value
    print(args.model_name)
    if args.model_name == "LUAR":
        model = SentenceTransformer("gabrielloiseau/LUAR-MUD-sentence-transformers")
        if args.dataset_name == 'amazon':
            threshold = set_threshold(args.threshold, 0.37)
        elif args.dataset_name == 'pan20':
            threshold = set_threshold(args.threshold, 0.57)
    elif args.model_name == "StyleDistance":
        model = SentenceTransformer("StyleDistance/styledistance") 
        if args.dataset_name == 'amazon':
            threshold = set_threshold(args.threshold, 0.80)
        elif args.dataset_name == 'pan20':
            threshold = set_threshold(args.threshold, 0.95)
    elif args.model_name == "ModernBERT":
        model = SentenceTransformer('gabrielloiseau/ModernBERT-base-authorship-verification')
        if args.dataset_name == 'amazon':
            threshold = set_threshold(args.threshold, 0.86)
        elif args.dataset_name == 'pan20':
            threshold = set_threshold(args.threshold, 0.96)
    else:
        print("Model name not recognised, choose one of: 'LUAR', 'StyleDistance', 'ModernBERT'")
        exit()

    if args.model_name == 'ModernBERT':
        y_true, y_pred = evaluate_model(data, model, batch_size=args.batch_size, modern_bert=True)
    else: 
        y_true, y_pred = evaluate_model(data, model, batch_size=args.batch_size)

    results = print_results(y_true, y_pred, threshold)

    # save results
    if args.perturb_second:
        mask = 'both'
    else:
        mask = 'first'
    metrics_file = f"explainableAV/results/predictions/{args.dataset_name}_{args.model_name}_predictions_mask_{mask}.json"
    if os.path.exists(metrics_file):
        metrics = load_dataset(metrics_file)
    else:
        metrics = {}

    metrics = defaultdict(lambda: defaultdict(dict), metrics)
    metrics[args.data_split][args.mask_type] = results

    create_dataset(metrics_file, dictify(metrics))
