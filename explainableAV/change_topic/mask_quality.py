#####################################################################################
# Perplexity computed with: https://github.com/asahi417/lmppl
#####################################################################################

import argparse
import os
import json
import numpy as np
from explainableAV.utils.utils import load_dataset, load_multiple_datasets, create_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import math
from collections import Counter, defaultdict
from nltk.util import ngrams
from nltk.probability import FreqDist

def compute_perplexity(data, masked_data, mask_one_text):
    '''
    Compute perplexity differences between original and perturbed text
    Inputs:
        data: original texts
        masked_data: perturbed texts
        masked_one_text: whether only the first text is masked
    Output:
        list with perplexity differences
    '''
    perplexities = []
    skips = 0
    for line, masked_line in zip(data, masked_data):
        perplexity1 = perplexity(line["Pair"][0])
        perplexity2 = perplexity(masked_line["Pair"][0])
        if perplexity1 == 'No good' or perplexity2 == 'No good':
            skips += 1
            continue
        perplexities.append(perplexity1 - perplexity2)
        if not mask_one_text:
            perplexity1 = perplexity(line["Pair"][0])
            perplexity2 = perplexity(masked_line["Pair"][0])
            if perplexity1 == 'No good' or perplexity2 == 'No good':
                skips += 1
                continue
            perplexities.append(perplexity1 - perplexity2)
    print(skips) # number of times no trigrams were found
    return perplexities


def perplexity(text, n=3):
    '''
    Compute perplexity of a text
    Inputs:
        text: text to compute perplexity of
        n: n-value for n-gram
    Output:
        perplexity of text
    '''
    words = text.split()
    n_grams = list(ngrams(words, n))
    freq_dist = FreqDist(n_grams)
    total_ngrams = sum(freq_dist.values())

    entropy = 0
    for ngram in n_grams:
        prob = freq_dist[ngram] / total_ngrams
        entropy += -math.log2(prob)
    if len(n_grams) == 0: # if no trigrams, then no perplexity
        return 'No good'
    return 2 ** (entropy / len(n_grams))


def compute_semantic_similarity(data, masked_data, mask_one_text):
    '''
    Compute semantic similarity between original and masked text with SBERT
    Inputs:
        data: original texts
        masked_data: perturbed texts
        masked_one_text: whether only the first text is masked
    Output:
        list of semantic similarities
    '''
    model = SentenceTransformer('all-MiniLM-L6-v2')
    similarities = []

    for line, masked_line in zip(data, masked_data):
        similarities.append(sbert_similarity(line["Pair"][0], masked_line["Pair"][0], model))
        if not mask_one_text:
            similarities.append(sbert_similarity(line["Pair"][1], masked_line["Pair"][1], model))
    return similarities


def sbert_similarity(text1, text2, model):
    '''
    Compute similarity between two texts based on SBERT
    Inputs:
        text1: first text
        text2: second text
        model: model to encode texts with (SBERT)
    Output:
        similarity between texts        
    '''
    text1_embedding = model.encode(text1, convert_to_tensor=True)
    text2_embedding = model.encode(text2, convert_to_tensor=True)
    
    similarity = util.cos_sim(text1_embedding, text2_embedding)
    return similarity.item()


def cosine_similarity_tfidf(text1, text2):
    '''
    Compute similarity between texts embedded with TF-IDF
    Inputs:
        text1: first text
        text2: second text
    Output:
        similarity between texts based on TF-IDF
    '''
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(vectors[0], vectors[1])[0][0]

def compute_syntax_similarity(data, masked_data, mask_one_text):
    '''
    Compute syntactic similarity between original and masked text with SBERT
    Inputs:
        data: original texts
        masked_data: perturbed texts
        masked_one_text: whether only the first text is masked
    Output:
        list of syntactic similarities
    '''
    similarities = []
    for line, masked_line in zip(data, masked_data):
        similarities.append(cosine_similarity_tfidf(line["Pair"][0], masked_line["Pair"][0]))
        if not mask_one_text:
            similarities.append(cosine_similarity_tfidf(line["Pair"][1], masked_line["Pair"][1]))
    return similarities

def similarity_metrics(similarities):
    '''
    Store metric similarities
    Input:
        similarities: similarity values over all text pairs
    Output:
        dictionary with various statistics of the similarities
    '''
    return {
        "mean": np.mean(similarities),
        "median": np.median(similarities),
        "min": min(similarities),
        "max": max(similarities),
        "std_dev": np.std(similarities),
        "variance": np.var(similarities),
        "q1": np.percentile(similarities, 25),
        "q3": np.percentile(similarities, 75),
        "iqr": np.percentile(similarities, 75) - np.percentile(similarities, 25)
    }

def dictify(d):
    '''
    Ensure that d is a dictionary so it can be easily saved
    '''
    if isinstance(d, defaultdict):
        d = {k: dictify(v) for k, v in d.items()}
    return d

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path_SS', type=str, required=True, help="Path to test dataset")
    parser.add_argument('--data_path_SD', type=str, required=True, help="Path to test dataset")
    parser.add_argument('--data_path_DS', type=str, required=True, help="Path to test dataset")
    parser.add_argument('--data_path_DD', type=str, required=True, help="Path to test dataset")
    parser.add_argument('--masked_data_path_SS', type=str, required=True, help="Path to masked test dataset")
    parser.add_argument('--masked_data_path_SD', type=str, required=True, help="Path to masked test dataset")
    parser.add_argument('--masked_data_path_DS', type=str, required=True, help="Path to masked test dataset")
    parser.add_argument('--masked_data_path_DD', type=str, required=True, help="Path to masked test dataset")
    parser.add_argument('--dataset_name', type=str, default='amazon')
    parser.add_argument('--mask_type', type=str, default='asterisk', help="Path to masked test dataset")
    parser.add_argument('--mask_one_text', action='store_true', help='If True, assume only first text of pair is masked and only compute similarity between first texts')
    return parser.parse_args()

if __name__ == '__main__':
    args = argument_parser()

    data = load_multiple_datasets([args.data_path_SS, args.data_path_SD, args.data_path_DS, args.data_path_DD])
    masked_data = load_multiple_datasets([args.masked_data_path_SS, args.masked_data_path_SD, args.masked_data_path_DS, args.masked_data_path_DD])

    metrics_file = f"explainableAV/change_topic/mask_quality_results_{args.dataset_name}.json"
    if os.path.exists(metrics_file):
        metrics = load_dataset(metrics_file)
    else:
        metrics = {}

    print(args.masked_data_path_SS)
    metrics = defaultdict(lambda: defaultdict(dict), metrics)

    # compute and save metrics
    syntax_similarities = compute_syntax_similarity(data, masked_data, args.mask_one_text)
    metrics[args.mask_type]["syntax"] = similarity_metrics(syntax_similarities)
    
    semantic_similarities = compute_semantic_similarity(data, masked_data, args.mask_one_text)
    metrics[args.mask_type]["semantic"] = similarity_metrics(semantic_similarities)

    perplexity_scores = compute_perplexity(data, masked_data, args.mask_one_text)
    metrics[args.mask_type]["perplexity"] = similarity_metrics(perplexity_scores)

    create_dataset(metrics_file, dictify(metrics))
