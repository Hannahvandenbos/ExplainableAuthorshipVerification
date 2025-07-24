import argparse
import spacy
import re
import random
import gensim
import guidedlda
import numpy as np
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from six.moves import cPickle as pickle
from gensim import corpora
from collections import defaultdict, Counter
from explainableAV.utils.utils import load_dataset, create_dataset
from explainableAV.utils.perturb_utils import get_individual_texts_per_topic
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ProcessPoolExecutor

def preprocess_text_for_recombination(text, nlp):
    '''
    Lemmatize one text
    Inputs:
        text: raw input text
        nlp: Spacy's NLP
    Outputs:
        list of original tokens and punctuation, whitespace and whether it is capitalized
        tokenized input with Spacy
        POS tags corresponding with tokens
        dictionary that maps original tokens with lemmatized tokens
    '''
    doc = nlp(text)
    unwanted_pattern = re.compile(r'[^ ]*\"[^ ]*')
    tokens = []
    pos_tags = []

    for token in doc:
        tokens.append(token.lemma_.lower())
        if token.ent_type_ != '': # look specifically for NER
            pos_tags.append(token.ent_type_)
        else:
            pos_tags.append(token.tag_)
    return tokens, pos_tags

def preprocess_word_list_batch(words, nlp):
    '''
    Helper function to process topic words in batches
    Inputs:
        words: words corresponding to a topic
        nlp: Spacy's NLP
    Outputs:
        tokenized version of words
        pos tags corresponding with tokens
    '''
    text = ' '.join(words)
    doc = nlp(text)
    
    tokens = []
    pos_tags = []

    for token in doc:
        tokens.append(token.lemma_.lower())
        if token.ent_type_ != '':
            pos_tags.append(token.ent_type_)
        else:
            pos_tags.append(token.tag_)
    
    return tokens, pos_tags


def preprocess_topic_words_batch(topic, topic_words, nlp):
    '''
    Preprocess topic words and keep only the nouns
    Inputs:
        topic: current topic
        topic_words: words corresponding with the topic
        nlp: Spacy's NLP
    Output:
        processed topic words
    '''
    tokens, pos_tags = preprocess_word_list_batch(topic_words, nlp)
    new_tokens = [token for token, pos_tag in zip(tokens, pos_tags) if pos_tag in ['NN', 'NNS', 'NNP', 'NNPS']]
    return new_tokens


def inter_topic_distance(topic_datasets):
    '''
    Compute inter topic distance between all topics
    Input:
        topic_datasets: datasets with topic words per topic
    Output:
        average inter topic distance
    '''
    vectorizer = TfidfVectorizer(stop_words='english')

    documents = [' '.join(topic) for topic in topic_datasets.values()]
    new_documents = []
    for document in documents:
        if len(document) != 0:
            new_documents.append(document)

    filtered_documents = [doc for doc in new_documents if doc.strip()]
    if len(filtered_documents) < 2:
        return float('nan')  # when not enough topics to compare

    tfidf_matrix = vectorizer.fit_transform(filtered_documents)
    topic_vectors = tfidf_matrix.toarray()
    cosine_sim_matrix = cosine_similarity(topic_vectors)
    cosine_dist_matrix = 1 - cosine_sim_matrix

    upper_triangle_indices = np.triu_indices(len(cosine_dist_matrix), k=1)
    upper_triangle_distances = cosine_dist_matrix[upper_triangle_indices]
    average_distance = upper_triangle_distances.mean()

    return average_distance

def mask(tokens, pos_tags, topic, topic_datasets, percentages):
    '''
    Compute the percentage of altered text
    Inputs:
        tokens: input tokens
        pos_tags: pos tags corresponding with the input tokens
        topic: current topic
        topic_datasets: datasets with topic words per topic
        percentages: dictionary to store results
    Output:
        percentages dictionary 
    '''
    positions = {'NN', 'NNP', 'NNPS', 'NNS'}
    
    topic_word_sets = {
        name: set(topic_related[topic]) 
        for name, topic_related in topic_datasets.items() 
        if topic in topic_related
    }

    masked_tokens = defaultdict(int)
    no_mask_tokens = defaultdict(int)

    for token, pos_tag in zip(tokens, pos_tags):
        for name, word_set in topic_word_sets.items():
            if pos_tag in positions and token in word_set:
                masked_tokens[name] += 1
            else:
                no_mask_tokens[name] += 1


    for name in topic_word_sets.keys():
        total = masked_tokens[name] + no_mask_tokens[name]
        if total > 0:
            percentages[name].append(masked_tokens[name] / total)
        else:
            percentages[name].append(0.0)  # or np.nan

    return percentages

def evaluate_masks(data, nlp, data_name, evaluate):
    '''
    Compute percentage of altered text for different topic word set sizes
    Inputs:
        data: input texts
        nlp: Spacy's NLP
        data_name: 'amazon' or 'pan20'
    Results are printed
    '''
    topic_datasets = {}
    if data_name == 'amazon': # load topic related dictionaries
        if not evaluate:
            topic_datasets[f"topic_related_8400"] = load_dataset(f"explainableAV/extract_topic/amazon_topic_related_8400_filtered.json") # for final dataset evaluation
        else:
            for n_top_words in range(700, 14001, 700):
                topic_datasets[f"topic_related_{n_top_words}"] = load_dataset(f"explainableAV/extract_topic/amazon_topic_related_{n_top_words}.json")
    else:
        if not evaluate:
            topic_datasets[f"topic_related_all_nouns"] = load_dataset(f"explainableAV/extract_topic/pan20_topic_related_all_nouns_filtered.json") # for final dataset evaluation
        else:
            for n_top_words in range(5000, 60001, 5000):
                topic_datasets[f"topic_related_{n_top_words}"] = load_dataset(f"explainableAV/extract_topic/pan20_topic_related_{n_top_words}.json")
    
    percentages = defaultdict(list)
    for line in data:
        tokens1, pos_tags1 = preprocess_text_for_recombination(line["Pair"][0], nlp)
        tokens2, pos_tags2 = preprocess_text_for_recombination(line["Pair"][1], nlp)
        percentages = mask(tokens1, pos_tags1, line["Topics"][0], topic_datasets, percentages)
        percentages = mask(tokens2, pos_tags2, line["Topics"][1], topic_datasets, percentages)

    if evaluate:
        create_dataset(f"explainableAV/extract_topic/{data_name}_evaluate_mask_percentage.json", dict(percentages))
    else:
        for name in percentages.keys():
            print(name, np.mean(percentages[name]))


def evaluate_inter_topic_distance(data, nlp, data_name, evaluate):
    '''
    Compute inter-topic distance for different topic word set sizes
    Inputs:
        data: input texts
        nlp: Spacy's NLP
        data_name: 'amazon' or 'pan20'
        evaluate: whether to evaluate multiple numbers of topic words or just one
    Results are printed
    '''
    topic_datasets = {}
    # Load topic-related dictionaries
    if data_name == 'amazon':
        if not evaluate:
            topic_datasets[f"topic_related_8400"] = load_dataset(f"explainableAV/extract_topic/amazon_topic_related_8400_filtered.json") # for final dataset evaluation
        else:
            for n_top_words in range(700, 14001, 700):
                topic_datasets[f"topic_related_{n_top_words}"] = load_dataset(f"explainableAV/extract_topic/amazon_topic_related_{n_top_words}.json")
    else:
        if not evaluate:
            topic_datasets[f"topic_related_all_nouns"] = load_dataset(f"explainableAV/extract_topic/pan20_topic_related_all_nouns_filtered.json") # for final dataset evaluation
        else:
            for n_top_words in range(5000, 60001, 5000):
                topic_datasets[f"topic_related_{n_top_words}"] = load_dataset(f"explainableAV/extract_topic/pan20_topic_related_{n_top_words}.json")
    
    results = []
    for topic_dataset in topic_datasets.values():
        for topic, topic_words in topic_dataset.items():
            results.append(preprocess_topic_words_batch(topic, topic_words, nlp))

    idx = 0
    for topic_dataset in topic_datasets.values():
        for topic in topic_dataset.keys():
            topic_dataset[topic] = results[idx]
            idx += 1

    for topic_dataset in topic_datasets.values():
        average_distance = inter_topic_distance(topic_dataset)
        print(f"Average distance for dataset: {average_distance}")

def total_number_of_nouns(data, nlp):
    '''
    Compute total number of nouns in the full dataset
    Inputs:
        data: text pairs
        nlp: Spacy's NLP
    Output:
        number of nouns
        number of non-nouns
    '''
    nouns = []
    non_nouns = []
    for line in data:
        for text in line["Pair"]:
            tokens, pos_tags = preprocess_text_for_recombination(text, nlp)
            for token, pos_tag in zip(tokens, pos_tags):
                if pos_tag in ['NN', 'NNS', 'NNP', 'NNPS']:
                    nouns.append(token)
                else:
                    non_nouns.append(token)
    return nouns, non_nouns


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help="Path to test dataset")
    parser.add_argument('--save', type=str, help="path to save the topic related dictionary")
    parser.add_argument('--data_name', default='amazon')
    parser.add_argument('--evaluate_masks', action='store_true')
    parser.add_argument('--inter_distance', action='store_true')
    parser.add_argument('--evaluate', action='store_true', help='If True, evaluate multiple number of topic words')
    return parser.parse_args()


if __name__ == '__main__':
    args = argument_parser()
    
    data = load_dataset(args.data_path)
    nlp = spacy.load('en_core_web_sm')

    if args.evaluate_masks:
        evaluate_masks(data, nlp, args.data_name, args.evaluate)

    if args.inter_distance:
        evaluate_inter_topic_distance(data, nlp, args.data_name, args.evaluate)
