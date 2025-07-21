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

def preprocess_text(text, nlp):
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
        if unwanted_pattern.fullmatch(token.text) and token.ent_type_ == '':
            continue
        if not token.is_stop and not token.is_punct and token.tag_ in ['NN', 'NNP', 'NNPS', 'NNS']:#, 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS']:
            tokens.append(token.lemma_.lower())
            if token.ent_type_ != '': # look specifically for NER
                pos_tags.append(token.ent_type_)
            else:
                pos_tags.append(token.tag_)
    return tokens, pos_tags, [token.lemma_.lower() for token in doc]

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

def correct_classifications(document_topics, text_to_topic, topic_texts):
    '''
    Compute and print number of correctly classified documents per topic
    Inputs:
        document_topics: documents per topic
        text_to_topic: dictionary that maps text to actual topic
        topic_texts: topic name for a text by LDA
    '''
    correct_count = 0
    incorrect_count = 0
    correct_assignments = defaultdict(int) 
    incorrect_assignments = defaultdict(int)

    for i, (doc, topic_idx) in enumerate(document_topics):
        actual_topic_name = text_to_topic[doc] 
        assigned_topic_name = list(topic_texts.keys())[topic_idx]

        if actual_topic_name == assigned_topic_name:
            correct_count += 1
            correct_assignments[assigned_topic_name] += 1
        else:
            incorrect_count += 1
            incorrect_assignments[assigned_topic_name] += 1

    print("\nTopic Assignment Results:")
    print(f"Correctly assigned documents: {correct_count}")
    print(f"Incorrectly assigned documents: {incorrect_count}")
    
    print("\nCorrect Assignments per Topic:")
    for topic_name, count in correct_assignments.items():
        print(f"Topic: {topic_name} - Correct assignments: {count}")

    print("\nIncorrect Assignments per Topic:")
    for topic_name, count in incorrect_assignments.items():
        print(f"Topic: {topic_name} - Incorrect assignments: {count}")   


def guided_lda_main(topic_texts, nlp, seed):
    '''
    Perform Guided LDA
    Inputs:
        topic_texts: dictionary with texts per topic
        nlp: Spacy's NLP
        seed: seed
    Outputs:
        LDA model
        topic assignments by model
        processed texts
        original mapping of texts to topic
        topic_texts
        vocab from LDA model
        doc_tokens: tokens used for the LDA 
        texts from the doc tokens 
        mapping between topic index and topic name
    '''
 
    word_to_pos = {}
    processed_texts = []
    seed_topic_list = []
    text_to_topic = {}

    doc_tokens = []
    doc_texts = defaultdict(list)
    for topic, texts in topic_texts.items(): # create input tokens
        token_list = []
        for text in texts:
            tokens, pos_tags, doc = preprocess_text(text, nlp)
            doc_texts[topic].append(doc)
            doc_tokens.append(tokens)
            processed_texts.append(" ".join(tokens))
            text_to_topic[" ".join(tokens)] = topic

            for token, pos in zip(tokens, pos_tags):  
                word_to_pos[token] = pos  # store POS tags
            token_list += tokens  
        topic_tokens = [token.lemma_.lower() for token in nlp(topic) if not token.is_stop and not token.is_punct]
        seed_topic_list.append(topic_tokens)
    print("Seed topic list: ", seed_topic_list)

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(processed_texts)  # create document-term matrix
    vocab = np.sort(vectorizer.get_feature_names_out())
    word2id = {word: idx for idx, word in enumerate(vocab)}

    np.random.seed(seed)
    random.seed(seed)

    model = guidedlda.GuidedLDA(n_topics=len(topic_texts), n_iter=100, random_state=seed, refresh=20)

    seed_topics = {}
    topic_index_to_name = {}
    for t_id, topic_words in enumerate(seed_topic_list):
        for word in topic_words:
            if word in word2id:
                seed_topics[word2id[word]] = t_id
        topic_index_to_name[t_id] = list(topic_texts.keys())[t_id]

    model.fit(X, seed_topics=seed_topics, seed_confidence=0.75)
    
    topic_assignments = model.transform(X) 

    return model, topic_assignments, processed_texts, text_to_topic, topic_texts, vocab, doc_tokens, doc_texts, topic_index_to_name

def get_topic_words(num_words, model, vocab, topic_index_to_name, save_name):
    '''
    Extract topic words from LDA
    Inputs:
        num_words: number of topic words per topic
        model: LDA model
        vocab: LDA vocabulary
        topic_index_to_name: mapping from topic index to topic name
        save_name: path/name to save the topic words
    '''
    topic_word = model.topic_word_
    topics_topic_word = []
    topic_related = {}

    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(num_words+1):-1]
        topic_name = topic_index_to_name.get(i, f"Topic {i}")
        topic_related[topic_name] = list(topic_words)
    create_dataset(save_name, topic_related)

def get_topic_words_all_nouns(topic_texts, nlp, save_name):
    '''
    Get all nouns as topic words
    Input:
        topic_texts: dictionary of texts ordered per topic
        nlp: Spacy's NLP
        save_name: path/name to save the topic words
    '''
    topic_related = {}
    for topic, texts in topic_texts.items():
        topic_words = []
        for text in texts:
            tokens, pos_tags, doc = preprocess_text(text, nlp)
            for token, pos_tag in zip(tokens, pos_tags):
                if pos_tag in ['NN', 'NNS', 'NNP', 'NNPS']:
                    topic_words.append(token)
        topic_related[topic] = list(set(topic_words))
    create_dataset(save_name, topic_related)

def filter_common_words(save_name, words_all_topics, new_save_name):
    '''
    Filter topic words based on common words
    Inputs:
        save_name: path/name where topic words are
        words_all_topics: topic words that are in all topics
        new_save_name: path/name to save filtered topic words
    '''
    topic_related = load_dataset(save_name)

    with open('explainableAV/extract_topic/500_most_common_English_words.txt', 'r') as file:
        common_words = [line.strip() for line in file if line.strip()]
    
    for key in topic_related.keys():
        topic_words = topic_related[key]
        topic_related[key] = [word for word in topic_words if word not in common_words and word not in words_all_topics]
    create_dataset(new_save_name, topic_related)

def count_words_in_x_topics(topic_dict, x):
    '''
    Get number of words that occur in x topics
    Inputs:
        topic_dict: dictionary with topic words per topic
        x: number of topics
    Outputs:
        number of words that occur in x topics
        words that occur in x topics
    '''
    word_topic_count = defaultdict(int)
    
    for words in topic_dict.values():
        unique_words_in_topic = set(word_topic_count)
        for word in unique_words_in_topic:
            word_topic_count[word] += 1

    words_in_x_topics_list = [word for word, count in word_topic_count.items() if count == x] # words that appear in exactly x topics
    
    return len(words_in_x_topics_list), words_in_x_topics_list

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help="Path to test dataset")
    parser.add_argument('--data_name', type=str, default='amazon')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--topic_words', type=int, default=8400)
    parser.add_argument('--save_name', type=str, default="explainableAV/extract_topic/amazon_topic_related_8400.json")
    parser.add_argument('--evaluate', action='store_true', help='If True, will evaluate different numbers of topic words')
    parser.add_argument('--save', type=str, help="path to save the topic related dictionary")
    return parser.parse_args()

if __name__ == '__main__':
    args = argument_parser()  
    data = load_dataset(args.data_path)
    nlp = spacy.load('en_core_web_sm')

    topic_texts = get_individual_texts_per_topic(data)
    if args.data_name == 'pan20' and not args.evaluate:
        get_topic_words_all_nouns(topic_texts, nlp, args.save_name)
        topic_dict = load_dataset(args.save_name)
        length, words = count_words_in_x_topics(topic_dict, len(list(topic_dict.keys())))
        filter_common_words(args.save_name, words, "explainableAV/extract_topic/pan20_topic_related_all_nouns_filtered.json")
    
    elif args.data_name == 'pan20' and args.evaluate:
        model, topic_assignments, processed_texts, text_to_topic, topic_texts, vocab, doc_tokens, doc_texts, topic_index_to_name = guided_lda_main(topic_texts, nlp, args.seed)
        for n_top_words in range(50, 1001, 50):
            save_name = f'explainableAV/extract_topic/pan20_topic_related_{n_top_words}.json'
            get_topic_words(n_top_words, model, vocab, topic_index_to_name, save_name)

    elif args.data_name == 'amazon' and args.evaluate:
        model, topic_assignments, processed_texts, text_to_topic, topic_texts, vocab, doc_tokens, doc_texts, topic_index_to_name = guided_lda_main(topic_texts, nlp, args.seed)
        for n_top_words in range(700, 14001, 700):
            save_name = f'explainableAV/extract_topic/amazon_topic_related_{n_top_words}.json'
            get_topic_words(n_top_words, model, vocab, topic_index_to_name, save_name)
    else:
        model, topic_assignments, processed_texts, text_to_topic, topic_texts, vocab, doc_tokens, doc_texts, topic_index_to_name = guided_lda_main(topic_texts, nlp, args.seed)
        get_topic_words(args.topic_words, model, vocab, topic_index_to_name, args.save_name)
        topic_dict = load_dataset(args.save_name)
        length, words = count_words_in_x_topics(topic_dict, len(list(topic_dict.keys())))
        filter_common_words(args.save_name, words, f"explainableAV/extract_topic/amazon_topic_related_{args.topic_words}_filtered.json")
    
