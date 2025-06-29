import argparse
import copy
import spacy
import re
import random
from collections import defaultdict
from explainableAV.utils.utils import load_dataset, create_dataset

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
    original_data = []
    orig_to_token = defaultdict(list)

    for token in doc:
        tokens.append(token.lemma_.lower())
        if token.ent_type_ != '': # look specifically for NER
            pos_tags.append(token.ent_type_)
        else:
            pos_tags.append(token.tag_)
        original_data.append((token.text, token.is_punct, token.whitespace_, token.is_title))
        orig_to_token[token.text].append(token.lemma_.lower())
    return original_data, tokens, pos_tags, orig_to_token

def reconstruct_text(tokens, original_data, orig_to_token):
    """
    Reconstructs the original text format while maintaining punctuation, spacing, and capitalization
    Inputs:
        tokens: input tokens
        original_data: original text
        orig_to_token: dictionary that maps original words to tokens
    Output:
        reconstruced text with masked topic words
    """
    reconstructed = []
    
    for token, (orig_text, is_punct, is_space, is_title) in zip(tokens, original_data):
        if is_punct or is_space:
            if token not in orig_to_token[orig_text]: 
                reconstructed.append(token.capitalize() if is_title else token)
            else:
                reconstructed.append(orig_text)
            if is_space:
                reconstructed.append(' ')
        else:
            if token not in orig_to_token[orig_text]: 
                reconstructed.append(token.capitalize() if is_title else token)
            else:
                reconstructed.append(orig_text)
    
    return "".join(reconstructed)

def mask_simple(tokens, original_data, pos_tags, positions, topic_words, baseline, mask_type='asterisk'):
    '''
    Replace token with asterisk or POS tag if it is a noun and a topic-related word
    Inputs:
        tokens: input tokens
        original_data: original text
        pos_tags: pos tags of input tokens
        positions: which type of tokens to replace (based on pos tags)
        topic_words: list of topic words that correspond with the topic of the tokens
        baseline: whether to randomly mask or based on topic words
        mask_type: asterisk to replace with * otherwise will replace with POS tag
    Output:
        tokens of text with replaced tokens
    '''
    new_tokens = []
    if baseline:
        counter = sum([1 for token, pos in zip(tokens, pos_tags) if pos in positions and token in topic_words]) # count how many tokens to replace

        # select random indices
        rng = random.Random(0)
        indices = rng.sample(range(len(tokens)), counter)

        if mask_type == 'asterisk':
            new_tokens = ['*' if i in indices else token for i, token in enumerate(tokens)]
        else:
            new_tokens = [pos_tag if i in indices else token for i, (token, pos_tag) in enumerate(zip(tokens, pos_tags))]
    else:
        for token, pos_tag in zip(tokens, pos_tags):
            if pos_tag in positions and token in topic_words: # check if this type of words should be masked
                if mask_type == 'asterisk':
                    new_tokens.append('*')
                else:
                    new_tokens.append(pos_tag)
            else:
                new_tokens.append(token)
    return new_tokens

def mask_one_word(tokens, original_data, pos_tags, positions, topic_words, baseline):
    '''
    Replace token with 'smurf' if it is a noun and a topic-related word
    Inputs:
        tokens: input tokens
        original_data: original text
        pos_tags: pos tags of input tokens
        positions: which type of tokens to replace (based on pos tags)
        topic_words: list of topic words that correspond with the topic of the tokens
        baseline: whether to randomly mask or based on topic words
    Output:
        tokens of text with replaced tokens
    '''
    new_tokens = []
    if baseline:
        smurf_dict_expanded = {
            'NN': 'smurf',
            'NNS': 'smurfs',
            'NNP': 'smurf',
            'NNPS': 'smurfs',
            'VB': 'smurf',
            'VBD': 'smurfed',
            'VBG': 'smurfing',
            'VBN': 'smurfed',
            'VBP': 'smurf',
            'VBZ': 'smurfs',
            'JJ': 'smurfy',
            'JJR': 'smurfier',
            'JJS': 'smurfiest',
            'RB': 'smurfily',
            'RBR': 'smurfier',
            'RBS': 'smurfiest',
            'DT': 'the',
            'IN': 'on',
            'CC': 'and',
            'UH': 'smurf!',
            'PRP': 'smurf',
            'PRP$': "smurf's",
            'MD': 'can',
            'TO': 'to',
            'EX': 'there',
            'FW': 'smurf',
            'POS': "'s",
            'SYM': '*',
            'CD': 'smurfy-two',
            '.': '.',
            '_SP': ' ',
        }
        counter = sum([1 for token, pos in zip(tokens, pos_tags) if pos in positions and token in topic_words]) # count how many tokens to replace

        # select random indices
        rng = random.Random(0)
        indices = rng.sample(range(len(tokens)), counter)

        new_tokens = [
            smurf_dict_expanded[pos_tag] if i in indices and pos_tag in smurf_dict_expanded else token
            for i, (token, pos_tag) in enumerate(zip(tokens, pos_tags))
        ]
    else:
        smurf_dict = {'NN': 'smurf', 'NNP': 'smurf', 'NNPS': 'smurfs', 'NNS': 'smurfs'}
        for token, pos_tag in zip(tokens, pos_tags):
            if pos_tag in positions and token in topic_words: # check if this type of words should be masked
                new_tokens.append(smurf_dict[pos_tag])
            else:
                new_tokens.append(token)
    return new_tokens

def mask_with_words_from_other_topic(tokens, original_data, pos_tags, positions, topic_words, new_topic_words, baseline):
    '''
    Replace tokens with topic words from other topic if it is a noun and a topic-related word
    Inputs:
        tokens: input tokens
        original_data: original text
        pos_tags: pos tags of input tokens
        positions: which type of tokens to replace (based on pos tags)
        topic_words: list of topic words that correspond with the topic of the tokens
        new_topic_words: list of topic words corresponding with the topic that the text should be changed to
        baseline: whether to randomly mask or based on topic words
    Output:
        tokens of text with replaced tokens
    '''
    new_tokens = []
    if baseline:
        for token, pos_tag in zip(tokens, pos_tags):
            if pos_tag in positions and token in topic_words: # check if this type of words should be masked
                new_tokens.append(random.choice(topic_words))
            else:
                new_tokens.append(token)
    else:
        for token, pos_tag in zip(tokens, pos_tags):
            if pos_tag in positions and token in topic_words: # check if this type of words should be masked
                new_tokens.append(random.choice(new_topic_words))
            else:
                new_tokens.append(token)
    return new_tokens


def mask_text(text, topic_words, new_topic_words, nlp, baseline, mask_type):
    '''
    Mask the words found in the topic_related dictionary with a mask
    only words of the type as found in pos_tags will be masked
    Inputs:
        text: input text to mask
        topic_words: list of topic words corresponding with topic of input text
        new_topic_words: list of topic words corresponding with topic that text should be changed to
        nlp: Spacy's NLP
        baseline: whether to perturb randomly or topic-related words
        mask_type: which perturbation technique to apply
    Output:
        perturbed text 
    '''
    original_data, tokens, pos_tags, orig_to_token = preprocess_text_for_recombination(text, nlp)
    positions = ['NN', 'NNP', 'NNPS', 'NNS']
    
    # return reconstruct_text(tokens, original_data)
    if mask_type == 'asterisk' or mask_type == 'pos tag':
        new_tokens = mask_simple(tokens, original_data, pos_tags, positions, topic_words, baseline, mask_type)
    elif mask_type == 'one word':
        new_tokens = mask_one_word(tokens, original_data, pos_tags, positions, topic_words, baseline)
    elif mask_type == 'change topic':
        new_tokens = mask_with_words_from_other_topic(tokens, original_data, pos_tags, positions, topic_words, new_topic_words, baseline)
    else:
        print('Not a valid mask type, choose one of the following: "asterisk", "pos_tag", "change topic"')

    return reconstruct_text(new_tokens, original_data, orig_to_token)

def mask_data(data, topic_related, nlp, baseline, mask_type, mask_one_text=False, different=False):
    '''
    Mask the all data
    Inputs:
        data: input texts to mask
        topic_related: dictionary with all topic words per topic
        nlp: Spacy's NLP
        baseline: whether to perturb randomly or topic-related words
        mask_type: which perturbation technique to apply
        mask_one_text: whether to mask one or both texts
        different: whether to change to the same or a different topic
    Output:
        perturbed texts 
    '''
    masked_data = copy.deepcopy(data)
    for line in masked_data:
        if different: 
            new_topic = random.choice([topic for topic in topic_related.keys() if topic != line["Topics"][0]])
            new_topic_words = topic_related[new_topic]
            line["Pair"][0] = mask_text(line["Pair"][0], topic_related[line["Topics"][0]], new_topic_words, nlp, baseline, mask_type=mask_type)
            line["Topics"][0] = new_topic
        else:
            line["Pair"][0] = mask_text(line["Pair"][0], topic_related[line["Topics"][0]], topic_related[line["Topics"][1]], nlp, baseline, mask_type=mask_type)
        
        if not mask_one_text:
            line["Pair"][1] = mask_text(line["Pair"][1], topic_related[line["Topics"][1]], topic_related[line["Topics"][1]], nlp, baseline, mask_type=mask_type)
    return masked_data

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help="Path to train dataset")
    parser.add_argument('--topic_related_path', type=str, default="explainableAV/extract_topic/amazon_topic_related_8400_filtered.json")
    parser.add_argument('--mask_type', type=str, default='asterisk', help='How to mask the text, choose from "asterisk", "pos tag", "one word", "change topic"')
    parser.add_argument('--mask_one_text', action='store_true', help='If True, only mask the first text from a pair')
    parser.add_argument('--save', action='store_true', help='If True, will save the new masked dataset as a json file')
    parser.add_argument('--pair_type', type=str, default='SS')
    parser.add_argument('--data_name', type=str, default='amazon')
    parser.add_argument('--different', action='store_true', help='Whether to change the first text to the same or different topic as the other text from the pair, mask_one_text should be true')
    parser.add_argument('--baseline', action='store_true')
    return parser.parse_args()
 
if __name__ == '__main__':
    args = argument_parser()
    random.seed(0)
    data = load_dataset(args.data_path)

    topic_related = load_dataset(args.topic_related_path)
    nlp = spacy.load('en_core_web_sm')
    
    masked_data = mask_data(data, topic_related, nlp, args.baseline, mask_type=args.mask_type, mask_one_text=args.mask_one_text, different=args.different)

    if args.save:
        create_dataset(f"explainableAV/change_topic/{args.data_name}_new_baseline_lda_{args.pair_type}_{args.mask_type}_{args.mask_one_text}_{args.different}.json", masked_data)
