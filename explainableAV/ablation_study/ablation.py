#########################################################################################
# This file provides the code to perform attention and layer ablation
# To run the code provide 
# --data_path: the path to the data on which you want to perform AV with ablation
# --topic_related_path: the path to the dataset with topic words corresponding with the data from data_path
# --model_name: the model you want to perform the ablation on
# --pair_type: the pair type corresponding with the data from data_path
# --ablate_layers: which layer to ablate (highest probing accuracy, second highest, or both) if you want to ablate the layers
# --ablate_attention: whether you want to perform attention ablation
#
# Example usage:
# python -m explainableAV.ablation_study.ablation --data_path "explainableAV/Amazon/SS_test_15000.json" --model_name "StyleDistance" --pair_type 'SS' --ablate_layers 'second' --ablate_attention
#########################################################################################

import argparse
from sentence_transformers import SentenceTransformer
from transformers import AutoModel
import torch
from explainableAV.utils.utils import load_dataset
from sklearn.metrics import accuracy_score
import numpy as np
from explainableAV.attention.attention import preserve_spaces, preprocess_text_for_recombination
import spacy
from tqdm import tqdm

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

def append_tokens_large(token_count, lemmatized_tokens, lemma_idx, new_tokens, new_pos_tags, pos_tags, pos_tag_idx):
    '''
    Helper function for modify attention mask, which appends tokens based on topic words
    Inputs:
        token_count: int indicating how many times word should be appended
        lemmatized_tokens: list of lemma words
        lemma_idx: int indicating which lemma word to append
        new_tokens: list that stores the new tokens
        new_pos_tags: list that stores the pos_tags
        pos_tags: list of original pos tags
        pos_tag_idx: int indicating which pos_tag to append
    Outputs:
        new_tokens, new_pos_tags
    '''
    try: # in case an index is out of range
        for j in range(token_count):
            new_tokens.append(lemmatized_tokens[lemma_idx])
            new_pos_tags.append(pos_tags[pos_tag_idx])
    except:
        print("error in appending")
    return new_tokens, new_pos_tags

def merge_decimal_tokens(tokens):
    '''
    Preprocess tokens that consists of decimals
    Input:
        tokens: the list of tokens
    Output:
        list of tokens with split decimals
    '''
    merged = []
    i = 0
    while i < len(tokens): 
        if (i + 2 < len(tokens)
            and tokens[i].isdigit()
            and (tokens[i+1] == '.' or tokens [i+1] == "'" or tokens[i+1] == "/" or tokens[i+1] == ',' or tokens[i+1] == ':')
            and tokens[i+2].isdigit()):
            merged.append(tokens[i] + tokens[i+1] + tokens[i+2])
            merged.append(tokens[i] + tokens[i+1] + tokens[i+2])
            merged.append(tokens[i] + tokens[i+1] + tokens[i+2])
            i += 3
        else:
            merged.append(tokens[i])
            i += 1
    return merged

def modify_attention_mask(sentence, attention_mask, tokens, topic_words, model, nlp):
    '''
    Identifies locations of topic word tokens and sets the attention to 0 in the attention mask
    Inputs:
        sentence: original input text
        attention_mask: input attention_mask from the model
        tokens: tokenized input text
        topic_words: list of topic words corresponding with topic of the text
        model: Transformer model (AutoModel)
        nlp: Spacy's NLP
    Output:
        attention_mask with topic word tokens' attention set to 0
    '''
    tokens_space = preserve_spaces(tokens)
    _, tokens_raw, pos_tags, _ = preprocess_text_for_recombination(sentence, nlp)
    tokens_merged = merge_decimal_tokens(tokens_raw)
    tokens_space = merge_decimal_tokens(tokens_space)
    i = 0
    tokens = []
    tokens_with_space = []
    while True: # clean special token cases
        if i == len(tokens_space):
            break
        if tokens_space[i] in ["'t"]:
            tokens_with_space.append('not')
        elif tokens_space[i] in ["'s", "'re", "'m"]:
            tokens_with_space.append('be')
        elif tokens_space[i] in ["'ve", "'d", "ve"]:
            tokens_with_space.append('have')
        elif tokens_space[i] in ["'ll"]:
            tokens_with_space.append('will')
        else:
            tokens_with_space.append(tokens_space[i])
        i += 1

    i = 0
    while True: # clean special token cases
        if i == len(tokens_merged):
            break
        if tokens_merged[i] in ["'t"]:
            tokens.append('not')
        elif tokens_merged[i] in ["'s", "'re", "'m"]:
            tokens.append('be')
        elif tokens_merged[i] in ["'ve", "'d"]:
            tokens.append('have')
        elif tokens_merged[i] in ["'ll"]:
            tokens.append('will')
        elif tokens_merged[i] == 'i.e.,':
            tokens.append('i')
            tokens.append('.')
            tokens.append('e')
            tokens.append('.,')
        elif tokens_merged[i] == 'i.e.':
            tokens.append('i')
            tokens.append('.')
            tokens.append('e')
            tokens.append('.')
        elif tokens_merged[i] == 'iuse':
            tokens.append('i')
            tokens.append('use')
        elif tokens_merged[i] == 'id="video':
            tokens.append('id')
            tokens.append('="')
            tokens.append("video")
        elif tokens_merged[i] == 'class="a':
            tokens.append('class')
            tokens.append('="')
            tokens.append('a')
        elif tokens_merged[i] == 'type="hidden':
            tokens.append('type')
            tokens.append('="')
            tokens.append('hidden')
        elif tokens_merged[i] == 'block"></':
            tokens.append('block')
            tokens.append('"></')
        elif tokens_merged[i] == "div><input":
            tokens.append('div')
            tokens.append('><')
            tokens.append('input')
        elif tokens_merged[i] == 'hook="product':
            tokens.append('hook')
            tokens.append('="')
            tokens.append('product')
        elif tokens_merged[i] == 'href="/':
            tokens.append('href')
            tokens.append('="/')
        elif "." in tokens_merged[i] and not tokens_merged[i].startswith(".") and not tokens_merged[i].endswith("."):
            tokens += tokens_merged[i].split(".")
        elif ")" in tokens_merged[i] and not tokens_merged[i].startswith(")") and not tokens_merged[i].endswith(")"):
            tokens += tokens_merged[i].split(")")
        elif "/" in tokens_merged[i] and not tokens_merged[i].startswith("/") and not tokens_merged[i].endswith("/"):
            tokens += tokens_merged[i].split("/")
        elif "-" in tokens_merged[i] and not tokens_merged[i].startswith("-") and not tokens_merged[i].endswith("-"):
            tokens += tokens_merged[i].split("-")
        elif "'" in tokens_merged[i] and not tokens_merged[i].startswith("'") and not tokens_merged[i].endswith("'"):
            tokens += tokens_merged[i].split("'")
        else:
            tokens.append(tokens_merged[i])
        i += 1

    special_tokens = ['="/', '"></', '><', '*', '<', '.', ',', '--', '="', '!', '?', '?!', '?)', ':-)', '+', ';', ":", '"', ".,", '(', ")", "'", ".)", ").", "),", "", '".', "'t", "..", "...", "....", "-", "!!", "!!!", "!!!!", "/", "  ", " ", "'s", "'ll", "'ve", "'d", "'m", "'re", "%", "\n", "\n\n"]
    new_tokens = []
    new_pos_tags = []
    for token, pos_tag in zip(tokens, pos_tags):
        if token not in special_tokens:
            new_tokens.append(token)
            new_pos_tags.append(pos_tag)

    if model == 'ModernBERT':
        special_cases = ['it', 'i', 'you', '', 'don', 'doesn', 'won', 'wouldn', 'couldn', 'they', 'shouldn']
    elif model == 'StyleDistance':
        special_cases = ['it', 'i', 'you', '', 'don', 'doesn', 'won', 'wouldn', 'couldn', 'they', 'shouldn']
    else:
        special_cases = ['it', 'i', 'you', '-', '', 'don', 'doesn', 'won', 'wouldn', 'couldn', 'they', 'shouldn']
    
    spaces = [' ', '  ']

    tokens_with_space_new = []
    for i, token in enumerate(tokens_with_space):
        if i < len(tokens_with_space) - 1:
            if token.lower() in special_cases and tokens_with_space[i+1].lower() not in special_tokens+['veland', 'nex', 'heit', 'phone', 'ut', 'ib', 'chy', 'ibo', 'us', 'ph', 'ke']:
                tokens_with_space_new.append(token)
                tokens_with_space_new.append(' ')
            elif token in special_tokens and tokens_with_space[i+1] in special_tokens and tokens_with_space[i+1] not in spaces:
                tokens_with_space_new.append(token)
                tokens_with_space_new.append(' ')
            else:
                tokens_with_space_new.append(token)
        else:
            tokens_with_space_new.append(token)
        
        if len(tokens_with_space_new) > 1:
            if tokens_with_space_new[-2] in spaces and tokens_with_space_new[-1] in spaces:
                tokens_with_space_new = tokens_with_space_new[:-1]
        if len(tokens_with_space_new) > 1:
            if tokens_with_space_new[-2] in special_tokens and tokens_with_space_new[-2] not in spaces+['"'] and tokens_with_space_new[-1] not in spaces:
                tokens_with_space_new = tokens_with_space_new[:-1] + [' '] + [tokens_with_space_new[-1]]
        if len(tokens_with_space_new) > 1:
            if tokens_with_space_new[-2] in spaces and tokens_with_space_new[-1] in special_tokens and tokens_with_space_new[-1] not in spaces:
                tokens_with_space_new = tokens_with_space_new[:-2] + [tokens_with_space_new[-1]]
        if len(tokens_with_space_new) > 4:
            if tokens_with_space_new[-4] in special_tokens and tokens_with_space_new[-4] not in spaces and tokens_with_space_new[-3] in spaces and tokens_with_space_new[-2] == '(' and tokens_with_space_new[-1] in spaces:
                tokens_with_space_new = tokens_with_space_new[:-1]
    
    tokens = new_tokens
    pos_tags = new_pos_tags
    tokens_with_space = tokens_with_space_new
    new_tokens = []
    new_pos_tags = []
    token_count = 0
    lemma_idx = 0
    pos_tag_idx = 0
    for j, token in enumerate(tokens_with_space): # identify topic word tokens
        if token in special_tokens or j == len(tokens_with_space) - 1:
            
            if lemma_idx < len(tokens):
                new_tokens, new_pos_tags = append_tokens_large(token_count, tokens, lemma_idx, new_tokens, new_pos_tags, pos_tags, pos_tag_idx)
                if token != ' ':
                    new_tokens.append(token)
                    new_pos_tags.append('NO POS')

                if token in ["", '--', '....', '\n', '*', '?!', '<', '><']:
                    lemma_idx -= 2
                    pos_tag_idx -= 2
                elif token not in [' ', '  ', "'t", "'s", "'ll", "'ve", "'d", "'m", "'re"]:
                    lemma_idx -= 1
                    pos_tag_idx -= 1
            else:
                if token != ' ':
                    new_tokens.append(token)
                    new_pos_tags.append('NO POS')

                if token in ["", '--', '....', '\n', '*', '?!', '<', '><']:
                    lemma_idx -= 2
                    pos_tag_idx -= 2
                elif token not in [' ', '  ', "'t", "'s", "'ll", "'ve", "'d", "'m", "'re"]:
                    lemma_idx -= 1
                    pos_tag_idx -= 1

            token_count = 0
            lemma_idx += 1
            pos_tag_idx += 1
            if len(new_tokens) > 1:
                if new_tokens[-1] == '\n' and new_tokens[-2] == '\n':
                    lemma_idx -= 1
                    pos_tag_idx -= 1
        else:
            token_count += 1

    new_attention_mask = attention_mask.clone()
    for i, (token, pos_tag) in enumerate(zip(new_tokens, new_pos_tags)):
        if token in topic_words and pos_tag in ['NN', 'NNP', 'NNPS', 'NNS']:
            new_attention_mask[0, i+1] = 0

    return new_attention_mask

def ablated_forward(transformer_model, input_ids, attention_mask, model_name, layers_to_ablate):
    '''
    Model forward with ablated layers
    Inputs:
        transformer_model: transformer model corresponding with model_name
        input_ids: input_ids of the input text
        attention_mask: attention_mask as provided by the model
        model_name: name of sentence Transformer model
        layers_to_ablate: list with the layers to ablate
    Output:
        hidden states of input_ids
    '''
    embedding_output = transformer_model.embeddings(input_ids)
    hidden_states = embedding_output

    attention_mask = attention_mask.to(dtype=torch.bool)
    if model_name == "ModernBERT":
        position_ids = torch.arange(input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

    if model_name == 'ModernBERT':
        layers = transformer_model.layers
    else:
        layers = transformer_model.encoder.layer

    for i, layer_module in enumerate(layers):
        if i in layers_to_ablate: # skip layer
            residual = hidden_states
            hidden_states = residual
        else:
            if model_name == "ModernBERT":
                layer_outputs = layer_module(hidden_states, attention_mask=attention_mask, position_ids=position_ids)
            else:
                layer_outputs = layer_module(hidden_states, attention_mask=attention_mask)
            hidden_states = layer_outputs[0]

    return (hidden_states,)

def inference_with_attention(sentence, model, transformer_model, model_name, topic_words, nlp, layers_to_ablate=None, attention=False):
    '''
    Obtain tokenized tokens, attention, and embedding for one text
    Inputs:
        sentence: input text
        model: SentenceTransformer model
        transformer_model: AutoModel corresponding with SentenceTransformer model
        model_name: SentenceTransformer model name
        topic_words: list of topic words corresponding with sentence
        nlp: Spacy's NLP
        layers_to_ablate: list of layers to ablate, if None all layers are kept
        attention: if true perform attention ablation
    Output:
        list of tokenized text, attention matrices, embedding of the text        
    '''
    if model_name == 'LUAR': # input depends on model input size
        tokenized_input = model.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)
    elif model_name == 'StyleDistance':
        tokenized_input = model.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
    elif model_name == 'ModernBERT':
        tokenized_input = model.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=8192)

    tokens = model.tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"][0])

    with torch.no_grad(): # inference with either attention ablation or layer ablation or both
        if attention and layers_to_ablate is not None: # layer and attention ablation
            attention_mask = modify_attention_mask(sentence, tokenized_input['attention_mask'], tokens[1:-1], topic_words, model_name, nlp)
            output = ablated_forward(transformer_model, tokenized_input['input_ids'], attention_mask, model_name, layers_to_ablate)
            attentions = None
        elif attention: # attention ablation
            attention_mask = modify_attention_mask(sentence, tokenized_input['attention_mask'], tokens[1:-1], topic_words, model_name, nlp)
            output = transformer_model(input_ids=tokenized_input['input_ids'], attention_mask=attention_mask)
            attentions = output.attentions
        elif layers_to_ablate is not None: # layer ablation
            attention_mask = tokenized_input['attention_mask']
            output = ablated_forward(transformer_model, tokenized_input['input_ids'], attention_mask, model_name, layers_to_ablate)
            attentions = None
    
    # sentence Transformer mean pooling
    features = {"token_embeddings": output[0], "attention_mask": attention_mask}
    mean_pooling = model[1](features)
    mean_pooling = {key: value.to('cuda') for key, value in mean_pooling.items()}

    if model_name == 'LUAR': # dense layer if present to obtain final embedding
        sentence_embedding = model[2](mean_pooling)['sentence_embedding'] # dense layer
    elif model_name == 'StyleDistance' or model_name == 'ModernBERT':
        sentence_embedding = mean_pooling['sentence_embedding'] # no dense layer

    return tokens, attentions, sentence_embedding

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='explainableAV/Amazon/test_set_15000x4.json')
    parser.add_argument('--topic_related_path', type=str, default="explainableAV/extract_topic/amazon_topic_related_8400_filtered.json")
    parser.add_argument('--model_name', type=str, default="LUAR", help="Model to use, one of: 'LUAR', 'ModernBERT', 'StyleDistance'")
    parser.add_argument('--pair_type', type=str, default='SS')
    parser.add_argument('--ablate_layers', type=str, default=None, help="choose from 'first', 'second', 'both'")
    parser.add_argument('--ablate_attention', action='store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = argument_parser()

    # set model name and threshold
    if args.model_name == "LUAR":
        model_name = "gabrielloiseau/LUAR-MUD-sentence-transformers"
        threshold = 0.37
    elif args.model_name == "StyleDistance":
        model_name = "StyleDistance/styledistance"
        threshold = 0.80
    elif args.model_name == "ModernBERT":
        model_name = 'gabrielloiseau/ModernBERT-base-authorship-verification'
        threshold = 0.86
    else:
        print("Model name not recognised, choose one of: 'LUAR', 'StyleDistance', 'ModernBERT'")

    # load required data and models
    model = SentenceTransformer(model_name)
    transformer_model = AutoModel.from_pretrained(model_name, output_attentions=True)
    data = load_dataset(args.data_path)

    topic_related = load_dataset(args.topic_related_path)
    nlp = spacy.load('en_core_web_sm')

    y_true = [int(line['Label']) for line in data]
    y_pred = []
    print("Model: ", args.model_name)
    print("Data split: ", args.pair_type)
    print("Ablate layers: ", args.ablate_layers)

    # determine which layers to ablate
    if args.ablate_layers is not None:
        probing_acc = load_dataset(f"explainableAV/probes/probing_metrics_{args.model_name}.json")["Test_accuracy"]
        probing_acc.pop("1") # do not select first layer
        if args.model_name == 'LUAR': # do not select last layer
            probing_acc.pop("6")
        elif args.model_name == 'ModernBERT':
            probing_acc.pop("22")
        elif args.model_name == 'StyleDistance':
            probing_acc.pop("12")
        sorted_acc = dict(sorted(probing_acc.items(), key=lambda x: x[1], reverse=True))
        if args.ablate_layers == 'first':
            layers_to_ablate = [int(list(sorted_acc.keys())[0]) - 1]
        elif args.ablate_layers == 'second':
            layers_to_ablate = [int(list(sorted_acc.keys())[1]) - 1]
        elif args.ablate_layers == 'both':
            layers_to_ablate = [int(list(sorted_acc.keys())[0]) - 1, int(list(sorted_acc.keys())[1]) - 1]
        else:
            print("Not a valid way to do layer ablation, choose ablate layers from 'first', 'second', 'both'")
            exit(0)
        print("Layer(s) to ablate: ", layers_to_ablate)

    for line in tqdm(data): # loop through all data
        sentence1 = line["Pair"][0]
        sentence2 = line["Pair"][1]
        topic1 = line["Topics"][0]
        topic2 = line["Topics"][1]

        tokens1, attentions1, embedding1 = inference_with_attention(sentence1, model, transformer_model, args.model_name, topic_related[topic1], nlp, layers_to_ablate=layers_to_ablate, attention=args.ablate_attention)
        tokens2, attentions2, embedding2 = inference_with_attention(sentence2, model, transformer_model, args.model_name, topic_related[topic2], nlp, layers_to_ablate=layers_to_ablate, attention=args.ablate_attention)   
        similarity = model.similarity(embedding1, embedding2).item()
        y_pred.append(similarity)

    print('Accuracy: ', accuracy_score(y_true, binarize(y_pred, threshold)))
