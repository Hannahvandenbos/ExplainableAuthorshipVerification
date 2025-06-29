from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
import torch
from explainableAV.utils.utils import load_dataset, create_dataset
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib as mpl
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
import matplotlib.colors as mcolors
import argparse
import torch.nn.functional as F
import seaborn as sns
import pandas as pd
from explainableAV.utils.roberta_value_zeroing import RobertaLayer
from explainableAV.utils.roberta_globenc import RobertaModel, RobertaConfig
from explainableAV.utils.modernbert_globenc import ModernBertModel, ModernBertConfig, ModernBertEncoderLayer
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_distances
import copy
import spacy
import os
import json
from collections import defaultdict
import random
from tqdm import tqdm
from tabulate import tabulate
from PIL import Image
import io
import re

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

def aggregate_attention(attention_matrix, aggregate_tokens=False, per_layer=False):
    '''
    Aggregate the attention matrices based on what is required for the visualisation, and normalize the values
    Input:
        attention_matrix: attention matrix of shape [L, T, T]
        L: number of layers, T: number of tokens
        aggregate_tokens: Bool if False will aggregate over the layers, if True will aggregate over the queries
        per_layer: Bool if True will return the matrix as is
    Output: aggregated/normalized attention matrix, shape depends on aggregation, [L, T, T], [T, T] or [L, T]
    '''
    if per_layer:
        return attention_matrix

    if aggregate_tokens:
        attention_matrix = attention_matrix.mean(dim=1) # average over the tokens
    else:
        attention_matrix = attention_matrix.mean(dim=0) # average over the layers
    
    return F.normalize(attention_matrix, p=2.0, dim = 1) # normalise

def raw_attention(attentions):
    '''
    Aggregate and normalize raw attention scores for visualisation
    Input:
        attentions: raw attentions from model output, shape: [L, H, T, T]
        L: number of layers, H: number of heads, T: number of tokens
    Output:
        attention matrix, shape is [T, T] or [L, T]
    '''
    attention_heads_averaged = []
    for i, attention_layer in enumerate(attentions):
        average = attention_layer.squeeze(0).mean(dim=0) # average over the heads
        attention_heads_averaged.append(average)
    attention_matrix = torch.stack(attention_heads_averaged)
    return attention_matrix

def attention_rollout(attentions):
    '''
    Compute, aggregate, and normalize rollout attention scores for visualisation
    Input:
        attentions: raw attentions from model output, shape: [L, H, T, T]
        L: number of layers, H: number of heads, T: number of tokens
    Output:
        attention matrix, shape is [T, T] or [L, T]
    '''
    attention_rollouts = {}
    attention_per_token = {}
    for i, attention_layer in enumerate(attentions):
        attention_matrix = attention_layer.squeeze(0).mean(dim=0) # average over the heads
        raw_attention = 0.5 * attention_matrix + 0.5 * torch.eye(attention_matrix.shape[0]) # take residual layer into account
        if i == 0:
            attention_rollouts[i] = raw_attention
        else:
            attention_rollouts[i] = raw_attention @ attention_rollouts[i-1]
    attention_matrix = torch.stack([attention_rollout for key, attention_rollout in attention_rollouts.items()])
    return attention_matrix

##################################################################
# Implementation of compute_joint_attention and value_zeroing by: hmohebbi
# Github link: https://github.com/hmohebbi/ValueZeroing/blob/main/README.md
# Paper link: https://arxiv.org/pdf/2301.12971
# Altered parts indicated by 'changed implementation' comments
##################################################################
def compute_joint_attention(att_mat, res=True):
    if res:
        residual_att = torch.eye(att_mat.size(1), device=att_mat.device).unsqueeze(0)
        att_mat = att_mat + residual_att
        att_mat = att_mat / att_mat.sum(dim=-1, keepdim=True)

    joint_attentions = torch.zeros_like(att_mat)
    layers = joint_attentions.size(0)
    joint_attentions[0] = att_mat[0]

    for i in range(1, layers):
        joint_attentions[i] = torch.matmul(att_mat[i], joint_attentions[i-1])

    return joint_attentions

def value_zeroing_numpy(sentence, model, transformer_model, model_name):
    '''
    Original version of value zeroing using numpy instead of torch
    '''
    all_valuezeroing_scores = []
    all_rollout_valuezeroing_scores = []

    if model_name == 'LUAR':
        inputs = model.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)
    elif model_name == 'StyleDistance':
        inputs = model.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
    elif model_name == 'ModernBERT':
        inputs = model.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=8192)
    
    tokens = model.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    with torch.no_grad():
        outputs = transformer_model(inputs['input_ids'], attention_mask=inputs['attention_mask'], output_hidden_states=True, output_attentions=False)

    org_hidden_states = torch.stack(outputs['hidden_states']).squeeze(1)
    input_shape = inputs['input_ids'].size() 
    batch_size, seq_length = input_shape

    # layerwise zeroing value
    score_matrix = np.zeros((len(transformer_model.encoder.layer), seq_length, seq_length))
    extended_blanking_attention_mask: torch.Tensor = transformer_model.get_extended_attention_mask(inputs['attention_mask'], input_shape)
    if model_name == 'LUAR' or model_name == 'StyleDistance':
        for l, layer_module in enumerate(transformer_model.encoder.layer):
            org_hidden_state = org_hidden_states[l].unsqueeze(0)
            y = org_hidden_states[l+1].detach().cpu().numpy()
            for t in range(seq_length):
                with torch.no_grad():
                    layer_outputs = layer_module(org_hidden_state, # previous layer's original output 
                                                attention_mask=extended_blanking_attention_mask,
                                                output_attentions=False,
                                                zero_value_index=t,
                                                )
                hidden_states = layer_outputs[0].squeeze().detach().cpu().numpy()
                # compute similarity between original and new outputs
                # cosine
                x = hidden_states
                
                distances = cosine_distances(x, y).diagonal()
                score_matrix[l, :, t] = distances # per layer, influence per query of zeroing a token 

    elif model_name == 'ModernBERT':
        for l, layer_module in enumerate(transformer_model.layers):
            seq_len = inputs['attention_mask'].size(1)
            position_ids = torch.arange(seq_len, dtype=torch.long, device=inputs['attention_mask'].device)
            position_ids = position_ids.unsqueeze(0).expand(inputs['attention_mask'].size())
            org_hidden_state = org_hidden_states[l].unsqueeze(0)
            y = org_hidden_states[l+1].detach().cpu().numpy()

            for t in range(seq_length):
                with torch.no_grad():
                    layer_outputs = layer_module(org_hidden_state, # previous layer's original output 
                                                attention_mask=extended_blanking_attention_mask,
                                                position_ids=position_ids,
                                                output_attentions=False,
                                                zero_value_index=t,
                                                )
                hidden_states = layer_outputs[0].squeeze().detach().cpu().numpy()
                # compute similarity between original and new outputs
                # cosine
                x = hidden_states
                
                distances = cosine_distances(x, y).diagonal()
                score_matrix[l, :, t] = distances # per layer, influence per query of zeroing a token 
    
    valuezeroing_scores = score_matrix / np.sum(score_matrix, axis=-1, keepdims=True) 
    rollout_valuezeroing_scores = compute_joint_attention(valuezeroing_scores, res=False)

    return torch.tensor(valuezeroing_scores), torch.tensor(rollout_valuezeroing_scores), tokens

def value_zeroing(sentence, model, transformer_model, model_name):
    '''
    Slightly sped up version of value zeroing
    '''
    all_valuezeroing_scores = []
    all_rollout_valuezeroing_scores = []

    max_length = {
        'LUAR': 128,
        'StyleDistance': 512,
        'ModernBERT': 8192
    }[model_name]
    inputs = model.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=max_length)

    device = next(transformer_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    tokens = model.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    with torch.no_grad():
        outputs = transformer_model(
            inputs['input_ids'], 
            attention_mask=inputs['attention_mask'], 
            output_hidden_states=True, 
            output_attentions=False
        )

    org_hidden_states = torch.stack(outputs['hidden_states']).squeeze(1)
    input_shape = inputs['input_ids'].size()
    batch_size, seq_length = input_shape

    # Layerwise zeroing value
    num_layers = len(transformer_model.encoder.layer) if model_name in ['LUAR', 'StyleDistance'] else len(transformer_model.layers)
    score_matrix = torch.zeros((num_layers, seq_length, seq_length), device=device)

    extended_attention_mask = transformer_model.get_extended_attention_mask(inputs['attention_mask'], input_shape).to(device)

    if model_name in ['LUAR', 'StyleDistance']:
        layers = transformer_model.encoder.layer

        for l, layer_module in enumerate(layers):
            org_hidden_state = org_hidden_states[l].unsqueeze(0)
            y = org_hidden_states[l+1].detach()

            for t in range(seq_length):
                with torch.no_grad():
                    layer_outputs = layer_module(
                        org_hidden_state,
                        attention_mask=extended_attention_mask,
                        output_attentions=False,
                        zero_value_index=t,
                    )
                    hidden_states = layer_outputs[0].squeeze(0)

                x = hidden_states
                cosine_sim = F.cosine_similarity(x.unsqueeze(1), y.squeeze(0).unsqueeze(0), dim=-1)
                distances = 1 - torch.diagonal(cosine_sim, 0)
                score_matrix[l, :, t] = distances
    elif model_name == 'ModernBERT':
        for l, layer_module in enumerate(transformer_model.layers):
            seq_len = inputs['attention_mask'].size(1)
            position_ids = torch.arange(seq_len, dtype=torch.long, device=inputs['attention_mask'].device)
            position_ids = position_ids.unsqueeze(0).expand(inputs['attention_mask'].size())
            org_hidden_state = org_hidden_states[l].unsqueeze(0)
            y = org_hidden_states[l+1].detach()

            for t in range(seq_length):
                with torch.no_grad():
                    layer_outputs = layer_module(org_hidden_state, # previous layer's original output 
                                                attention_mask=extended_attention_mask,
                                                position_ids=position_ids,
                                                output_attentions=False,
                                                zero_value_index=t,
                                                )
                    hidden_states = layer_outputs[0].squeeze(0)
                x = hidden_states
                cosine_sim = F.cosine_similarity(x.unsqueeze(1), y.squeeze(0).unsqueeze(0), dim=-1) 
                distances = 1 - torch.diagonal(cosine_sim, 0)
                score_matrix[l, :, t] = distances


    valuezeroing_scores = score_matrix / (score_matrix.sum(dim=-1, keepdim=True) + 1e-8)
    rollout_valuezeroing_scores = compute_joint_attention(valuezeroing_scores, res=False)

    return valuezeroing_scores.cpu(), rollout_valuezeroing_scores.cpu(), tokens

##################################################################
# Implementation of compute_joint_attention_globenc and compute_flows by: mohsenfayyaz
# Github link: https://github.com/mohsenfayyaz/GlobEnc/blob/main/GlobEnc_Demo.ipynb
# Paper link: https://arxiv.org/pdf/2205.03286
# Altered parts indicated by 'changed implementation' comments
##################################################################
def compute_joint_attention_globenc(att_mat):
    joint_attentions = np.zeros(att_mat.shape)
    layers = joint_attentions.shape[0]
    joint_attentions[0] = att_mat[0]
    for i in np.arange(1, layers):
        joint_attentions[i] = att_mat[i].dot(joint_attentions[i - 1])

    return joint_attentions

def compute_flows(attentions_list, desc="", output_hidden_states=False, num_cpus=0, disable_tqdm=False):
        """
        :param attentions_list: list of attention maps (#examples, #layers, #sent_len, #sent_len)
        :param desc:
        :param output_hidden_states:
        :param num_cpus:
        :return:
        """
        attentions_rollouts = []
        for i in range(len(attentions_list)):
            if output_hidden_states:
                attentions_rollouts.append(compute_joint_attention_globenc(attentions_list[i]))
            else:
                attentions_rollouts.append(compute_joint_attention_globenc(attentions_list[i])[[-1]])
        return attentions_rollouts

def globenc(sentence, model, transformer_model, model_name):
    if model_name == 'LUAR':
        tokenized_input = model.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)
    elif model_name == 'StyleDistance':
        tokenized_input = model.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
    elif model_name == 'ModernBERT':
        tokenized_input = model.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=8192)

    tokens = model.tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"][0])
    with torch.no_grad():
        outputs = transformer_model(input_ids=tokenized_input['input_ids'], attention_mask=tokenized_input['attention_mask'], output_attentions=True, output_norms=True, return_dict=False)
        attentions = outputs[-2]
        norms = outputs[-1]
        num_layers = len(attentions)
        norm_nenc = torch.stack([norms[i][4] for i in range(num_layers)]).squeeze().cpu().numpy()

        globenc = compute_flows([norm_nenc], output_hidden_states=True)[0]
    return norm_nenc, torch.tensor(globenc), tokens 

def inference(sent1, sent2, model):
    '''
    Compute similarity between embeddings with SentenceTransformer model
    Inputs:
        sent1: first text in a pair
        sent2: second text in a pair
        model: sentenceTransformer model
    Output:
        similarity between the embeddings
    '''
    emb1 = torch.tensor(model.encode(sent1)).unsqueeze(0)
    emb2 = torch.tensor(model.encode(sent2)).unsqueeze(0)
    similarity = model.similarity(emb1, emb2).item()
    return similarity

def inference_with_attention(sentence, model, transformer_model, model_name):
    '''
    Obtain tokenized tokens, attention, and embedding for one text
    Inputs:
        sentence: input text
        model: SentenceTransformer model
        transformer_model: AutoModel corresponding with SentenceTransformer model
        model_name: SentenceTransformer model name
    Outputs:
        list of tokenized text, attention matrices, embedding of the text        
    '''
    if model_name == 'LUAR': # input depends on model input size
        tokenized_input = model.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)
    elif model_name == 'StyleDistance':
        tokenized_input = model.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
    elif model_name == 'ModernBERT':
        tokenized_input = model.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=8192)

    tokens = model.tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"][0])
    with torch.no_grad():
        output = transformer_model(input_ids=tokenized_input['input_ids'], attention_mask=tokenized_input['attention_mask'])
    
    # sentence Transformer mean pooling
    features = {"token_embeddings": output[0], "attention_mask": tokenized_input["attention_mask"]}
    mean_pooling = model[1](features)
    mean_pooling = {key: value.to('cuda') for key, value in mean_pooling.items()} # move tensors to cuda

    if model_name == 'LUAR': # dense layer if present to obtain final embedding
        sentence_embedding = model[2](mean_pooling)['sentence_embedding'] # dense layer
    elif model_name == 'StyleDistance' or model_name == 'ModernBERT':
        sentence_embedding = mean_pooling['sentence_embedding'] # no dense layer

    return tokens, output.attentions, sentence_embedding

def confidence(similarity, threshold):
    '''
    Transform embedding similarity to confidence score with softmax function
    Inputs:
        similarity: similarity between embeddings of text pair
        threshold: threshold for model
    Output:
        confidence score
    '''
    return 1 / (1 + np.exp(-10 * (similarity - threshold)))

def confidence_V2(similarity, threshold, label):
    '''
    Transform embedding similarity to confidence score with softmax function relative to predicted class
    Inputs:
        similarity: similarity between embeddings of text pair
        threshold: threshold for model
    Output:
        confidence score
    '''
    sigmoid = 1 / (1 + np.exp(-10 * (similarity - threshold)))
    if label == 0:
        return 1 - sigmoid
    elif label == 1:
        return sigmoid


def attention_heatmap(attention_matrix1, tokens1, attention_matrix2, tokens2, confidence, save_name, aggregate_tokens=False):
    '''
    Visualise attention heatmap for a text pair per layer or per token
    Inputs:
        attention_matrix1: aggregated attention matrix for text 1 of size [L, T] or [T, T]
        L: number of layers, T: number of tokens
        tokens1: tokens as tokenized by model for text 1
        attention_matrix2: aggregated attention matrix for text 2 of size [L, T] or [T, T]
        tokens2: tokens as tokenized by model for text 2
        confidence: confidence score (float) of prediction by model for this text pair
        save_name: file name to save the image as
        aggregate_tokens: Bool if False assumes aggregation over the layers, if True assumes aggregation over the queries
    '''
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    combined = torch.cat((attention_matrix1.flatten(), attention_matrix2.flatten()))
    min_val = combined.min()
    max_val = combined.max()

    attention_matrix1 = (attention_matrix1 - min_val) / (max_val - min_val)
    attention_matrix2 = (attention_matrix2 - min_val) / (max_val - min_val)
    
    tokens1 = pretty_tokens(tokens1)
    if aggregate_tokens:
        df1 = pd.DataFrame(attention_matrix1.detach().numpy(), index=np.arange(attention_matrix1.shape[0])+1, columns=tokens1).iloc[::-1]
    else:
        df1 = pd.DataFrame(attention_matrix1.detach().numpy(), index=tokens1, columns=tokens1)
    sns.heatmap(df1, ax=ax1, cmap="flare", cbar=False)
    ax1.set_title(f"Attention Weights - Sentence 1", fontsize=16)
    ax1.set_xlabel("Attended to")
    ax1.set_ylabel("Attending")
    ax1.set_yticks(np.arange(len(tokens1)) + 0.5)
    ax1.set_yticklabels(tokens1, rotation=0, fontsize=10)
    ax1.set_xticks(np.arange(len(tokens1)) + 0.5)
    ax1.set_xticklabels(tokens1, rotation=90, fontsize=10)

    tokens2 = pretty_tokens(tokens2)
    if aggregate_tokens:
        df2 = pd.DataFrame(attention_matrix2.detach().numpy(), index=np.arange(attention_matrix2.shape[0])+1, columns=tokens2).iloc[::-1]
    else:
        df2 = pd.DataFrame(attention_matrix2.detach().numpy(), index=tokens2, columns=tokens2)

    sns.heatmap(df2, ax=ax2, cmap="flare")
    ax2.set_title(f"Attention Weights - Sentence 2", fontsize=16)
    ax2.set_xlabel("Attended to")
    ax2.set_ylabel("Attending")

    ax2.set_yticks(np.arange(len(tokens2)) + 0.5)
    ax2.set_yticklabels(tokens2, rotation=0, fontsize=10)
    ax2.set_xticks(np.arange(len(tokens2)) + 0.5)
    ax2.set_xticklabels(tokens2, rotation=90, fontsize=10)

    fig.suptitle(f"Confidence: {confidence:.3f}", fontsize=20)

    plt.tight_layout()
    plt.savefig(f"explainableAV/attention_images/{save_name}")
    plt.show()

def attention_over_tokens_per_layer(attentions1, attentions2, tokens1, tokens2, save_name):
    '''
    Visualise attention matrices for a text pair over the tokens per layer
    Input:
        attentions1: attention matrix for text 1 of size [L, T, T]
        L: number of layers, T: number of tokens
        attentions2: attention matrix for text 2 of size [L, T, T]
        tokens1: tokens as tokenized by model for text 1
        tokens2: tokens as tokenized by model for text 2
        save_name: file name to save the image as
    '''
    modified_tokens = pretty_tokens(tokens1)
    modified_tokens2 = pretty_tokens(tokens2)

    all_attentions = np.concatenate([attention.flatten() for attention in attentions1])
    vmin = all_attentions.min()
    vmax = all_attentions.max()
    
    fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(25, 8))
    axes = axes.flatten()

    for i, attention in enumerate(attentions1):
        ax = axes[i]
        im = ax.imshow(attention, aspect='auto', cmap="flare", vmin=vmin, vmax=vmax)

        ax.set_xticks(range(len(tokens1)), labels=modified_tokens, rotation=90, ha="right")
        ax.set_yticks(range(len(tokens1)), labels=modified_tokens)
        ax.set_title(f"Layer {i+1}", fontsize=12)
        ax.set_xlabel("Tokens")
        ax.set_ylabel("Tokens")

    for i, attention in enumerate(attentions2):
        ax = axes[i+6]
        im = ax.imshow(attention, aspect='auto', cmap="flare")

        ax.set_xticks(range(len(tokens2)), labels=modified_tokens2, rotation=90, ha="right", rotation_mode="anchor")
        ax.set_yticks(range(len(tokens2)), labels=modified_tokens2)
        ax.set_title(f"Layer {i+1}", fontsize=12)
        ax.set_xlabel("Tokens")
        ax.set_ylabel("Tokens")

    fig.tight_layout()
    plt.savefig(f"explainableAV/attention_images/{save_name}.png")
    plt.show()

def tokens_for_text_plot(tokens1, tokens2, values):
    '''
    Prepare tokens for the text plot by including spaces and new lines
    Inputs:
        tokens1: tokens as tokenized by model for text 1
        tokens2: tokens as tokenized by model for text 2
        values: aggregated attention scores for both texts
    Outputs:
        prepared tokens1, prepared tokens2, values with 0 for spaces or end of line symbols
    '''
    new_tokens1 = preserve_spaces(tokens1)
    new_tokens2 = preserve_spaces(tokens2)
    new_values = []
    i = 0
    for token in new_tokens1+new_tokens2:
        if token == ' ' or token == '\n':
            new_values.append(0)
        else:
            new_values.append(values[i])
            i += 1
    return new_tokens1, new_tokens2, new_values

def change_width(width):
    '''
    Alter the widths of the tokens for the custom text plot
    Input:
        width
    Output:
        altered width
    '''
    if width <= 0.15:
        width = 0.024
    if width > 0.15 and width <= 0.2:
        width = 0.0355
    elif width > 0.25 and width <= 0.3:
        width = 0.046
    elif width > 0.3 and width <= 0.35:
        width = 0.057
    elif width > 0.4 and width <= 0.45:
        width = 0.068
    elif width > 0.45 and width <= 0.5:
        width = 0.077
    elif width > 0.5 and width <= 0.55:
        width = 0.0785
    elif width > 0.55 and width <= 0.6:
        width = 0.09
    elif width > 0.65 and width <= 0.7:
        width = 0.1
    elif width > 0.75 and width <= 0.8:
        width = 0.11
    elif width > 0.8 and width <= 0.85:
        width = 0.125
    elif width > 0.9 and width <= 0.95:
        width = 0.13
    elif width > 1.0 and width <= 1.05:
        width = 0.1425
    elif width > 1.05 and width <= 1.1:
        width = 0.15
    elif width > 1.15 and width <= 1.2:
        width = 0.16
    elif width > 1.3 and width <= 1.35:
        width = 0.185
    elif width > 1.75 and width < 1.8:
        width = 2
    return width + 0.003

def custom_text_plot(attention_matrix1, attention_matrix2, tokens1, tokens2, sentence1, sentence2, topic_words1, topic_words2, pair_type, model, save_name):
    '''
    Manual text plot with the text pairs highlighting attention
    Input:
        attention_matrix1: attention matrix of text 1 aggregated over eitht the layers or queries
        attention_matrix2: attention matrix of text 2 aggregated over eitht the layers or queries
        tokens1: tokens as tokenized by model for text 1
        tokens2: tokens as tokenized by model for text 2
        sentence1: masked input text
        sentence2: masked input text2
        topic_words1: list of topic words corresponding with topic of first text
        topic_words2: list of topic words corresponding with topic of second text
        pair_type: name of pair type as string, for example 'SS'
        save_name: file name to save the image as
    '''
    attention_matrix1 = attention_matrix1.mean(dim=0)[1:-1] # aggregate over layers or queries
    attention_matrix2 = attention_matrix2.mean(dim=0)[1:-1] # aggregate over layers or queries
    values = attention_matrix1.tolist() + attention_matrix2.tolist()
    tokens1, tokens2, values = tokens_for_text_plot(tokens1, tokens2, values)
 
    custom_orange_cmap = LinearSegmentedColormap.from_list(
        "custom_orange",
        ["#FFFFFF", "#ffa849", "#B34700"]
    )

    cmap_red = custom_orange_cmap
    vmin = np.min(values)
    vmax = 0.43
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    alpha_values = np.interp(values, (vmin, vmax), (0.2, 0.8))
    token_colors = [(*cmap_red(norm(value))[:3], 0.8) for value in values]

    sm = plt.cm.ScalarMappable(cmap=cmap_red, norm=norm)
    sm.set_array([])

    max_values_idx = np.argsort(np.abs(values))[::-1][:10]
    values_to_print = [value if i in max_values_idx else None for i, value in enumerate(values)]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    xlim = ax.get_xlim() 
    ylim = ax.get_ylim()

    center_x = (xlim[0] + xlim[1]) / 2
    center_y = (ylim[0] + ylim[1]) / 2

    nlp = spacy.load('en_core_web_sm')
    attention_for_topic_words1, attention_for_non_topic_words1, topic_words1, _ = topic_attention_for_plot(sentence1, attention_matrix1, tokens1, topic_words1, model, nlp)
    attention_for_topic_words2, attention_for_non_topic_words2, topic_words2, _ = topic_attention_for_plot(sentence2, attention_matrix2, tokens2, topic_words2, model, nlp)

    # compute relative topic-attention ratio
    relative_ratio1 = (sum(attention_for_topic_words1) / len(attention_for_topic_words1) - sum(attention_for_non_topic_words1) / len(attention_for_non_topic_words1)) / attention_matrix1.sum()

    ax.text(0.0, 1.0, f"Text 1 (Relative Topic-Attention Ratio: {relative_ratio1:.3f})", ha='left', va='center', fontsize=10, color='black', fontweight='normal', fontname='serif')
    
    # first text
    y_pos = 0.9
    x_pos = 0.0
    lowest_y = y_pos
    for i, (token, color, value) in enumerate(zip(tokens1, token_colors[:len(tokens1)], values_to_print[:len(tokens1)])):
        if token == ' ':
            x_pos += 0.015
        elif token == '\n':
            y_pos -= 0.1  # next line
            x_pos = 0.0 # reset x-axis
        else:
            if x_pos >= center_x - 0.1:
                y_pos -= 0.1  # next line
                x_pos = 0.0 # reset x-axis
            if token in topic_words1:
                text_obj = ax.text(x_pos, y_pos, token, ha='left', va='center', fontsize=10, color='black', fontweight='bold', fontname='monospace', bbox=dict(facecolor=color, edgecolor='none', boxstyle="round,pad=0.2"))
            else:
                text_obj = ax.text(x_pos, y_pos, token, ha='left', va='center', fontsize=10, color='black', fontweight='normal', fontname='monospace', bbox=dict(facecolor=color, edgecolor='none', boxstyle="round,pad=0.2"))
            width = text_obj.get_window_extent().width / fig.dpi
            width = change_width(width)

            x_pos += width -0.007
        if y_pos < lowest_y:
            lowest_y = y_pos

    relative_ratio2 = (sum(attention_for_topic_words2) / len(attention_for_topic_words2) - sum(attention_for_non_topic_words2) / len(attention_for_non_topic_words2)) / attention_matrix2.sum()

    ax.text(0.0, lowest_y - 0.1, f"Text 2 (Relative Topic-Attention Ratio: {relative_ratio2:.3f})", ha='left', va='center', fontsize=10, color='black', fontweight='normal', fontname='serif')

    # second text
    y_pos = lowest_y - 0.2
    x_pos = 0.0
    for i, (token, color, value) in enumerate(zip(tokens2, token_colors[len(tokens1):], values_to_print[len(tokens1):])):
        if token == ' ':
            x_pos += 0.015
        elif token == '\n':
            y_pos -= 0.1  # next line
            x_pos = 0.0 # reset x-axis
        else:
            if x_pos >= center_x - 0.1:
                y_pos -= 0.1  # next line
                x_pos = 0.0
            if token in topic_words2:
                text_obj = ax.text(x_pos, y_pos, token, ha='left', va='center', fontsize=10, color='black', fontweight='bold', fontname='monospace', bbox=dict(facecolor=color, edgecolor='none', boxstyle="round,pad=0.2"))
            else:
                text_obj = ax.text(x_pos, y_pos, token, ha='left', va='center', fontsize=10, color='black', fontweight='normal', fontname='monospace', bbox=dict(facecolor=color, edgecolor='none', boxstyle="round,pad=0.2"))
            width = text_obj.get_window_extent().width / fig.dpi
            width = change_width(width)

            x_pos += width - 0.007
        
        if y_pos < lowest_y:
            lowest_y = y_pos

    cbar = fig.colorbar(
        sm, ax=ax,
        orientation='vertical',
        fraction=0.03,
        pad=0.02,
        shrink=0.45       
    )

    # save and crop image
    cbar.set_label('Attention Score', fontsize=10)
    pos = cbar.ax.get_position()
    vertical_offset = 0.05 if lowest_y > 0.55 else 0.0
    print(lowest_y, vertical_offset)
    new_pos = [center_x - 0.03, lowest_y - vertical_offset, pos.width, pos.height]
    cbar.ax.set_position(new_pos)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close()

    buf.seek(0)
    img = Image.open(buf)
    width, height = img.size

    crop_bottom_pixel = int(height * (1 - (lowest_y - 0.1)))
    crop_right_pixel = int(width * (center_x+0.08))

    cropped_img = img.crop((0, 0, crop_right_pixel,  crop_bottom_pixel))
    cropped_img = cropped_img.convert("RGB")
    cropped_img.save(f"explainableAV/attention_images/{save_name}")


def custom_masked_text_plot(attention_matrix1, attention_matrix2, tokens1, tokens2, sentence1, sentence2, sentence1_orig, sentence2_orig, topic_words1, topic_words2, pair_type, save_name):
    '''
    Manual text plot with the text pairs highlighting attention for masked texts
    Inputs:
        attention_matrix1: attention matrix of text 1 aggregated over eitht the layers or queries
        attention_matrix2: attention matrix of text 2 aggregated over eitht the layers or queries
        tokens1: tokens as tokenized by model for text 1
        tokens2: tokens as tokenized by model for text 2
        sentence1: masked input text
        sentence2: masked input text2
        sentence1_orig: original input text
        sentence2_orig: original input text2
        topic_words1: list of topic words corresponding with topic of first text
        topic_words2: list of topic words corresponding with topic of second text
        pair_type: name of pair type as string, for example 'SS'
        save_name: file name to save the image as
    '''
    attention_matrix1 = attention_matrix1.mean(dim=0)[1:-1] # aggregate over layers or queries
    attention_matrix2 = attention_matrix2.mean(dim=0)[1:-1] # aggregate over layers or queries
    values = attention_matrix1.tolist() + attention_matrix2.tolist()
    tokens1, tokens2, values = tokens_for_text_plot(tokens1, tokens2, values)

    custom_orange_cmap = LinearSegmentedColormap.from_list(
        "custom_orange",
        ["#FFFFFF", "#ffa849", "#B34700"]
    )

    cmap_red = custom_orange_cmap
    vmin = np.min(values)
    vmax = np.max(values)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    alpha_values = np.interp(values, (vmin, vmax), (0.2, 0.8))
    token_colors = [(*cmap_red(norm(value))[:3], 0.8) for value in values]

    sm = plt.cm.ScalarMappable(cmap=cmap_red, norm=norm)
    sm.set_array([])

    max_values_idx = np.argsort(np.abs(values))[::-1][:10]
    values_to_print = [value if i in max_values_idx else None for i, value in enumerate(values)]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    xlim = ax.get_xlim() 
    ylim = ax.get_ylim()

    center_x = (xlim[0] + xlim[1]) / 2
    center_y = (ylim[0] + ylim[1]) / 2

    nlp = spacy.load('en_core_web_sm')
    attention_for_topic_words1, attention_for_non_topic_words1, topic_words1 = topic_attention_masked_for_plot(sentence1, sentence1_orig, attention_matrix1, tokens1, topic_words1, nlp)
    attention_for_topic_words2, attention_for_non_topic_words2, topic_words2 = topic_attention_masked_for_plot(sentence2, sentence2_orig, attention_matrix2, tokens2, topic_words2, nlp)

    relative_ratio1 = (sum(attention_for_topic_words1) / len(attention_for_topic_words1) - sum(attention_for_non_topic_words1) / len(attention_for_non_topic_words1)) / attention_matrix1.sum()

    ax.text(0.0, 1.0, f"Text 1 (Topic-Attention Ratio: {relative_ratio1:.3f})", ha='left', va='center', fontsize=12, color='black', fontweight='normal', fontname='serif')

    # first text
    y_pos = 0.9
    x_pos = 0.0
    lowest_y = y_pos
    for i, (token, color, value) in enumerate(zip(tokens1, token_colors[:len(tokens1)], values_to_print[:len(tokens1)])):
        if token == ' ':
            x_pos += 0.015
        elif token == '\n':
            y_pos -= 0.1  # next line
            x_pos = 0.0 # reset x-axis
        else:
            if x_pos >= center_x - 0.1:
                y_pos -= 0.1  # next line
                x_pos = 0.0 # reset x-axis
            if token in topic_words1:
                text_obj = ax.text(x_pos, y_pos, token, ha='left', va='center', fontsize=10, color='black', fontweight='bold', fontname='monospace', bbox=dict(facecolor=color, edgecolor='none', boxstyle="round,pad=0.2"))
            else:
                text_obj = ax.text(x_pos, y_pos, token, ha='left', va='center', fontsize=10, color='black', fontweight='normal', fontname='monospace', bbox=dict(facecolor=color, edgecolor='none', boxstyle="round,pad=0.2"))
            width = text_obj.get_window_extent().width / fig.dpi
            width = change_width(width)

            x_pos += width -0.007
        if y_pos < lowest_y:
            lowest_y = y_pos

    relative_ratio2 = (sum(attention_for_topic_words2) / len(attention_for_topic_words2) - sum(attention_for_non_topic_words2) / len(attention_for_non_topic_words2)) / attention_matrix2.sum()

    ax.text(0.0, lowest_y - 0.1, f"Text 2 (Topic-Attention Ratio: {relative_ratio2:.3f})", ha='left', va='center', fontsize=12, color='black', fontweight='normal', fontname='serif')

    # second text
    y_pos = lowest_y - 0.2
    x_pos = 0.0
    for i, (token, color, value) in enumerate(zip(tokens2, token_colors[len(tokens1):], values_to_print[len(tokens1):])):
        if token == ' ':
            x_pos += 0.015
        elif token == '\n':
            y_pos -= 0.1  # next line
            x_pos = 0.0 # reset x-axis
        else:
            if x_pos >= center_x - 0.1:
                y_pos -= 0.1  # next line
                x_pos = 0.0
            if token in topic_words2:
                text_obj = ax.text(x_pos, y_pos, token, ha='left', va='center', fontsize=10, color='black', fontweight='bold', fontname='monospace', bbox=dict(facecolor=color, edgecolor='none', boxstyle="round,pad=0.2"))
            else:
                text_obj = ax.text(x_pos, y_pos, token, ha='left', va='center', fontsize=10, color='black', fontweight='normal', fontname='monospace', bbox=dict(facecolor=color, edgecolor='none', boxstyle="round,pad=0.2"))
            width = text_obj.get_window_extent().width / fig.dpi
            width = change_width(width)

            x_pos += width - 0.007
        
        if y_pos < lowest_y:
            lowest_y = y_pos

    cbar = fig.colorbar(
        sm, ax=ax,
        orientation='horizontal',
        fraction=0.03,   
        pad=0.02,
        shrink=1.0
    )

    # save and crop image
    cbar.set_label('Attention Score', fontsize=14)
    pos = cbar.ax.get_position()
    new_pos = [pos.x0, lowest_y-0.12, pos.width, pos.height]
    cbar.ax.set_position(new_pos)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close()

    buf.seek(0)
    img = Image.open(buf)
    width, height = img.size

    crop_bottom_pixel = int(height * (1 - (lowest_y - 0.32)))

    cropped_img = img.crop((0, 0, width, crop_bottom_pixel))
    cropped_img = cropped_img.convert("RGB")
    cropped_img.save(f"explainableAV/attention_images/{save_name}")

def topic_attention_for_plot(sentence, att_mat, tokens, topic_words, model, nlp):
    '''
    Find attention distributed to topic word tokens
    Inputs:
        sentence: input text
        att_mat: input attention matrix (averaged over heads)
        tokens: input tokens for input text
        topic_words: list of topic words corresponding with topic of the text
        model: Transformer model (AutoModel)
        nlp: Spacy's NLP
    Outputs:
        list of attention values for topic words, 
        list of attention values for non-topic words,
        list of topic words found in text,
        list of indices corresponding with topic words
    '''
    tokens_with_space = preserve_spaces(tokens)
    original_data, tokens, pos_tags, orig_to_token = preprocess_text_for_recombination(sentence, nlp)

    if model == 'ModernBERT':
        tokens_no_lemma = [token for token in tokens_with_space if token != ' ' and token != '\n' and token != '  ']
    else:
        tokens_no_lemma = [token for token in tokens_with_space if token != ' ' and token != '\n']
        
    special_tokens = ['.', ',', '!', '?', ';', ':', '"', '(', ")", "'", ").", "'t", "...", "-", "''", "!!", "  ", " ", "\n"]

    new_tokens = []
    new_pos_tags = []
    new_att_mat = []
    token_count = 0
    lemma_idx = 0
    pos_tag_idx = 0
    att_mat_idx = 0
    for j, token in enumerate(tokens_with_space): # find tokens corresponding with topic words
        if token in special_tokens or j == len(tokens_with_space) - 1:
            if model == 'ModernBERT':
                if lemma_idx < len(tokens) - 1:
                    if tokens[lemma_idx] == ' ':
                        lemma_idx += 1
                        pos_tag_idx += 1
                        att_mat_idx += 1
            if lemma_idx < len(tokens):
                if tokens[lemma_idx] in special_tokens and token not in ['  ', ' ', '\n']:
                    new_tokens.append(token)
                    new_pos_tags.append('NO POS')
            if (j == len(tokens_with_space) - 1 and token not in special_tokens) or token == "'t":
                token_count += 1
            
            if lemma_idx < len(tokens):
                new_tokens, lemma_idx, new_pos_tags, new_att_mat, att_mat_idx = append_tokens_large(token_count, tokens, lemma_idx, new_tokens, new_pos_tags, pos_tags, pos_tag_idx, new_att_mat, att_mat, att_mat_idx)
                
            if lemma_idx < len(tokens):
                if tokens[lemma_idx] in special_tokens and token not in ['  ', ' ', '\n']:
                    new_tokens.append(token)
                    new_pos_tags.append('NO POS')

            token_count = 0
            pos_tag_idx += 1
            att_mat_idx += 1
        else:
            token_count += 1

    attention_for_topic_words = []
    attention_for_non_topic_words = []
    topic_words_orig = []
    topic_tokens_idxs = []
    for i, (token, pos_tag, orig_token) in enumerate(zip(new_tokens, new_pos_tags, tokens_no_lemma)):
        if token in topic_words and pos_tag in ['NN', 'NNP', 'NNPS', 'NNS']: # check if topic word
            attention_for_topic_words.append(att_mat[i])
            topic_words_orig.append(orig_token)
            topic_tokens_idxs.append(i)
        else:
            attention_for_non_topic_words.append(att_mat[i])

    return attention_for_topic_words, attention_for_non_topic_words, topic_words_orig, topic_tokens_idxs

def topic_attention_masked_for_plot(sentence, sentence_orig, att_mat, tokens, topic_words, nlp):
    '''
    Find attention distributed to topic word tokens for a masked text
    Inputs:
        sentence: input text (masked version)
        sentence_orig: input text (original version)
        att_mat: input attention matrix (averaged over heads) for masked text
        tokens: input tokens for input text
        topic_words: list of topic words corresponding with topic of the text
        nlp: Spacy's NLP
    Outputs:
        list of attention values for topic words, 
        list of topic words found in text
    '''
    tokens_with_space = preserve_spaces(tokens)
    original_data, tokens, pos_tags, _ = preprocess_text_for_recombination(sentence, nlp)
    _, tokens_orig, pos_tags_orig, orig_to_token = preprocess_text_for_recombination(sentence_orig, nlp)
    for token, pos_tag in zip(tokens_orig, pos_tags_orig): # link original words with masked words
        orig_to_token[token] = pos_tag
    orig_to_token['*'] = 'NO POS'
    dict = orig_to_token

    tokens_no_lemma = [token for token in tokens_with_space if token != ' ' and token != '\n']
    special_tokens = ['.', ',', '!', '?', ';', ':', '"', '(', ")", "'", ").", "'t", "...", "-", "''", "!!", " ", "\n"]

    new_tokens = []
    new_pos_tags = []
    new_att_mat = []
    token_count = 0
    lemma_idx = 0
    pos_tag_idx = 0
    att_mat_idx = 0
    for j, token in enumerate(tokens_with_space): # find topic word tokens
        if token in special_tokens or j == len(tokens_with_space) - 1:
            if lemma_idx < len(tokens):
                if tokens[lemma_idx] in special_tokens and token not in [' ', '\n']:
                    new_tokens.append(token)
                    new_pos_tags.append('NO POS')
            if (j == len(tokens_with_space) - 1 and token not in special_tokens) or token == "'t":
                token_count += 1
            
            if lemma_idx < len(tokens):
                new_tokens, lemma_idx, new_pos_tags, new_att_mat, att_mat_idx = append_tokens_large_masked(token_count, tokens, lemma_idx, new_tokens, new_pos_tags, dict, new_att_mat, att_mat, att_mat_idx)
                
            if lemma_idx < len(tokens):
                if tokens[lemma_idx] in special_tokens and token not in [' ', '\n']:
                    new_tokens.append(token)
                    new_pos_tags.append('NO POS')

            token_count = 0
            pos_tag_idx += 1
            att_mat_idx += 1
        else:
            token_count += 1

    attention_for_topic_words = 0
    topic_words_orig = []
    for i, (token, pos_tag, orig_token) in enumerate(zip(new_tokens, new_pos_tags, tokens_no_lemma)):
        if (token in topic_words and pos_tag in ['NN', 'NNP', 'NNPS', 'NNS']) or token == '*': # check if topic word or masked '*'
            attention_for_topic_words += att_mat[i]
            topic_words_orig.append(orig_token)

    return attention_for_topic_words, topic_words_orig

def pretty_tokens(tokens):
    '''
    Remove space indicators from the input tokens, and change new line token to \n
    Input:
        tokens: list of tokens
    Output:
        list of altered tokens
    '''
    modified_tokens = []
    for token in tokens:
        if token.startswith('Ġ'):  
            modified_tokens.append(token[1:])
        elif token == 'Ċ':
            modified_tokens.append('\n')
        else:
            modified_tokens.append(token)
    return modified_tokens

def preserve_spaces(tokens):
    '''
    Replace space and new line indicators with actual space and \n
    Input:
        tokens: list of tokens
    Output:
        list of altered tokens
    '''
    modified_tokens = []

    for token in tokens:
        if token.startswith('Ġ'):  
            modified_tokens.append(' ')
            modified_tokens.append(token[1:])
        elif token == 'Ċ':
            modified_tokens.append('\n')
        else:
            modified_tokens.append(token)
    return modified_tokens

def append_tokens_large(token_count, lemmatized_tokens, lemma_idx, new_tokens, new_pos_tags, pos_tags, pos_tag_idx, new_att_mat, att_mat, att_mat_idx):
    '''
    Helper function for topic_attention_for_plot, which appends tokens based on topic words
    Inputs:
        token_count: int indicating how many times word should be appended
        lemmatized_tokens: list of lemma words
        lemma_idx: int indicating which lemma word to append
        new_tokens: list that stores the new tokens
        new_pos_tags: list that stores the pos_tags
        pos_tags: list of original pos tags
        pos_tag_idx: int indicating which pos_tag to append
        new_att_mat: updated attention matrix
        att_mat: original attention matrix
        att_mat_idx: int indicating which attention value to append
    Outputs:
        new_tokens, lemma_idx, new_pos_tags, new_att_mat, att_mat_idx
    '''
    for j in range(token_count):
        new_tokens.append(lemmatized_tokens[lemma_idx])
        new_pos_tags.append(pos_tags[pos_tag_idx])
        new_att_mat.append(att_mat[att_mat_idx])
    lemma_idx += 1
    return new_tokens, lemma_idx, new_pos_tags, new_att_mat, att_mat_idx

def append_tokens_large_masked(token_count, lemmatized_tokens, lemma_idx, new_tokens, new_pos_tags, dict, new_att_mat, att_mat, att_mat_idx):
    '''
    Helper function for topic_attention_masked_for_plot, which appends tokens based on topic words
    Inputs:
        token_count: int indicating how many times word should be appended
        lemmatized_tokens: list of lemma words
        lemma_idx: int indicating which lemma word to append
        new_tokens: list that stores the new tokens
        new_pos_tags: list that stores the pos_tags
        dict: list of original pos tags
        new_att_mat: updated attention matrix
        att_mat: original attention matrix
        att_mat_idx: int indicating which attention value to append
    Outputs:
        new_tokens, lemma_idx, new_pos_tags, new_att_mat, att_mat_idx
    '''
    for j in range(token_count):
        new_tokens.append(lemmatized_tokens[lemma_idx])
        new_pos_tags.append(dict[lemmatized_tokens[lemma_idx]])
        new_att_mat.append(att_mat[att_mat_idx])
    lemma_idx += 1
    return new_tokens, lemma_idx, new_pos_tags, new_att_mat, att_mat_idx

def append_tokens(token_count, lemmatized_tokens, lemma_idx, new_tokens):
    '''
    Helper function for topic_attention, which appends tokens based on topic words
    Inputs:
        token_count: int indicating how many times word should be appended
        lemmatized_tokens: list of lemma words
        lemma_idx: int indicating which lemma word to append
        new_tokens: list that stores the new tokens
    Outputs:
        new_tokens, lemma_idx
    '''
    for j in range(token_count):
        new_tokens.append(lemmatized_tokens[lemma_idx])
    lemma_idx += 1
    return new_tokens, lemma_idx

def topic_attention(sentence, att_mat, tokens, topic_words, nlp, baseline=False):
    '''
    Find attention distributed to topic word tokens
    Inputs:
        sentence: input text
        att_mat: input attention matrix (averaged over heads)
        tokens: input tokens for input text
        topic_words: list of topic words corresponding with topic of the text
        nlp: Spacy's NLP
        baseline: whether to compute baseline values
    Outputs:
        attention value for topic words
    '''
    tokens_with_space = preserve_spaces(tokens)
    
    doc = nlp(sentence)
    lemmatized_tokens = [token.lemma_.lower() for token in doc]

    new_tokens = []
    token_count = 0
    lemma_idx = 0
    for j, token in enumerate(tokens_with_space):
        if token == '<s>':
            continue
        elif token == '</s>':
            if lemma_idx < len(lemmatized_tokens):
                new_tokens, lemma_idx = append_tokens(token_count, lemmatized_tokens, lemma_idx, new_tokens)
        elif token in ['.', ',', '!', '?', ';', ':', '"', '(', ")", "'", ")."] and (tokens_with_space[j+1] == ' ' or tokens_with_space[j+1] == '\n' or tokens_with_space[j+1] == '</s>'):
            if lemma_idx < len(lemmatized_tokens):
                new_tokens, lemma_idx = append_tokens(token_count, lemmatized_tokens, lemma_idx, new_tokens)
            new_tokens.append(token)
            token_count = 0
        elif token != ' ' and token != '\n':
            token_count += 1
        else:
            if lemma_idx < len(lemmatized_tokens):
                new_tokens, lemma_idx = append_tokens(token_count, lemmatized_tokens, lemma_idx, new_tokens)
            token_count = 0

    if not baseline:
        attention_for_topic_words = 0
        for i, token in enumerate(new_tokens):
            if token in topic_words:
                attention_for_topic_words += att_mat[i]
    else:
        topic_words_number = 0
        for token in new_tokens:
            if token in topic_words:
                topic_words_number += 1

    return attention_for_topic_words

def topic_attention_over_layers(sentence, att_mat, tokens, topic_words, nlp):
    '''
    Inputs:
        sentence: input text
        att_mat: attention matrix for input text
        tokens: tokenized input text
        topic_words: list of topic words corresponding with topic of input text
        nlp: Spacy's NLP
    Output:
        ratio of attention per layer
    '''
    layers_ratio = []
    layers = list(range(att_mat.shape[0]))
    for layer in layers:
        new_mat = att_mat[layer].mean(dim=0)
        attention_topics = topic_attention(sentence, new_mat, tokens, topic_words, nlp, baseline=False)
        layers_ratio.append(attention_topics / new_mat.sum())
    return layers_ratio

def topic_attention_ratio_line(layers_ratio1_raw, layers_ratio2_raw, layers_ratio1_rollout, layers_ratio2_rollout, layers_ratio1_vz, layers_ratio2_vz, save_name):
    '''
    Plot the topic attention ratio over the layers
    Inputs:
        layers_ratio1_raw: layer ratio for raw attention text 1
        layers_ratio2_raw: layer ratio for raw attention text 2 
        layers_ratio1_rollout: layer ratio for attention rollout text 1 
        layers_ratio2_rollout: layer ratio for attention rollout text 2 
        layers_ratio1_vz: layer ratio for value zeroing text 1 
        layers_ratio2_vz: layer ratio for raw attention text 2 
        save_name: path/name to save the plot
    '''
    layers = range(1, len(layers_ratio1_raw) + 1)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 3), sharex=True)
    custom_cmap = LinearSegmentedColormap.from_list("strong_puor", ["#5E00B5", "#FF8500"])

    line1, = axes[0].plot(layers, layers_ratio1_raw, marker='o', color=custom_cmap(0.05), linewidth=2, label="Raw Attention")
    line2, = axes[0].plot(layers, layers_ratio1_rollout, marker='o', color=custom_cmap(0.5), linewidth=2, label="Attention Rollout")
    line3, = axes[0].plot(layers, layers_ratio1_vz, marker='o', color=custom_cmap(0.95), linewidth=2, label="Value Zeroing+Rollout")

    axes[0].set_title("Text 1", fontsize=16)
    axes[0].set_ylabel("Topic Attention Ratio", fontsize=14)
    axes[0].set_xlabel("Layer", fontsize=14)
    axes[0].tick_params(axis='x', labelsize=12)
    axes[0].tick_params(axis='y', labelsize=12)

    axes[1].plot(layers, layers_ratio2_raw, marker='o', color=custom_cmap(0.05), linewidth=2)
    axes[1].plot(layers, layers_ratio2_rollout, marker='o', color=custom_cmap(0.5), linewidth=2)
    axes[1].plot(layers, layers_ratio2_vz, marker='o', color=custom_cmap(0.95), linewidth=2)

    axes[1].set_title("Text 2", fontsize=16)
    axes[1].set_xlabel("Layer", fontsize=14)
    axes[1].set_xticks(layers)
    axes[1].tick_params(axis='x', labelsize=12)
    axes[1].tick_params(axis='y', labelsize=12)

    fig.legend(handles=[line1, line2, line3],
               loc="lower center",
               ncol=3,
               bbox_to_anchor=(0.5, -0.05),
               fontsize=14)

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(f"explainableAV/attention_images/{save_name}", dpi=300, bbox_inches='tight')
    plt.close()


def sanitize_for_json(obj):
    """
    Convert objects to JSON-serializable types to ensure that it can be saved as json
    """
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, (np.ndarray, torch.Tensor)):
        return obj.tolist()
    elif isinstance(obj, (str, bool)) or obj is None:
        return obj
    else:
        return str(obj)

def dictify(d):
    '''
    Ensure that d is a dictionary so it can be easily saved
    '''
    if isinstance(d, defaultdict):
        d = {k: dictify(v) for k, v in d.items()}
    return d

def get_masked_sentence(masks, tokens, mask_token, model, space_marker="Ġ"):
    '''
    Mask sentence for faithfulness evaluation
    Inputs:
        masks: which places to mask
        tokens: tokenized sentence
        mask_token: mask symbol
        model: Transformer model (AutoModel)
        space_marker: symbol to indicate spaces
    Output:
        masked sentence
    '''
    masked_tokens = []
    for i, token in enumerate(tokens):
        if i in masks:
            if token.startswith(space_marker):
                masked_tokens.append(space_marker + mask_token)
            else:
                masked_tokens.append(mask_token)
        else:
            masked_tokens.append(token)
    sentence = model.tokenizer.convert_tokens_to_string(masked_tokens)

    return sentence

def get_confidence(masks1, tokens1, masks2, tokens2, mask_token, model, threshold, label):
    '''
    Compute confidence score for faithfulness evaluation based on masked texts
    Inputs:
        masks1: how to mask first text
        tokens1: tokenized text 1
        masks2: how to mask second text
        tokens2: tokenized text 2
        mask_token: mask symbol
        model: Transformer model (AutoModel)
        threshold: model threshold for classification
        label: original prediction label of model
    Output:
        confidence score relative to original prediction
    '''
    sentence1 = get_masked_sentence(masks1, tokens1, mask_token, model)
    sentence2 = get_masked_sentence(masks2, tokens2, mask_token, model)
    similarity = inference(sentence1, sentence2, model)
    conf_score = confidence_V2(similarity, threshold, label)
    return conf_score

def faithfulness(attention_matrix1, attention_matrix2, tokens1, tokens2, model, threshold, results):
    '''
    Compute faithfulness scores (sufficiency, insufficiency, comprehensiveness, incomprehensiveness)
    Inputs:
        attention_matrix1: attention scores of first text
        attention_matrix2: attention scores of second text
        tokens1: tokenized input text 1
        tokens2: tokenized input text 2
        model: Transformer model (AutoModel)
        threshold: model threshold for classification
        results: dictionary to store faithfulness scores
    Output:
        results
    '''
    attentions1 = attention_matrix1[-1].mean(dim=0)[1:-1] # skip special tokens <s> and </s>
    attentions2 = attention_matrix2[-1].mean(dim=0)[1:-1]
    attentions1_idxs = np.argsort(attentions1) # ascending order
    attentions2_idxs = np.argsort(attentions2)
    token_length1 = len(tokens1)
    token_length2 = len(tokens2)
    ks = [0.01, 0.1, 0.25]
    mask_token = model.tokenizer.mask_token
    for k in ks:
        masks_num1 = min(round(token_length1 / 2), max(1, round(k * token_length1)))
        masks_num2 = min(round(token_length2 / 2), max(1, round(k * token_length2)))
        sentence1 = model.tokenizer.convert_tokens_to_string(tokens1)
        sentence2 = model.tokenizer.convert_tokens_to_string(tokens2)
        similarity = inference(sentence1, sentence2, model)
        orig_conf = confidence(similarity, threshold)
        if orig_conf < 0.5:
            label = 0
        else:
            label = 1
        
        # comprehensiveness
        comp_conf = get_confidence(attentions1_idxs[-masks_num1:], tokens1, attentions2_idxs[-masks_num2:], tokens2, mask_token, model, threshold, label)
        comp = orig_conf - comp_conf
        results['comp'][k].append(comp)
        
        # incomprehensiveness
        incomp_conf = get_confidence(attentions1_idxs[:masks_num1], tokens1, attentions2_idxs[:masks_num2], tokens2, mask_token, model, threshold, label)
        incomp = orig_conf - incomp_conf
        results['incomp'][k].append(incomp)

        # sufficiency
        suff_conf = get_confidence(attentions1_idxs[:-masks_num1], tokens1, attentions2_idxs[:-masks_num2], tokens2, mask_token, model, threshold, label)
        suff = orig_conf - suff_conf
        results['suff'][k].append(suff)

        # insufficiency
        insuff_conf = get_confidence(attentions1_idxs[masks_num1:], tokens1, attentions2_idxs[masks_num2:], tokens2, mask_token, model, threshold, label)
        insuff = orig_conf - insuff_conf
        results['insuff'][k].append(insuff)

    return results

def top_k_topic_words(results, attention_matrix, tokens, sentence, topic_words, model, nlp, k=0.25):
    '''
    Compute topic coverage and relative topic-attention ratio
    Inputs:
        results: dictionary to store results
        attention_matrix: attention matrix for input text
        tokens: tokenized input text
        sentence: original input text
        topic_words: list of topic words corresponding with topic of input text
        model: Transformer model (AutoModel)
        nlp: Spacy's NLP
        k: k%
    Output:
        results
    '''
    attention_matrix_for_topic_idxs = attention_matrix.mean(dim=0)[1:-1]
    _, _, topic_word_idxs = topic_attention_for_plot(sentence, attention_matrix_for_topic_idxs, tokens, topic_words, model, nlp)
    token_length = len(tokens)
   
    num_tokens_top_k = max(1, round(k * token_length))
    layer_list_topk = []
    layer_list_ratio = []
    for layer in attention_matrix:
        attention_layer = layer.mean(dim=0)[1:-1] # average over queries and remove special tokens
        attentions_idxs = np.argsort(attention_layer) # ascending order

        # compute top-k metric
        topic_words_top_k = sum([1 for idx in attentions_idxs[-num_tokens_top_k:] if idx in topic_word_idxs])
        if len(topic_word_idxs) <= num_tokens_top_k:
            top_k = topic_words_top_k / max(1, len(topic_word_idxs))
        else:
            top_k = topic_words_top_k / num_tokens_top_k
        layer_list_topk.append(top_k)

         # compute topic-attention ratio
        non_topic_word_attention = [attention for i, attention in enumerate(attention_layer) if i not in topic_word_idxs]
        non_topic_word_average = sum(non_topic_word_attention) / max(1, len(non_topic_word_attention))
        topic_word_attention = [attention for i, attention in enumerate(attention_layer) if i in topic_word_idxs]
        topic_word_average = sum(topic_word_attention) / max(1, len(topic_word_attention))
        difference = topic_word_average - non_topic_word_average
        ratio = difference / attention_layer.sum(dim=0)
        layer_list_ratio.append(ratio.item())
        
    results['top-k'].append(layer_list_topk)
    results['ratio'].append(layer_list_ratio)
    return results


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='explainableAV/Amazon/test_set_15000x4.json')
    parser.add_argument('--model_name', type=str, default="LUAR", help="Model to use, one of: 'LUAR', 'ModernBERT', 'StyleDistance'")
    parser.add_argument('--seed', default=0, help='Set seed')
    parser.add_argument('--attention_type', type=str, default='raw', help="Type of attention to apply, choose from 'raw', 'rollout', 'value_zeroing', 'value_zeroing_rollout', 'globenc'")
    parser.add_argument('--pair_type', type=str, default='SS')
    parser.add_argument('--plot_type', type=str, default='over_tokens', help="Choose from: 'over_tokens', 'over_layers', 'text_plot', 'per_layer_over_tokens', 'topic_attention_layers'")
    parser.add_argument('--topic_related_path', type=str, default="explainableAV/extract_topic/amazon_topic_related_8400_filtered.json")
    parser.add_argument('--datapoint', type=int, help='index for topic attention ratio plot')
    parser.add_argument('--faithfulness', action='store_true')
    parser.add_argument('--topic_words_attention', action='store_true')
    parser.add_argument('--visualize_masked', action='store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = argument_parser()
    random.seed(args.seed)

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
    if args.visualize_masked:
        masked_data = load_dataset('explainableAV/attention_most_influence_asterisk.json')
        combined = list(zip(data, masked_data))
        random.shuffle(combined)
        data, masked_data = zip(*combined)
    else:       
        random.shuffle(data)
    topic_related = load_dataset(args.topic_related_path)
    nlp = spacy.load('en_core_web_sm')

    # check attention type
    if args.attention_type not in ["raw", "rollout", "value_zeroing", "value_zeroing_rollout", "globenc"]:
        print("Attention type not recognised, choose one of: 'raw', 'rollout', 'value_zeroing', 'value_zeroing_rollout', 'globenc'")
        exit()

    # check plot type
    if args.plot_type not in ['over_tokens', 'over_layers', 'text_plot', 'per_layer_over_tokens', 'one_token', 'topic_attention_layers']:
        print("Plot type not recognised, choose one of: 'over_tokens', 'over_layers', 'text_plot', 'per_layer_over_tokens', 'one_token', 'topic_attention_layers'")
        exit()

    # determine how to aggregated
    if args.plot_type == 'over_layers':
        aggregate_tokens = True
    else:
        aggregate_tokens= False

    # replace layers for value zeroing
    if args.attention_type == 'value_zeroing' or args.attention_type == 'value_zeroing_rollout' or args.spearman:
        if args.model_name == 'LUAR' or args.model_name == 'StyleDistance':
            transformer_model_vz = copy.deepcopy(transformer_model)
            original_layers = list(transformer_model.encoder.layer)
            transformer_model_vz.encoder.layer = nn.ModuleList([RobertaLayer(transformer_model_vz.config) for _ in range(len(transformer_model_vz.encoder.layer))])
            for l, layer_module in enumerate(transformer_model_vz.encoder.layer):
                layer_module.load_state_dict(original_layers[l].state_dict())
            transformer_model_vz.eval()
        elif args.model_name == 'ModernBERT':
            transformer_model_vz = copy.deepcopy(transformer_model)
            original_layers = list(transformer_model.layers)
            config = ModernBertConfig.from_pretrained(model_name) 
            transformer_model_vz.layers = nn.ModuleList([ModernBertEncoderLayer(config, layer_id=idx) for idx in range(len(transformer_model_vz.layers))])
            for l, layer_module in enumerate(transformer_model_vz.layers):
                layer_module.load_state_dict(original_layers[l].state_dict(), strict=False)
            transformer_model_vz.eval()

    # replace model when computing globenc
    if args.attention_type == 'globenc':
        if args.model_name == 'LUAR' or args.model_name == 'StyleDistance':
            transformer_model_ge = RobertaModel.from_pretrained(model_name, config=transformer_model.config)
        elif args.model_name == 'ModernBERT':
            config = ModernBertConfig.from_pretrained(model_name)
            transformer_model_ge = ModernBertModel(config)
            weights = transformer_model.state_dict()
            transformer_model_ge.load_state_dict(weights, strict=False)

    # select datapoint for which to plot
    if args.plot_type == 'topic_attention_layers' or args.plot_type == 'text_plot':
        data = data[args.datapoint:args.datapoint+1]

    # initialize results dictionary for faithfulness evaluation
    if args.faithfulness:
        results = defaultdict(lambda: defaultdict(list))

    # initialize results dictionary for topic coverage and relative topic-attention ratio
    if args.topic_words_attention:
        results = defaultdict(list)

    # compute attention for all datapoints
    for line in tqdm(data):
        sentence1 = line["Pair"][0]
        sentence2 = line["Pair"][1]
        topic1 = line["Topics"][0]
        topic2 = line["Topics"][1]

        if args.visualize_masked:
            sentence1_orig = sentence1
            sentence2_orig = sentence2
            sentence1 = masked_data[args.datapoint]["Pair"][0]
            sentence2 = masked_data[args.datapoint]["Pair"][1]

        tokens1, attentions1, embedding1 = inference_with_attention(sentence1, model, transformer_model, args.model_name)
        tokens2, attentions2, embedding2 = inference_with_attention(sentence2, model, transformer_model, args.model_name)   
        similarity = model.similarity(embedding1, embedding2).item()
        confidence_score = confidence(similarity, threshold)
        
        if args.plot_type == 'topic_attention_layers':
            save_name = args.pair_type + '_' + args.model_name + '_' + str(args.datapoint) + '_attention_' + args.plot_type + '.pdf'
        else:
            if args.plot_type == 'text_plot':
                if topic1 == topic2 and line["Label"] == 1:
                    pair_type = 'SS'
                elif topic1 != topic2 and line["Label"] == 1:
                    pair_type = 'SD'
                elif topic1 == topic2 and line["Label"] == 0:
                    pair_type = 'DS'
                elif topic1 != topic2 and line["Label"] == 0:
                    pair_type = 'DD'
                print(pair_type, confidence_score)
                if args.visualize_masked:
                    save_name = pair_type + '_' + args.model_name + '_' + str(args.datapoint) + '_' + 'asterisk' + '_' + args.plot_type + '.pdf'
                else:
                    save_name = pair_type + '_' + args.model_name + '_' + str(args.datapoint) + '_' + args.plot_type + '.pdf'
            else:
                save_name = args.attention_type + '_attention_' + args.plot_type + '.pdf'

            if args.attention_type == 'raw':
                tokens1, attentions1, embedding1 = inference_with_attention(sentence1, model, transformer_model, args.model_name)
                tokens2, attentions2, embedding2 = inference_with_attention(sentence2, model, transformer_model, args.model_name)
                attention_matrix1 = raw_attention(attentions1)
                attention_matrix2 = raw_attention(attentions2)
            elif args.attention_type == 'rollout':
                tokens1, attentions1, embedding1 = inference_with_attention(sentence1, model, transformer_model, args.model_name)
                tokens2, attentions2, embedding2 = inference_with_attention(sentence2, model, transformer_model, args.model_name)
                attention_matrix1 = attention_rollout(attentions1)
                attention_matrix2 = attention_rollout(attentions2)
            elif args.attention_type == 'value_zeroing':
                attention_matrix1, _, tokens1 = value_zeroing(sentence1, model, transformer_model_vz, args.model_name)
                attention_matrix2, _, tokens2 = value_zeroing(sentence2, model, transformer_model_vz, args.model_name)
            elif args.attention_type == 'value_zeroing_rollout':
                _, attention_matrix1, tokens1 = value_zeroing(sentence1, model, transformer_model_vz, args.model_name)
                _, attention_matrix2, tokens2 = value_zeroing(sentence2, model, transformer_model_vz, args.model_name)
            elif args.attention_type == 'globenc':
                _, attention_matrix1, tokens1 = globenc(sentence1, model, transformer_model_ge, args.model_name)
                _, attention_matrix2, tokens2 = globenc(sentence2, model, transformer_model_ge, args.model_name)

        if args.faithfulness:
            tokens1, attentions1, embedding1 = inference_with_attention(sentence1, model, transformer_model, args.model_name)
            tokens2, attentions2, embedding2 = inference_with_attention(sentence2, model, transformer_model, args.model_name)   
            results = faithfulness(attention_matrix1, attention_matrix2, tokens1[1:-1], tokens2[1:-1], model, threshold, results)

        if args.topic_words_attention:
            tokens1, attentions1, embedding1 = inference_with_attention(sentence1, model, transformer_model, args.model_name)
            tokens2, attentions2, embedding2 = inference_with_attention(sentence2, model, transformer_model, args.model_name)
            results = top_k_topic_words(results, attention_matrix1, tokens1[1:-1], sentence1, topic_related[topic1], args.model_name, nlp)
            results = top_k_topic_words(results, attention_matrix2, tokens2[1:-1], sentence2, topic_related[topic2], args.model_name, nlp)

        if not args.faithfulness and not args.topic_words_attention:
            if args.plot_type == 'text_plot':
                attention_matrix1 = aggregate_attention(attention_matrix1, aggregate_tokens=aggregate_tokens, per_layer=False)
                attention_matrix2 = aggregate_attention(attention_matrix2, aggregate_tokens=aggregate_tokens, per_layer=False)
                if args.visualize_masked:
                    custom_masked_text_plot(attention_matrix1, attention_matrix2, tokens1[1:-1], tokens2[1:-1], sentence1, sentence2, sentence1_orig, sentence2_orig, topic_related[topic1], topic_related[topic2], pair_type, save_name)
                else:
                    custom_text_plot(attention_matrix1, attention_matrix2, tokens1[1:-1], tokens2[1:-1], sentence1, sentence2, topic_related[topic1], topic_related[topic2], pair_type, args.model_name, save_name)
            elif args.plot_type == 'per_layer_over_tokens':
                attention_matrix1 = aggregate_attention(attention_matrix1, aggregate_tokens=False, per_layer=True)
                attention_matrix2 = aggregate_attention(attention_matrix2, aggregate_tokens=False, per_layer=True)
                attention_over_tokens_per_layer(attention_matrix1, attention_matrix2, tokens1, tokens2, save_name)
            elif args.plot_type == 'topic_attention_layers':
                tokens1, attentions1, embedding1 = inference_with_attention(sentence1, model, transformer_model, args.model_name)
                tokens2, attentions2, embedding2 = inference_with_attention(sentence2, model, transformer_model, args.model_name)

                attention_matrix1_raw = raw_attention(attentions1)
                attention_matrix2_raw = raw_attention(attentions2)
                attention_matrix1_rollout = attention_rollout(attentions1)
                attention_matrix2_rollout = attention_rollout(attentions2)
                _, attention_matrix1_vz, tokens1 = value_zeroing(sentence1, model, transformer_model_vz, args.model_name)
                _, attention_matrix2_vz, tokens2 = value_zeroing(sentence2, model, transformer_model_vz, args.model_name)

                layers_ratio1_raw = topic_attention_over_layers(sentence1, attention_matrix1_raw, tokens1, topic_related[topic1], nlp)
                layers_ratio2_raw = topic_attention_over_layers(sentence2, attention_matrix2_raw, tokens2, topic_related[topic2], nlp)
                layers_ratio1_rollout = topic_attention_over_layers(sentence1, attention_matrix1_rollout, tokens1, topic_related[topic1], nlp)
                layers_ratio2_rollout = topic_attention_over_layers(sentence2, attention_matrix2_rollout, tokens2, topic_related[topic2], nlp)
                layers_ratio1_vz = topic_attention_over_layers(sentence1, attention_matrix1_vz, tokens1, topic_related[topic1], nlp)
                layers_ratio2_vz = topic_attention_over_layers(sentence2, attention_matrix2_vz, tokens2, topic_related[topic2], nlp)
                topic_attention_ratio_line(layers_ratio1_raw, layers_ratio2_raw, layers_ratio1_rollout, layers_ratio2_rollout, layers_ratio1_vz, layers_ratio2_vz, save_name)
            else:
                attention_matrix1 = aggregate_attention(attention_matrix1, aggregate_tokens=aggregate_tokens, per_layer=False)
                attention_matrix2 = aggregate_attention(attention_matrix2, aggregate_tokens=aggregate_tokens, per_layer=False)
                attention_heatmap(attention_matrix1, tokens1, attention_matrix2, tokens2, confidence_score, save_name, aggregate_tokens=aggregate_tokens)

    if args.topic_words_attention: # save topic coverage and relative topic-attention ratio
        metrics_file = f"explainableAV/attention_top_{args.model_name}_{args.attention_type}_non_topic.json"
        if os.path.exists(metrics_file): # load or initialize file
            metrics = load_dataset(metrics_file)
        else:
            metrics = {}

        metrics['top-k'] = results['top-k']
        metrics['ratio'] = results['ratio']
        create_dataset(metrics_file, dictify(metrics))

    if args.faithfulness: # save faithfulness evaluation
        metrics_file = "explainableAV/attention/attention_faithfulness.json"

        if os.path.exists(metrics_file):
            metrics_dict = load_dataset(metrics_file)
            metrics = {}
            for attention_type in metrics_dict:
                metrics[attention_type] = {}
                for model_name in metrics_dict[attention_type]:
                    metrics[attention_type][model_name] = {}
                    for metric_type in metrics_dict[attention_type][model_name]:
                        metrics[attention_type][model_name][metric_type] = {}
                        for k, values in metrics_dict[attention_type][model_name][metric_type].items():
                            metrics[attention_type][model_name][metric_type][k] = values if isinstance(values, list) else []
        else:
            metrics = {}

        if args.attention_type not in metrics:
            metrics[args.attention_type] = {}
        if args.model_name not in metrics[args.attention_type]:
            metrics[args.attention_type][args.model_name] = {'comp': {}, 'incomp': {}, 'suff': {}, 'insuff': {}}
            
        for metric_type in ['comp', 'incomp', 'suff', 'insuff']:
            if metric_type not in metrics[args.attention_type][args.model_name]:
                metrics[args.attention_type][args.model_name][metric_type] = {}
            
            for k in [0.01, 0.1, 0.25]:
                k_str = str(k)
                if k_str not in metrics[args.attention_type][args.model_name][metric_type]:
                    metrics[args.attention_type][args.model_name][metric_type][k_str] = []

        for k in [0.01, 0.1, 0.25]:
            k_str = str(k)
            metrics[args.attention_type][args.model_name]['comp'][k_str].append(np.mean(results['comp'][k]))
            metrics[args.attention_type][args.model_name]['incomp'][k_str].append(np.mean(results['incomp'][k]))
            metrics[args.attention_type][args.model_name]['suff'][k_str].append(np.mean(results['suff'][k]))
            metrics[args.attention_type][args.model_name]['insuff'][k_str].append(np.mean(results['insuff'][k]))

        create_dataset(metrics_file, dictify(metrics))
