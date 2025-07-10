from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer, AutoConfig
from explainableAV.utils.utils import load_dataset, create_dataset
import torch
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import os
import random
from tqdm import tqdm
from collections import defaultdict
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import copy

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def dictify(d):
    '''
    Ensure that d is a dictionary so it can be easily saved
    '''
    if isinstance(d, defaultdict):
        d = {k: dictify(v) for k, v in d.items()}
    return d

def balance_dataset(data, samples_per_class):
    '''
    Create a balanced dataset in terms of topic based on smallest topic class
    Inputs:
        data: text pairs data
        samples_per_class: number of texts per topic
    Output:
        balanced data
    '''
    class_buckets = defaultdict(list)
    max_samples_per_class = samples_per_class

    # group data by topic
    for line in data:
        class_buckets[line["Topics"][0]].append(line["Pair"][0])
        class_buckets[line["Topics"][1]].append(line["Pair"][1])

    balanced_data = defaultdict(list)
    max_samples_per_class = samples_per_class

    # sample uniformly from each class
    for topic, texts in class_buckets.items():
        if len(texts) < max_samples_per_class:
            print(f"Warning: not enough samples for class '{topic}' ({len(texts)} available)")
            sampled = texts  # use all
            max_samples_per_class = len(texts)
        else:
            sampled = random.sample(texts, max_samples_per_class)
        balanced_data[topic].extend(sampled)
    if samples_per_class != max_samples_per_class:
        for topic, texts in balanced_data.items():
            balanced_data[topic] = balanced_data[topic][:max_samples_per_class]

    print(f"Total balanced samples: {len(balanced_data[next(iter(balanced_data))])}")
    return balanced_data

def downsampling(data, max_samples_per_class):
    '''
    Create a balanced dataset in terms of topic based on fixed number
    Inputs:
        data: text pairs data
        samples_per_class: number of texts per topic
    Output:
        balanced data
    '''
    class_buckets = defaultdict(list)

    # group data by topic
    for line in data:
        class_buckets[line["Topics"][0]].append(line["Pair"][0])
        class_buckets[line["Topics"][1]].append(line["Pair"][1])

    balanced_data = defaultdict(list)

    # sample uniformly from each class
    for topic, texts in class_buckets.items():
        if len(texts) < max_samples_per_class:
            print(f"Warning: not enough samples for class '{topic}' ({len(texts)} available)")
            sampled = texts  # use all
        else:
            sampled = random.sample(texts, max_samples_per_class)
        balanced_data[topic].extend(sampled)
    for topic, texts in balanced_data.items():
        balanced_data[topic] = balanced_data[topic][:max_samples_per_class]

    print(f"Total balanced samples: {len(balanced_data[next(iter(balanced_data))])}")
    return balanced_data
        

def get_hidden_states(sentence, model, transformer_model, model_name):
    '''
    Obtain hidden representations from the model
    Inputs:
        sentence: input text
        model: SentenceTransformer model
        transformer_model: Transformer model (AutoModel)
        model_name: name of SentenceTransformer
    Output:
        hidden representations over layers
    '''
    if model_name == 'LUAR':
        tokenized_input = model.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)
    elif model_name == 'StyleDistance':
        tokenized_input = model.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
    elif model_name == 'ModernBERT':
        tokenized_input = model.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=8192)
    tokens = model.tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"][0])
    with torch.no_grad():
        output = transformer_model(input_ids=tokenized_input['input_ids'], attention_mask=tokenized_input['attention_mask'])

    hidden_states = output.hidden_states
    return hidden_states

def collect_probing_data(data, model, transformer_model, model_name):
    '''
    Collect hidden representations for given data
    Inputs:
        data: input texts
        model: SentenceTransformer model
        transformer_model: Transformer model (AutoModel)
        model_name: name of SentenceTransformer
    Output:
        dataset for probing
    '''
    probe_data = []
    for topic, texts in data.items():
        for text in tqdm(texts):
            hidden_states = get_hidden_states(text, model, transformer_model, model_name)
            entry = {
            "sentence": text,
            "label": topic,
            }

            for i in range(1, len(hidden_states)):
                entry[f"hidden_state_{i}"] = hidden_states[i].squeeze().detach().cpu().numpy()

            probe_data.append(entry)
    return probe_data

def collect_probing_data_random(data, model, transformer_model, model_name):
    '''
    Collect hidden representations randomly
    Inputs:
        data: input texts
        model: SentenceTransformer model
        transformer_model: Transformer model (AutoModel)
        model_name: name of SentenceTransformer
    Output:
        dataset for probing
    '''
    probe_data = []
    hidden_state_shapes = None
    for topic, texts in data.items():
        for text in tqdm(texts):
            if hidden_state_shapes is None:
                hidden_states = get_hidden_states(text, model, transformer_model, model_name)
                hidden_state_shapes = [hs.squeeze().detach().cpu().numpy().shape for hs in hidden_states]
            entry = {
            "sentence": text,
            "label": topic,
            }

            for i in range(1, len(hidden_state_shapes)):
                entry[f"hidden_state_{i}"] = np.random.normal(loc=0.0, scale=1.0, size=hidden_state_shapes[i]).astype(np.float32)
            probe_data.append(entry)
    return probe_data

class MLPModel(nn.Module):
    '''
    Simple MLP model
    '''
    def __init__(self, input_dim, output_dim):
        super(MLPModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.model(x)

def train_logistic_regression(X_train, y_train, X_val, y_val, input_dim, output_dim, patience=3, num_epochs=1000, lr=1e-3):
    '''
    Train MLP model
    Inputs:
        X_train: train probing data
        y_train: train probing labels
        X_val: validation probing data
        y_val: validation probing labels
        input_dim: input dimensionality of the model
        output_dim: number of classes
        patience: patience
        num_epochs: maximum number of epochs
        lr: learning rate
    Outputs:
        trained model
        label encoder
        losses for training
        losses for validation
    '''
    label_encoder = LabelEncoder() # encode the labels
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_encoded, dtype=torch.long)

    model = MLPModel(input_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs): # train
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()

        print(f"Epoch {epoch+1}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")
        train_losses.append(loss.item())
        val_losses.append(val_loss)
        
        delta = 1e-4

        if val_loss < best_val_loss - delta:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    model.load_state_dict(best_model_state)
    return model, label_encoder, train_losses, val_losses


def perform_logistic_regression_cv(probe_data, layer_num, patience=3):
    '''
    Probe with cross-validation
    Inputs:
        probe_data: data for probing
        layer_num: which layer to probe
        patience: patience

    '''
    X = []
    y = []
    for entry in probe_data:
        hidden_state = entry[f"hidden_state_{layer_num}"]
        X.append(hidden_state.mean(axis=0))  # create sentence-level hidden state
        y.append(entry["label"])
    X = np.vstack(X)
    y = np.array(y)

    skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    all_reports_train = defaultdict(list)
    all_reports_test = defaultdict(list)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"Fold: {fold}")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.1, stratify=y_train, random_state=0
        )

        # normalize
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        input_dim = X_train.shape[1]
        output_dim = len(np.unique(y))

        # train
        model, label_encoder, _, _ = train_logistic_regression(
            X_train, y_train, X_val, y_val, input_dim, output_dim, patience
        )

        # evaluate
        model.eval()
        with torch.no_grad():
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            y_train_pred_tensor = model(X_train_tensor)
            y_train_pred = torch.argmax(y_train_pred_tensor, dim=1).numpy()
            y_train_pred_labels = label_encoder.inverse_transform(y_train_pred)

        print("Training Classification Report:")
        print(classification_report(y_train, y_train_pred_labels, digits=4))

        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            y_test_pred_tensor = model(X_test_tensor)
            y_test_pred = torch.argmax(y_test_pred_tensor, dim=1).numpy()
            y_test_pred_labels = label_encoder.inverse_transform(y_test_pred)

        print("Test Classification Report:")
        print(classification_report(y_test, y_test_pred_labels, digits=4))

        report = classification_report(y_train, y_train_pred_labels, digits=4, output_dict=True, zero_division=0)
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                for metric_name, score in metrics.items():
                    all_reports_train[f"{label}_{metric_name}"].append(score)

        report = classification_report(y_test, y_test_pred_labels, digits=4, output_dict=True, zero_division=0)
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                for metric_name, score in metrics.items():
                    all_reports_test[f"{label}_{metric_name}"].append(score)

    print("Average classification over folds train")
    for metric, scores in all_reports_train.items():
        print(f"{metric}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

    print("Average classification over folds test")
    for metric, scores in all_reports_test.items():
        print(f"{metric}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

    summary = {}
    for combined_key, values in all_reports_test.items():
        label, metric = combined_key.rsplit("_", 1)
        if label not in summary:
            summary[label] = {}
        mean = np.mean(values)
        std = np.std(values)
        summary[label][metric] = f"{mean:.4f} ± {std:.4f}"

    df = pd.DataFrame.from_dict(summary, orient='index')
    df = df[["precision", "recall", "f1-score", "support"]] 

    print("Average classification over folds (test):\n")
    print(df.to_string())

def perform_logistic_regression(probe_data, layer_num, model_name, random_repr=False, random_weights=False, pretrained=False, masked_data=False, new_baseline=False, patience=3):
     '''
    Probe
    Inputs:
        probe_data: data for probing
        layer_num: which layer to probe
        patience: patience

    '''
    X = []
    y = []
    for entry in probe_data:
        hidden_state = entry[f"hidden_state_{layer_num}"]
        X.append(hidden_state.mean(axis=0))  # create sentence-level hidden state
        y.append(entry["label"])
    X = np.vstack(X)
    y = np.array(y)
    print(X.shape)
    print(y.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=0
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, stratify=y_train, random_state=0
    )

    # normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    input_dim = X_train.shape[1]
    output_dim = len(np.unique(y))

    # train
    model, label_encoder, train_losses, val_losses = train_logistic_regression(
        X_train, y_train, X_val, y_val, input_dim, output_dim, patience
    )

    # evaluate
    model.eval()
    with torch.no_grad():
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_pred_tensor = model(X_train_tensor)
        y_train_pred = torch.argmax(y_train_pred_tensor, dim=1).numpy()
        y_train_pred_labels = label_encoder.inverse_transform(y_train_pred)

    print("Training Classification Report:")
    print(classification_report(y_train, y_train_pred_labels, digits=4))
    train_report = classification_report(y_train, y_train_pred_labels, digits=4, output_dict=True)

    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_pred_tensor = model(X_test_tensor)
        y_test_pred = torch.argmax(y_test_pred_tensor, dim=1).numpy()
        y_test_pred_labels = label_encoder.inverse_transform(y_test_pred)

    print("Test Classification Report:")
    print(classification_report(y_test, y_test_pred_labels, digits=4))

    # for saving
    report = classification_report(y_test, y_test_pred_labels, digits=4, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    class_metrics_df = df_report.loc[~df_report.index.isin(['accuracy', 'macro avg', 'weighted avg'])]
    precisions = class_metrics_df['precision'].tolist()
    recalls = class_metrics_df['recall'].tolist()
    f1_scores = class_metrics_df['f1-score'].tolist()

    # save metrics
    metric_file = f"explainableAV/results/probing_metrics/probing_metrics_{model_name}.json"
    if os.path.exists(metric_file):
        metrics = load_dataset(metric_file)
    else:
        metrics = {}
    metrics = defaultdict(lambda: defaultdict(dict), metrics)
    
    if random_repr:
        metrics['Random_representations'][layer_num] = report['accuracy']
    elif random_weights:
        metrics['Random weights'][layer_num] = report['accuracy']
    elif pretrained:
        metrics['Pre-trained Accuracy'][layer_num] = report['accuracy']
    elif masked_data:
        metrics['Precision_masked'][layer_num] = precisions
        metrics['Recall_masked'][layer_num] = recalls
        metrics['F1-scores_masked'][layer_num] = f1_scores
        metrics['Test_accuracy_masked'][layer_num] = report['accuracy']
    elif new_baseline:
        metrics['Precision_masked_baseline'][layer_num] = precisions
        metrics['Recall_masked_baseline'][layer_num] = recalls
        metrics['F1-scores_masked_baseline'][layer_num] = f1_scores
        metrics['Test_accuracy_masked_baseline'][layer_num] = report['accuracy']
    else:
        metrics['Precision'][layer_num] = precisions
        metrics['Recall'][layer_num] = recalls
        metrics['F1-scores'][layer_num] = f1_scores
        metrics['Test_accuracy'][layer_num] = report['accuracy']
        metrics['Train_accuracy'][layer_num] = train_report['accuracy']
        metrics['Label_names'] = sorted(set(y_test)) 
    create_dataset(metric_file, dictify(metrics))

    if not random_repr and not random_weights and not masked_data and not new_baseline and not pretrained:
        # save losses
        loss_file = f"explainableAV/results/probing_losses/probing_losses_{model_name}.json"
        if os.path.exists(loss_file):
            losses = load_dataset(loss_file)
        else:
            losses = {}
        losses = defaultdict(lambda: defaultdict(dict), losses)
        losses[layer_num]['Train_loss'] = train_losses
        losses[layer_num]['Val_loss'] = val_losses 
        create_dataset(loss_file, dictify(losses))

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='explainableAV/Amazon/SS_test_15000.json')
    parser.add_argument('--model_name', type=str, default="LUAR", help="Model to use, one of: 'LUAR', 'ModernBERT', 'StyleDistance'")
    parser.add_argument('--seed', default=0, help='Set seed')
    parser.add_argument('--random_representations', action='store_true')
    parser.add_argument('--random_weights', action='store_true')
    parser.add_argument('--pretrained_model', action='store_true')
    parser.add_argument('--masked_data', action='store_true')
    parser.add_argument('--masked_baseline_data', action='store_true')
    parser.add_argument('--heads', action='store_true', help='Probe on the heads if true else on the layers')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = argument_parser()

    # set model name
    if args.model_name == "LUAR":
        model_name = "gabrielloiseau/LUAR-MUD-sentence-transformers"
    elif args.model_name == "StyleDistance":
        model_name = "StyleDistance/styledistance"
    elif args.model_name == "ModernBERT":
        model_name = 'gabrielloiseau/ModernBERT-base-authorship-verification'
    else:
        print("Model name not recognised, choose one of: 'LUAR', 'StyleDistance', 'ModernBERT'")

    # load required data and models
    model = SentenceTransformer(model_name) 
    if args.random_weights: # always on the layers only
        config = AutoConfig.from_pretrained(model_name)
        config.output_hidden_states = True
        transformer_model = AutoModel.from_config(config)
    elif args.pretrained_model:
        if args.model_name == 'LUAR':
            transformer_model = AutoModel.from_pretrained("distilroberta-base", output_hidden_states=True)
        elif args.model_name == 'ModernBERT':
            transformer_model = AutoModel.from_pretrained("answerdotai/ModernBERT-base", output_hidden_states=True)
        elif args.model_name == 'StyleDistance':
            transformer_model = AutoModel.from_pretrained("roberta-base", output_hidden_states=True)
    else:
        transformer_model = AutoModel.from_pretrained(model_name, output_hidden_states=True)

    # generate data if necessary otherwise load data from dataset
    file_name = f'explainableAV/probes/hidden_states_probes_{args.model_name}.pkl.gz'
    data = load_dataset(args.data_path)
    # balanced_data = downsampling(data, 2000)
    balanced_data = balance_dataset(data, samples_per_class=2000)
    if args.random_representations:
        probe_data = collect_probing_data_random(balanced_data, model, transformer_model, args.model_name)
    else:
        probe_data = collect_probing_data(balanced_data, model, transformer_model, args.model_name)
 

    if args.model_name == 'LUAR' or args.model_name == 'StyleDistance':
        for i in range(1, len(transformer_model.encoder.layer)+1):
            if args.random_representations:
                perform_logistic_regression(probe_data, i, args.model_name, random_repr=True)
            elif args.random_weights:
                perform_logistic_regression(probe_data, i, args.model_name, random_weights=True)
            elif args.pretrained_model:
                perform_logistic_regression(probe_data, i, args.model_name, pretrained=True)
            elif args.masked_data:
                perform_logistic_regression(probe_data, i, args.model_name, masked_data=True)
            elif args.masked_baseline_data:
                perform_logistic_regression(probe_data, i, args.model_name, new_baseline=True)
            else:
                perform_logistic_regression(probe_data, i, args.model_name)
                print('--------CROSS-VALIDATION-----------')
                perform_logistic_regression_cv(probe_data, i)
    elif args.model_name == 'ModernBERT':
        for i in range(len(transformer_model.layers)+1): # set manually if necessary due to size of representations
            if args.random_representations:
                perform_logistic_regression(probe_data, i, args.model_name, random_repr=True)
            elif args.random_weights:
                perform_logistic_regression(probe_data, i, args.model_name, random_weights=True)
            elif args.pretrained_model:
                perform_logistic_regression(probe_data, i, args.model_name, pretrained=True)
            elif args.masked_data:
                perform_logistic_regression(probe_data, i, args.model_name, masked_data=True)
            elif args.masked_baseline_data:
                perform_logistic_regression(probe_data, i, args.model_name, new_baseline=True)
            else:
                perform_logistic_regression(probe_data, i, args.model_name)
                print('--------CROSS-VALIDATION-----------')
                perform_logistic_regression_cv(probe_data, i)
