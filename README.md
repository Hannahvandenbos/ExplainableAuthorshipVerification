**Code will be uploaded soon**

# Explainable Authorship Verification: Topic Reliance in Transformer Models

This repository provides the code used for the following MSc AI thesis at the University of Amsterdam and as part of an internship at the Netherlands Forensic Institute: \
*Explainable Authorship Verification: Topic Reliance in Transformer Models*

**Abstract** \
Authorship Verification (AV) analyzes two texts to determine whether they are written by the same author. In plagiarism detection, digital forensics, and legal proceedings, AV can have severe implications, resulting in a need for transparent decision making. Although the impact of various stylometric features has been studied for traditional machine learning techniques, there is limited research on the explainability of faster and higher performing Transformer models in the context of AV. Specifically, the effect of topic information, an inconsistent indicator for authorship, is underexplored. This thesis investigates the role of the topic of a text for BERT-related AV models. We introduce a three-level explainability framework for AV that examines input-output relations, attention patterns, and hidden state representations. Quantitative experiments show that substituting topic-related words can affect accuracy by -12.5\% to +43.85\%, depending on the model, dataset, and perturbation technique. Additionally, we find that topic information is reflected in the attention distributions, demonstrating a noticeable effect on topic bias in the predictions. Finally, probing experiments reveal that topic information is consistently encoded in the hidden representations of the models. These results indicate the importance of controlling for topic information in AV tasks to preserve style-based decision making, improving model performance and interpretability.

## Installation
To install the necessary packages run the following:

```sh
# Upgrade pip
pip install --upgrade pip
```

```sh
# Install and build GuidedLDA
git clone https://github.com/vi3k6i5/GuidedLDA
cd GuidedLDA
sh build_dist.sh
python setup.py sdist
pip install -e .
cd ..
```

```sh
# Install dependencies
pip install -r requirements.txt
```

```sh
# Download Spacy model
python -m spacy download en_core_web_sm
```

## Data Preparation
Here, we explain how and where to download the data from and how to process it to ensure applicability to our experiments. We use two datasets: Amazon reviews and Fanfictions from the PAN2020 competition.

### Downloading the Data
The *Amazon Reviews* dataset needs to be downloaded from: https://nijianmo.github.io/amazon/index.html \
Download the 5-core files for the following categories and extract them to the *explainableAV/Amazon* folder:
<ol>
  <li>"Amazon Fashion"</li>
  <li>"All Beauty"</li>
  <li>"Appliances"</li>
  <li>"Arts, Crafts and Sewing"</li>
  <li>"Automotive"</li>
  <li>"CDs and Vinyl"</li>
  <li>"Cell Phones and Accessoires"</li>
  <li>"Clothing, Shoes and Jewelry"</li>
  <li>"Digital Music"</li>
  <li>"Gift Cards"</li>
  <li>"Grocery and Gourmet Foods"</li>
  <li>"Home and Kitchen"</li>
  <li>"Industrial and Scientific"</li>
  <li>"Prime Pantry"</li>
  <li>"Software"</li>
  <li>"Video Games"</li>
</ol> 

The PAN20 dataset needs to be downloaded from: https://zenodo.org/records/3724096 \
You can opt for both the large or small version. The small version was used in this research. \
Store the two jsonl files in the *explainableAV/PAN20* folder.

### Data Preprocessing
Run the following commands to filter and reorder the datasets:
```sh
# Amazon
python -m explainableAV/data_prep/reorder_Amazon.py

# PAN20 (small)
python -m explainableAV/data_prep/reorder_PAN20.py --texts_path "explainableAV/PAN20/.pan20-authorship-verification-training-small.jsonl" --label_path "explainableAV/PAN20/pan20-authorship-verification-training-small-truth.jsonl"

# PAN20 (large)
python -m explainableAV/reorder_PAN20.py --texts_path "explainableAV/PAN20/pan20-authorship-verification-training-large.jsonl" --label_path "explainableAV/PAN20/pan20-authorship-verification-training-large-truth.jsonl"
```

To create all text pairs (SS, SD, DS, DD), run the following commands:
```sh
# Amazon 
python -m explainableAV.data_prep.create_pairs --dataset_path "explainableAV/Amazon/amazon_reviews_final.json" --SS_file_path "explainableAV/Amazon/SS.json" --SD_file_path "explainableAV/Amazon/SD.json" --DS_file_path "explainableAV/Amazon/DS.json" --DD_file_path "explainableAV/Amazon/DD.json"

# PAN20
python -m explainableAV.data_prep.create_pairs --dataset_path "explainableAV/PAN20/PAN20_filtered.json" --SS_file_path "explainableAV/PAN20/SS.json" --SD_file_path "explainableAV/PAN20/SD.json" --DS_file_path "explainableAV/PAN20/DS.json" --DD_file_path "explainableAV/PAN20/DD.json"
```

Finally, to create the train, test, and validation splits of the text pairs, run the following commands:
```sh
# Amazon
python -m explainableAV.data_prep.data_split --samples_per_pair 15000 --SS_file_path "explainableAV/Amazon/SS.json" --SD_file_path "explainableAV/Amazon/SD.json" --DS_file_path "explainableAV/Amazon/DS.json" --DD_file_path "explainableAV/Amazon/DD.json"

# PAN20
python -m explainableAV.data_prep.data_split --samples_per_pair 2500 --SS_file_path "explainableAV/PAN20/SS.json" --SD_file_path "explainableAV/PAN20/SD.json" --DS_file_path "explainableAV/PAN20/DS.json" --DD_file_path "explainableAV/PAN20/DD.json"
```

## Experiments
### Attributional (Attention) 
There are various experiments that can be run for the attention examination. The experiments from the thesis can be run by the following commands:
```sh
# faithfulness evaluation
python -m explainableAV.attention.attention --data_path 'explainableAV/attention/attention_most_influence.json' --attention_type 'raw' --model_name 'LUAR' --faithfulness
python -m explainableAV.attention.attention --data_path 'explainableAV/attention/attention_most_influence.json' --attention_type 'rollout' --model_name 'LUAR' --faithfulness
python -m explainableAV.attention.attention --data_path 'explainableAV/attention/attention_most_influence.json' --attention_type 'value_zeroing' --model_name 'LUAR' --faithfulness
python -m explainableAV.attention.attention --data_path 'explainableAV/attention/attention_most_influence.json' --attention_type 'raw' --model_name 'ModernBERT' --faithfulness
python -m explainableAV.attention.attention --data_path 'explainableAV/attention/attention_most_influence.json' --attention_type 'rollout' --model_name 'ModernBERT' --faithfulness
python -m explainableAV.attention.attention --data_path 'explainableAV/attention/attention_most_influence.json' --attention_type 'value_zeroing' --model_name 'ModernBERT' --faithfulness
python -m explainableAV.attention.attention --data_path 'explainableAV/attention/attention_most_influence.json' --attention_type 'raw' --model_name 'StyleDistance' --faithfulness
python -m explainableAV.attention.attention --data_path 'explainableAV/attention/attention_most_influence.json' --attention_type 'rollout' --model_name 'StyleDistance' --faithfulness
python -m explainableAV.attention.attention --data_path 'explainableAV/attention/attention_most_influence.json' --attention_type 'value_zeroing' --model_name 'StyleDistnace' --faithfulness

# attention scores for topic words
python -m explainableAV.attention.attention --data_path 'explainableAV/attention/attention_most_influence.json' --attention_type 'rollout' --model_name 'LUAR' --topic_words_attention
python -m explainableAV.attention.attention --data_path 'explainableAV/attention/attention_most_influence.json' --attention_type 'value_zeroing' --model_name 'ModernBERT' --topic_words_attention 
python -m explainableAV.attention.attention --data_path 'explainableAV/attention/attention_most_influence.json' --attention_type 'raw' --model_name 'StyleDistance' --topic_words_attention 

# Qualitative experiments text plots
python -m explainableAV.attention.attention --data_path 'explainableAV/attention/attention_most_influence.json' --attention_type 'value_zeroing' --model_name 'LUAR' --plot_type 'text_plot' --datapoint 246
python -m explainableAV.attention.attention --data_path 'explainableAV/attention/attention_most_influence.json' --attention_type 'value_zeroing' --model_name 'ModernBERT' --plot_type 'text_plot' --datapoint 246
python -m explainableAV.attention.attention --data_path 'explainableAV/attention/attention_most_influence.json' --attention_type 'value_zeroing' --model_name 'StyleDistance' --plot_type 'text_plot' --datapoint 246
python -m explainableAV.attention.attention --data_path 'explainableAV/attention/attention_most_influence.json' --attention_type 'value_zeroing' --model_name 'LUAR' --plot_type 'text_plot' --datapoint 338
python -m explainableAV.attention.attention --data_path 'explainableAV/attention/attention_most_influence.json' --attention_type 'value_zeroing' --model_name 'ModernBERT' --plot_type 'text_plot' --datapoint 338
python -m explainableAV.attention.attention --data_path 'explainableAV/attention/attention_most_influence.json' --attention_type 'value_zeroing' --model_name 'StyleDistance' --plot_type 'text_plot' --datapoint 338

# Attention Ablation (this one deviates from the others and is explained inside the file)
python -m explainableAV.ablation_study.ablation --data_path "explainableAV/Amazon/SS_test_15000.json" --model_name "LUAR" --pair_type 'SS' --ablate_attention
```

The file *explainableAV/attention/attention.py* supports the following arguments:

| Argument        | Type  | Default | Description                         |
|----------------|-------|---------|-------------------------------------|
| `--data_path`      | str   | 'explainableAV/Amazon/test_set_15000x4.json'      | Data file path           |
| `--model_name`  | str   | "LUAR"      | Model to use, one of: 'LUAR', 'ModernBERT', 'StyleDistance'         |
| `--seed`   | int   | 0    | Set seed       |
| `--attention_type`   | str   | 'raw'    | Type of attention to apply, choose from 'raw', 'rollout', 'value_zeroing', 'value_zeroing_rollout', 'globenc'       |
| `--pair_type`   | str   | 'SS'    | Pair type: 'SS', 'SD', 'DS', 'DD'       |
| `--plot_type`   | str   | None    | Choose from: 'over_tokens', 'over_layers', 'text_plot', 'per_layer_over_tokens', 'topic_attention_layers'       |
| `--topic_related_path`   | str   | 'explainableAV/extract_topic/amazon_topic_related_8400_filtered.json'    | Path to data with topic-related words       |
| `--datapoint`   | int   | None    | Index for topic attention ratio plot       |
| `--faithfulness`   | action   | None    | Compute faithfulness scores if true       |
| `--topic_words_attention`   | action   | None    | Compute attention to topic words if true      |
| `--visualize_masked`   | action   | None    | Visualize masked version of datapoint for text plot if correct datapoint      |
