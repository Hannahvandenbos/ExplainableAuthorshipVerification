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

## Perturb the Texts
In order to perturb the texts, you first have the extract the topic words with Guided lda:
```sh
# Amazon
python -m explainableAV.extract_topic.guided_lda --data_path "explainableAV/Amazon/test_set_15000x4.json"
# PAN20
python -m explainableAV.extract_topic.guided_lda --data_path "explainableAV/PAN20/test_set_2500x4.json" --save_name "explainableAV/extract_topic/pan20_topic_related_all_nouns.json" --data_name 'pan20'
```

To evaluate Guided LDA run:
```sh
# Amazon
python -m explainableAV.extract_topic.guided_lda_evaluation --data_path "explainableAV/Amazon/test_set_15000x4.json" --data_name 'amazon' --evaluate_masks 
python -m explainableAV.extract_topic.guided_lda_evaluation --data_path "explainableAV/Amazon/test_set_15000x4.json" --data_name 'amazon' --inter_distance

# PAN20
python -m explainableAV.extract_topic.guided_lda_evaluation --data_path "explainableAV/PAN20/test_set_2500x4.json" --data_name 'pan20' --inter_distance
python -m explainableAV.extract_topic.guided_lda_evaluation --data_path "explainableAV/PAN20/test_set_2500x4.json" --data_name 'pan20' --evaluate_masks
```

To compute the perturbed texts, you can run the following file, and see the file for the command line arguments (there are a lot of variations):
```sh
# general perturbations
python -m explainableAV.change_topic.mask_words

# LLM perturbations (set the prefered file name)
python -m explainableAV.change_topic.llm_perturbations --data_path "explainableAV/Amazon/test_set_15000x4.json" --save "explainableAV/change_topic/..."
```

To compute the mask quality, run commands like the following:
```sh
# Example for pos tag perturbation on PAN20 data only comparing the first text in each pair
python -m explainableAV.change_topic.mask_quality --data_path_SS "explainableAV/PAN20/SS_test_2500.json" --data_path_SD "explainableAV/PAN20/SD_test_2500.json" --data_path_DS "explainableAV/PAN20/DS_test_2500.json" --data_path_DD "explainableAV/PAN20/DD_test_2500.json" --masked_data_path_SS "explainableAV/change_topic/pan20_lda_SS_pos tag_False_False.json" --masked_data_path_SD "explainableAV/change_topic/pan20_lda_SD_pos tag_False_False.json" --masked_data_path_DS "explainableAV/change_topic/pan20_lda_DS_pos tag_False_False.json" --masked_data_path_DD "explainableAV/change_topic/pan20_lda_DD_pos tag_False_False.json" --mask_one_text --mask_type 'pos tag' --dataset_name 'pan20'
```

## Experiments
### Behavioral (Input-Output Relations)
In order to compute the model classification thresholds, you can run (change the model name accordingly):
```sh
# Amazon
python -m explainableAV.models.find_threshold --SS_val_path "explainableAV/Amazon/SS_val_7500.json" --SD_val_path "explainableAV/Amazon/SD_val_7500.json" --DS_val_path "explainableAV/Amazon/DS_val_7500.json" --DD_val_path "explainableAV/Amazon/DD_val_7500.json" --model_name "LUAR" --dataset_name "amazon"

# PAN20
python -m explainableAV.models.find_threshold --SS_val_path "explainableAV/PAN20/SS_val_1250.json" --SD_val_path "explainableAV/PAN20/SD_val_1250.json" --DS_val_path "explainableAV/PAN20/DS_val_1250.json" --DD_val_path "explainableAV/PAN20/DD_val_1250.json" --model_name "LUAR" --dataset_name "pan20"
```

To get the original model performance on the Amazon SS data for LUAR (on the original data):
```sh
python -m explainableAV.models.test --data_path "explainableAV/Amazon/SS_test_15000.json" --model_name "LUAR" --mask_type 'original' --data_split 'SS'

# Example of running with masked POS tag data, only altering the first text (single-sided perturbation)
python -m explainableAV.models.test --data_path "explainableAV/Amazon/SS_test_15000.json" --extra_data_path "explainableAV/change_topic/amazon_lda_SS_pos tag_False_False.json" --model_name "LUAR" --mask_type 'pos tag' --data_split 'SS' 

```
Similarly the arguments can be altered to run the other combinations (also for the masked data)

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

### Concept-Based (Probing)
To probe the hidden states of the model, run:
```sh
python -m explainableAV.probes.probing --model_name 'LUAR' --pair_type 'SS' --data_path 'explainableAV/Amazon/SS_test_15000.json'
```
Use --pretrained_model for the pretrained results, and --masked_data for the masked data results

### Plotting the results
All results can be plotted as following:
```sh
# lda evaluation
python -m explainableAV.models.results.plot_results --plot_type "lda"

# topic distribution of data
python -m explainableAV.models.results.plot_results --plot_type "topic_distribution" --dataset_name 'amazon'
python -m explainableAV.models.results.plot_results --plot_type "topic_distribution" --dataset_name 'pan20'

# text similarity
python -m explainableAV.models.results.plot_results --plot_type "text_similarity" --dataset_name 'amazon'
python -m explainableAV.models.results.plot_results --plot_type "text_similarity" --dataset_name 'pan20'

# confusion plots
python -m explainableAV.models.results.plot_results --plot_type "confusion" --experiment 'both' --baseline --dataset_name 'amazon'
python -m explainableAV.models.results.plot_results --plot_type "confusion" --experiment 'first' --baseline --dataset_name 'amazon'
python -m explainableAV.models.results.plot_results --plot_type "confusion" --experiment 'both' --baseline --dataset_name 'pan20'
python -m explainableAV.models.results.plot_results --plot_type "confusion" --experiment 'first' --baseline --dataset_name 'pan20'

# attention top words
 python -m explainableAV.models.results.plot_results --plot_type "top_attention_words"

# heatmaps
python -m explainableAV.models.results.plot_results --plot_type "heatmaps" --experiment 'both' --dataset_name 'pan20' --baseline
python -m explainableAV.models.results.plot_results --plot_type "heatmaps" --experiment 'first' --dataset_name 'pan20' --baseline

# probing accuracy 
python -m explainableAV.models.results.plot_results --plot_type 'probing_line_plot'

# probing heatmaps
python -m explainableAV.models.results.plot_results --plot_type 'probing_heatmap' --model_name 'LUAR'
python -m explainableAV.models.results.plot_results --plot_type 'probing_heatmap' --model_name 'ModernBERT'
python -m explainableAV.models.results.plot_results --plot_type 'probing_heatmap' --model_name 'StyleDistance'
python -m explainableAV.models.results.plot_results --plot_type 'probing_heatmap_f1'

# probing learning curve
python -m explainableAV.models.results.plot_results --plot_type 'probing_learning_curve' --model_name 'LUAR'
python -m explainableAV.models.results.plot_results --plot_type 'probing_learning_curve' --model_name 'ModernBERT'
python -m explainableAV.models.results.plot_results --plot_type 'probing_learning_curve' --model_name 'StyleDistance'
```
