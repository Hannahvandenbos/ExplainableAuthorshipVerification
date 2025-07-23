# Explainable Authorship Verification: Topic Reliance in Transformer Models

This repository provides the code used for the following MSc AI thesis at the University of Amsterdam and as part of an internship at the Netherlands Forensic Institute: \
*Explainable Authorship Verification: Topic Reliance in Transformer Models*

**Abstract** \
Authorship Verification (AV) analyzes two texts to determine whether they are written by the same author. In plagiarism detection, digital forensics, and legal proceedings, AV can have severe implications, resulting in a need for transparent decision making. Although the impact of various stylometric features has been studied for traditional machine learning techniques, there is limited research on the explainability of faster and higher performing Transformer models in the context of AV. Specifically, the effect of topic information, an inconsistent indicator for authorship, is underexplored. This thesis investigates the role of the topic of a text for BERT-related AV models. We introduce a three-level explainability framework for AV that examines input-output relations, attention patterns, and hidden state representations. Quantitative experiments show that substituting topic-related words can affect accuracy by -12.5\% to +43.85\%, depending on the model, dataset, and perturbation technique. Additionally, we find that topic information is reflected in the attention distributions, demonstrating a noticeable effect on topic bias in the predictions. Finally, probing experiments reveal that topic information is consistently encoded in the hidden representations of the models. These results indicate the importance of controlling for topic information in AV tasks to preserve style-based decision making, improving model performance and interpretability.

## Code explanation
The code explanation generally follows the storyline from the thesis, including all experiments and figures of how to obtain the same results.
The thesis can be found [here](https://dspace.uba.uva.nl/server/api/core/bitstreams/a2c86cd4-9e90-47e7-8dc6-4638d5650766/content).

## Installation
After downloading the project, install the necessary packages by running the following, we recommend to install it in a virtual or conda environment:

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
The *Amazon Reviews* dataset needs to be downloaded from [Amazon Reviews](https://nijianmo.github.io/amazon/index.html) \
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

The PAN20 dataset needs to be downloaded from [PAN20](https://zenodo.org/records/3724096) \
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

### Results

To print an overview of the number of pairs per pair type in the data run:
```sh
# Amazon
python -m explainableAV.data_prep.data_distributions

# PAN20
python -m explainableAV.data_prep.data_distributions --data_name "PAN20"
```

To print an overview of the number of pairs per split per pair type in the data run:
```sh
# Amazon
python -m explainableAV.data_prep.data_distributions --statistic 'splits'

# PAN20
python -m explainableAV.data_prep.data_distributions --statistic 'splits' --data_name "PAN20"
```

To plot the topic distributions of the train and test dataset, run the following:
```sh
# Amazon
python -m explainableAV.data_prep.data_distributions --statistic 'topic_distribution'

# PAN20
python -m explainableAV.data_prep.data_distributions --statistic 'topic_distribution' --data_name "PAN20"
```

## Perturbing the Texts
In order to perturb the text, first extract topic words with Guided LDA, then perturb in various ways.

### Guided LDA
In order to perturb the texts, you first have the extract the topic words with Guided lda. If you want to use the topic words as used in the thesis, do:
```sh
# Amazon
python -m explainableAV.extract_topic.guided_lda --data_path "explainableAV/Amazon/test_set_15000x4.json"
python -m explainableAV.extract_topic.guided_lda_evaluation --data_path "explainableAV/Amazon/test_set_15000x4.json" --data_name 'amazon' --evaluate_masks # evaluation 
python -m explainableAV.extract_topic.guided_lda_evaluation --data_path "explainableAV/Amazon/test_set_15000x4.json" --data_name 'amazon' --inter_distance # evaluation

# PAN20
python -m explainableAV.extract_topic.guided_lda --data_path "explainableAV/PAN20/test_set_2500x4.json" --save_name "explainableAV/extract_topic/pan20_topic_related_all_nouns.json" --data_name 'pan20'
python -m explainableAV.extract_topic.guided_lda_evaluation --data_path "explainableAV/PAN20/test_set_2500x4.json" --data_name 'pan20' --inter_distance # evaluation
python -m explainableAV.extract_topic.guided_lda_evaluation --data_path "explainableAV/PAN20/test_set_2500x4.json" --data_name 'pan20' --evaluate_masks # evaluation
```

To evaluate multiple number of topic words with Guided LDA run:
```sh
# Amazon
python -m explainableAV.extract_topic.guided_lda --data_"explainableAV/Amazon/test_set_15000x4.json" --evaluate
python -m explainableAV.extract_topic.guided_lda_evaluation --data_path "explainableAV/Amazon/test_set_15000x4.json" --data_name 'amazon' --evaluate_masks --evaluate # evaluation
python -m explainableAV.extract_topic.guided_lda_evaluation --data_path "explainableAV/Amazon/test_set_15000x4.json" --data_name 'amazon' --inter_distance --evaluate # evaluation

# PAN20
python -m explainableAV.extract_topic.guided_lda --data_path "explainableAV/PAN20/test_set_2500x4.json" --data_name 'pan20' --evaluate
python -m explainableAV.extract_topic.guided_lda_evaluation --data_path "explainableAV/PAN20/test_set_2500x4.json" --data_name 'pan20' --inter_distance --evaluate # evaluation
python -m explainableAV.extract_topic.guided_lda_evaluation --data_path "explainableAV/PAN20/test_set_2500x4.json" --data_name 'pan20' --evaluate_masks --evaluate # evaluation
```

### Perturbations
To create the perturbed texts (Asterisk, POS tag, One words, and Swap), for single-sided perturbation, you can run the following commands for the SS test set on the Amazon data: 
```sh
# general perturbations
python -m explainableAV.change_topic.mask_words --data_path "explainableAV/Amazon/SS_test.json --topic_related_path "explainableAV/extract_topic/amazon_topic_related_8400_filtered.json" --mask_type "asterisk" --save --mask_one_text # asterisk
python -m explainableAV.change_topic.mask_words --data_path "explainableAV/Amazon/SS_test.json --topic_related_path "explainableAV/extract_topic/amazon_topic_related_8400_filtered.json" --mask_type "pos tag" --save --mask_one_text # pos tag 
python -m explainableAV.change_topic.mask_words --data_path "explainableAV/Amazon/SS_test.json --topic_related_path "explainableAV/extract_topic/amazon_topic_related_8400_filtered.json" --mask_type "one word" --save --mask_one_text # one word
python -m explainableAV.change_topic.mask_words --data_path "explainableAV/Amazon/SS_test.json --topic_related_path "explainableAV/extract_topic/amazon_topic_related_8400_filtered.json" --mask_type "change topic" --save --mask_one_text --different #swap 
```
Replace the --data_path to correspond with the other pair types (SD_test.json, DS_test.json, and DD_test.json). For dual perturbation remove --mask_one_text. For swap, there is no dual perturbation. Additionally, when changing the pair type for swap, SS and DS should have --different, SD and DD should **not** have --different. 
To run on the PAN20 data, replace the --data_path with "explainableAV/PAN20/..." with ... the corresponding pair type file and replace --topic_related_path with "explainableAV/extract_topic/pan20_topic_related_all_nouns_filtered.json" and add the argument --data_name "pan20"
Last, to create the **perturbation-specific baselines** for the behavioral experiment, add --baseline to each command

To create the LLM perturbation (Amazon only), run:
```sh
python -m explainableAV.change_topic.llm_perturbations --data_path "explainableAV/Amazon/SS_test.json" --save "explainableAV/change_topic/Amazon/amazon_llama_SS.json"

# To clean the perturbation afterwards (remove some artifacts from LLMs), run:
ython -m explainableAV.change_topic.llm_clean --llm_data_path "explainableAV/change_topic/Amazon/amazon_llama_SS.json" --original_data_path "explainableAV/Amazon/SS_test.json" --save "explainableAV/change_topic/Amazon/amazon_llama_SS_cleaned.json"
```
Again, replace the --data_path to correspond with the other pair types (SD_test.json, DS_test.json, and DD_test.json).

## Perturbation Quality
To compute the mask quality, run commands like the following:
```sh
# Example for asterisk perturbation on Amazon data only comparing the first text in each pair
python -m explainableAV.change_topic.mask_quality --data_path_SS "explainableAV/Amazon/SS_test.json" --data_path_SD "explainableAV/Amazon/SD_test.json" --data_path_DS "explainableAV/Amazon/DS_test.json" --data_path_DD "explainableAV/Amazon/DD_test.json" --masked_data_path_SS "explainableAV/change_topic/Amazon/mazon_lda_SS_asterisk_False_False.json" --masked_data_path_SD "explainableAV/change_topic/Amazon/amazon_lda_SD_asterisk_False_False.json" --masked_data_path_DS "explainableAV/change_topic/Amazon/amazon_lda_DS_asterisk_False_False.json" --masked_data_path_DD "explainableAV/change_topic/Amazon/amazon_lda_DD_asterisk_False_False.json" --mask_one_text --mask_type 'asterisk'
```
Replace the files to match the 'POS tag', 'One word', 'Swap', and 'LLM' perturbations or PAN20 dataset.
--mask_one_text ensures a fair comparison between all perturbation techniques, but can be removed for 'Asterisk', 'POS tag', and 'One word'.

### Plot
To plot the results from the perturbation quality, run the following:
```sh
# Amazon
python -m explainableAV.change_topic.perturbation_quality_plot

# PAN20
python -m explainableAV.change_topic.perturbation_quality_plot --dataset_name "pan20"
```

## Experiments Setup
### Model Thresholds
To find the optimal thresholds for the AV models, according to the minimum standard deviation between the accuracies over the pair types, run the following code whereby the plots over the thresholds are plotted as well (in explainableAV/models/results/):
```sh
python -m explainableAV.models.find_thresholds # Amazon data, LUAR model
python -m explainableAV.models.find_thresholds --model_name 'ModernBERT' # Amazon data, ModernBERT model
python -m explainableAV.models.find_thresholds --model_name 'StyleDistance' # Amazon data, StyleDistance model

python -m explainableAV.models.find_thresholds --SS_val_path "explainableAV/PAN20/SS_val.json" --SD_val_path "explainableAV/PAN20/SD_val.json" --DS_val_path "explainableAV/PAN20/DS_val.json" --DD_val_path "explainableAV/PAN20/DD_val.json" --dataset_name "pan20"  # PAN20 data, LUAR model
python -m explainableAV.models.find_thresholds --SS_val_path "explainableAV/PAN20/SS_val.json" --SD_val_path "explainableAV/PAN20/SD_val.json" --DS_val_path "explainableAV/PAN20/DS_val.json" --DD_val_path "explainableAV/PAN20/DD_val.json" --dataset_name "pan20" --model_name 'ModernBERT' # PAN20 data, ModernBERT model
python -m explainableAV.models.find_thresholds --SS_val_path "explainableAV/PAN20/SS_val.json" --SD_val_path "explainableAV/PAN20/SD_val.json" --DS_val_path "explainableAV/PAN20/DS_val.json" --DD_val_path "explainableAV/PAN20/DD_val.json" --dataset_name "pan20" --model_name 'StyleDistance' # PAN20 data, StyleDistance model
```

## Behavioral (Input-Output Relations)
To get the **original/baseline** model performance, run the following, results are stored in explainableAV/results/predictions:
```sh
python -m explainableAV.models.test --data_path "explainableAV/Amazon/SS_test.json" --model_name "LUAR" --data_split "SS" --dataset_name "amazon" # Amazon SS data, LUAR model
```
Replace the data, model_name, data_split and dataset_name to obtain all results
Additionally, run all commands again with --perturb_second to ensure that future results can be computed

### Perturbed Texts
To obtain the results for the **perturbed texts**, run the same file with different arguments, results are stored in explainableAV/results/predictions:
```sh
# example usage
python -m explainableAV.models.test --data_path "explainableAV/Amazon/SS_test.json" --extra_data_path "explainableAV/change_topic/Amazon/amazon_lda_DD_asterisk_False_False.json" --perturb_second --model_name "LUAR" --mask_type 'asterisk' --data_split "DD" --dataset_name "amazon" # Amazon SS data, LUAR model, asterisk perturbation, DD pair type, dual perturbation

# --data_path: provide the original text path
# --extra_data_path: provide the corresponding altered text path (including the new baseline path)
# --perturb_second: activate this for dual perturbation (only for 'Asterisk', 'POS tag', and 'One word')
# --model_name: provide AV model: 'LUAR', 'ModernBERT', 'StyleDistance'
# --mask_type: the mask type: 'asterisk', 'pos tag', 'one word', 'change topic', 'llm', 'asterisk_baseline', 'pos tag_baseline', 'one word_baseline', 'change topic_baseline'
# --data_split: pair type: 'SS', 'SD', 'DS', 'DD'
# --dataset_name: 'amazon' or 'pan20'
# --threshold: set if your model threshold differs from the ones in the thesis
```

### Plots
To plot the results of the behavioral experiment:
```sh
# Confusion plot, plotting the TPs, TNs, FPs, and FNs
python -m explainableAV.results.predictions --plot_type 'confusion' --experiment 'first' --dataset_name 'amazon' --baseline # Confusion plot, single-sided perturbation, Amazon data
python -m explainableAV.results.predictions --plot_type 'confusion' --experiment 'both' --dataset_name 'amazon' --baseline # Confusion plot, dual perturbation, Amazon data
python -m explainableAV.results.predictions --plot_type 'confusion' --experiment 'first' --dataset_name 'pan20' --baseline # Confusion plot, single-sided perturbation, PAN20 data
python -m explainableAV.results.predictions --plot_type 'confusion' --experiment 'both' --dataset_name 'pan20' --baseline # Confusion plot, dual perturbation, PAN20 data

# Heatmaps 
python -m explainableAV.results.predictions --plot_type 'heatmaps' --experiment 'first' --dataset_name 'amazon' --baseline # Heatmap plot, single-sided perturbation, Amazon data
python -m explainableAV.results.predictions --plot_type 'heatmaps' --experiment 'both' --dataset_name 'amazon' --baseline # Heatmap plot, dual perturbation, Amazon data
python -m explainableAV.results.predictions --plot_type 'heatmaps' --experiment 'first' --dataset_name 'pan20' --baseline # Heatmap plot, single-sided perturbation, PAN20 data
python -m explainableAV.results.predictions --plot_type 'heatmaps' --experiment 'both' --dataset_name 'pan20' --baseline # Heatmap plot, dual perturbation, PAN20 data

# Additionally, you can manually set the paths to your results when using different names through: --luar_results_path, --modernbert_results_path, --styledistance_results_path
```

## Attributional (Attention)
To create the smaller datasets based on the most (and least) influenced datapoints, run the following:
```sh
python -m explainableAV.data_prep_data_split_attention --data_name 'amazon' --experiment 'both' --mask_type 'asterisk' # arguments used in thesis, but can be altered
```
The subdatasets are stored in explainableAV/Amazon/

### Faithfulness Evaluation
```sh
python -m explainableAV.attention.attention --data_path 'explainableAV/Amazon/attention_most_influence.json' --attention_type 'raw' --model_name 'LUAR' --faithfulness # raw attention, LUAR model
python -m explainableAV.attention.attention --data_path 'explainableAV/Amazon/attention_most_influence.json' --attention_type 'rollout' --model_name 'LUAR' --faithfulness # attention rollout, LUAR model
python -m explainableAV.attention.attention --data_path 'explainableAV/Amazon/attention_most_influence.json' --attention_type 'value_zeroing' --model_name 'LUAR' --faithfulness # value zeroing, LUAR model
python -m explainableAV.attention.attention --data_path 'explainableAV/Amazon/attention_most_influence.json' --attention_type 'raw' --model_name 'ModernBERT' --faithfulness # raw attention, ModernBERT model
python -m explainableAV.attention.attention --data_path 'explainableAV/Amazon/attention_most_influence.json' --attention_type 'rollout' --model_name 'ModernBERT' --faithfulness # attention rollout, ModernBERT model
python -m explainableAV.attention.attention --data_path 'explainableAV/Amazon/attention_most_influence.json' --attention_type 'value_zeroing' --model_name 'ModernBERT' --faithfulness # value zeroing, ModernBERT model
python -m explainableAV.attention.attention --data_path 'explainableAV/Amazon/attention_most_influence.json' --attention_type 'raw' --model_name 'StyleDistance' --faithfulness # raw attention, StyleDistance model
python -m explainableAV.attention.attention --data_path 'explainableAV/Amazon/attention_most_influence.json' --attention_type 'rollout' --model_name 'StyleDistance' --faithfulness # attention rollout, StyleDistance model
python -m explainableAV.attention.attention --data_path 'explainableAV/Amazon/attention_most_influence.json' --attention_type 'value_zeroing' --model_name 'StyleDistnace' --faithfulness # value zeroing, StyleDistance model
```
The results are stored in explainableAV/results/attention/

### Attention Scores for Topic Words
```sh
python -m explainableAV.attention.attention --data_path 'explainableAV/Amazon/attention_most_influence.json' --attention_type 'rollout' --model_name 'LUAR' --topic_words_attention
python -m explainableAV.attention.attention --data_path 'explainableAV/Amazon/attention_most_influence.json' --attention_type 'value_zeroing' --model_name 'ModernBERT' --topic_words_attention 
python -m explainableAV.attention.attention --data_path 'explainableAV/Amazon/attention_most_influence.json' --attention_type 'raw' --model_name 'StyleDistance' --topic_words_attention 
```
Results are stored in explainableAV/results/attention/

#### Plots
```sh
python -m explainableAV.results.attention.plot

# Additionally, you can manually set the paths to your results when using different names through: --luar_results_path, --modernbert_results_path, --styledistance_results_path
```

### Attention Distribution
```sh
python -m explainableAV.attention.attention --data_path 'explainableAV/Amazon/attention_most_influence.json' --attention_type 'value_zeroing' --model_name 'LUAR' --plot_type 'text_plot' --datapoint 246
python -m explainableAV.attention.attention --data_path 'explainableAV/Amazon/attention_most_influence.json' --attention_type 'value_zeroing' --model_name 'ModernBERT' --plot_type 'text_plot' --datapoint 246
python -m explainableAV.attention.attention --data_path 'explainableAV/Amazon/attention_most_influence.json' --attention_type 'value_zeroing' --model_name 'StyleDistance' --plot_type 'text_plot' --datapoint 246
python -m explainableAV.attention.attention --data_path 'explainableAV/Amazon/attention_most_influence.json' --attention_type 'value_zeroing' --model_name 'LUAR' --plot_type 'text_plot' --datapoint 338
python -m explainableAV.attention.attention --data_path 'explainableAV/Amazon/attention_most_influence.json' --attention_type 'value_zeroing' --model_name 'ModernBERT' --plot_type 'text_plot' --datapoint 338
python -m explainableAV.attention.attention --data_path 'explainableAV/Amazon/attention_most_influence.json' --attention_type 'value_zeroing' --model_name 'StyleDistance' --plot_type 'text_plot' --datapoint 338
```
The plots are stored in explainableAV/results/attention/

### Attention Ablation
```sh
# LUAR
python -m explainableAV.ablation_study.ablation --data_path "explainableAV/Amazon/SS_test.json" --model_name "LUAR" --pair_type 'SS' --ablate_attention
python -m explainableAV.ablation_study.ablation --data_path "explainableAV/Amazon/SD_test.json" --model_name "LUAR" --pair_type 'SD' --ablate_attention
python -m explainableAV.ablation_study.ablation --data_path "explainableAV/Amazon/DS_test.json" --model_name "LUAR" --pair_type 'DS' --ablate_attention
python -m explainableAV.ablation_study.ablation --data_path "explainableAV/Amazon/DD_test.json" --model_name "LUAR" --pair_type 'DD' --ablate_attention

# ModernBERT
python -m explainableAV.ablation_study.ablation --data_path "explainableAV/Amazon/SS_test.json" --model_name "ModernBERT" --pair_type 'SS' --ablate_attention
python -m explainableAV.ablation_study.ablation --data_path "explainableAV/Amazon/SD_test.json" --model_name "ModernBERT" --pair_type 'SD' --ablate_attention
python -m explainableAV.ablation_study.ablation --data_path "explainableAV/Amazon/DS_test.json" --model_name "ModernBERT" --pair_type 'DS' --ablate_attention
python -m explainableAV.ablation_study.ablation --data_path "explainableAV/Amazon/DD_test.json" --model_name "ModernBERT" --pair_type 'DD' --ablate_attention

# StyleDistance
python -m explainableAV.ablation_study.ablation --data_path "explainableAV/Amazon/SS_test.json" --model_name "StyleDistance" --pair_type 'SS' --ablate_attention
python -m explainableAV.ablation_study.ablation --data_path "explainableAV/Amazon/SD_test.json" --model_name "StyleDistance" --pair_type 'SD' --ablate_attention
python -m explainableAV.ablation_study.ablation --data_path "explainableAV/Amazon/DS_test.json" --model_name "StyleDistance" --pair_type 'DS' --ablate_attention
python -m explainableAV.ablation_study.ablation --data_path "explainableAV/Amazon/DD_test.json" --model_name "StyleDistance" --pair_type 'DD' --ablate_attention
```
The results are printed.

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
