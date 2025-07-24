# Explainable Authorship Verification: Topic Reliance in Transformer Models

This repository provides the code used for the following MSc AI thesis at the University of Amsterdam and as part of an internship at the Netherlands Forensic Institute: \
*Explainable Authorship Verification: Topic Reliance in Transformer Models*

**Abstract** \
Authorship Verification (AV) analyzes two texts to determine whether they are written by the same author. In plagiarism detection, digital forensics, and legal proceedings, AV can have severe implications, resulting in a need for transparent decision making. Although the impact of various stylometric features has been studied for traditional machine learning techniques, there is limited research on the explainability of faster and higher performing Transformer models in the context of AV. Specifically, the effect of topic information, an inconsistent indicator for authorship, is underexplored. This thesis investigates the role of the topic of a text for BERT-related AV models. We introduce a three-level explainability framework for AV that examines input-output relations, attention patterns, and hidden state representations. Quantitative experiments show that substituting topic-related words can affect accuracy by -12.5\% to +43.85\%, depending on the model, dataset, and perturbation technique. Additionally, we find that topic information is reflected in the attention distributions, demonstrating a noticeable effect on topic bias in the predictions. Finally, probing experiments reveal that topic information is consistently encoded in the hidden representations of the models. These results indicate the importance of controlling for topic information in AV tasks to preserve style-based decision making, improving model performance and interpretability.

## Code explanation
The code explanation generally follows the storyline from the thesis, including all experiments and figures of how to obtain the same results.
The thesis can be found [here](https://dspace.uba.uva.nl/server/api/core/bitstreams/a2c86cd4-9e90-47e7-8dc6-4638d5650766/content).

We recommend the following order to ensure you have everything you need:
1. [Initialization](Usage/Initialization.md)
2. [Data Preparation](Usage/Data_preparation.md)
3. [Text Perturbations](Usage/Text_perturbation.md)
4. [Experiments Setup](Usage/Experiments_setup.md)
5. [Behavioral Experiments](Usage/Behavioral.md)
6. [Attributional Experiments](Usage/Attributional.md)
7. [Concept-based Experiments](Usage/Concept_based.md)

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

## Concept-Based (Probing)
To probe the hidden states of the model, run:
```sh
# Fine-tuned test
python -m explainableAV.probes.probing --model_name 'LUAR' --pair_type 'SS' --data_path 'explainableAV/Amazon/SS_test.json'
python -m explainableAV.probes.probing --model_name 'LUAR' --pair_type 'SD' --data_path 'explainableAV/Amazon/SD_test.json'
python -m explainableAV.probes.probing --model_name 'LUAR' --pair_type 'DS' --data_path 'explainableAV/Amazon/DS_test.json'
python -m explainableAV.probes.probing --model_name 'LUAR' --pair_type 'DD' --data_path 'explainableAV/Amazon/DD_test.json'
# Repeat for ModernBERT and StyleDistance

# Pre-trained test
python -m explainableAV.probes.probing --model_name 'LUAR' --pair_type 'SS' --data_path 'explainableAV/Amazon/SS_test.json' --pretrained_model
python -m explainableAV.probes.probing --model_name 'LUAR' --pair_type 'SD' --data_path 'explainableAV/Amazon/SD_test.json' --pretrained_model
python -m explainableAV.probes.probing --model_name 'LUAR' --pair_type 'DS' --data_path 'explainableAV/Amazon/DS_test.json' --pretrained_model
python -m explainableAV.probes.probing --model_name 'LUAR' --pair_type 'DD' --data_path 'explainableAV/Amazon/DD_test.json' --pretrained_model
# Repeat for ModernBERT and StyleDistance

# Fine-tuned masked
python -m explainableAV.probes.probing --model_name 'LUAR' --pair_type 'SS' --data_path 'explainableAV/Amazon/amazon_lda_SS_asterisk_False_False.json' --masked_data
python -m explainableAV.probes.probing --model_name 'LUAR' --pair_type 'SD' --data_path 'explainableAV/Amazon/amazon_lda_SD_asterisk_False_False.json' --masked_data
python -m explainableAV.probes.probing --model_name 'LUAR' --pair_type 'DS' --data_path 'explainableAV/Amazon/amazon_lda_DS_asterisk_False_False.json' --masked_data
python -m explainableAV.probes.probing --model_name 'LUAR' --pair_type 'DD' --data_path 'explainableAV/Amazon/amazon_lda_DD_asterisk_False_False.json' --masked_data
# Repeat for ModernBERT and StyleDistance
```
Results are stored in explainableAV/results/probing_metrics/
Results of the probing losses are stored in explainableAV/results/probing_losses/

### Plots
```sh
# Probing accuracy line plot
python -m explainableAV.results.probing_metrics.plot --plot_type 'probing_line_plot'
# Additionally, you can manually set the paths to your results when using different names through: --luar_results_path, --modernbert_results_path, --styledistance_results_path

# Probing heatmaps
python -m explainableAV.results.probing_metrics.plot --plot_type 'heatmap' --model_name 'LUAR'
python -m explainableAV.results.probing_metrics.plot --plot_type 'heatmap' --model_name 'ModernBERT'
python -m explainableAV.results.probing_metrics.plot --plot_type 'heatmap' --model_name 'StyleDistance'
python -m explainableAV.results.probing_metrics.plot --plot_type 'heatmap_f1'

# probing learning curve
python -m explainableAV.results.probing_losses.plot --plot_type 'probing_learning_curve' --model_name 'LUAR'
python -m explainableAV.results.probing_losses.plot --plot_type 'probing_learning_curve' --model_name 'ModernBERT'
python -m explainableAV.results.probing_losses.plot --plot_type 'probing_learning_curve' --model_name 'StyleDistance'
```
