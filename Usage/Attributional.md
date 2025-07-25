# Attributional (Attention)
To create the smaller datasets based on the most (and least) influenced datapoints, run the following:
```sh
python -m explainableAV.data_prep.data_split_attention --data_name 'amazon' --experiment 'both' --mask_type 'asterisk' # arguments used in thesis, but can be altered
```
The subdatasets are stored in explainableAV/Amazon/

## Faithfulness Evaluation
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
The results are stored in explainableAV/results/attention

## Attention Scores for Topic Words
```sh
python -m explainableAV.attention.attention --data_path 'explainableAV/Amazon/attention_most_influence.json' --attention_type 'rollout' --model_name 'LUAR' --topic_words_attention
python -m explainableAV.attention.attention --data_path 'explainableAV/Amazon/attention_most_influence.json' --attention_type 'value_zeroing' --model_name 'ModernBERT' --topic_words_attention 
python -m explainableAV.attention.attention --data_path 'explainableAV/Amazon/attention_most_influence.json' --attention_type 'raw' --model_name 'StyleDistance' --topic_words_attention 
```
Results are stored in explainableAV/results/attention

### Plots
```sh
python -m explainableAV.results.attention.plot

# Additionally, you can manually set the paths to your results when using different names through: --luar_results_path, --modernbert_results_path, --styledistance_results_path
```
Plots are stored in explainableAV/results/attention

## Attention Distribution

### Plots
```sh
python -m explainableAV.attention.attention --data_path 'explainableAV/Amazon/attention_most_influence.json' --attention_type 'value_zeroing' --model_name 'LUAR' --plot_type 'text_plot' --datapoint 246
python -m explainableAV.attention.attention --data_path 'explainableAV/Amazon/attention_most_influence.json' --attention_type 'value_zeroing' --model_name 'ModernBERT' --plot_type 'text_plot' --datapoint 246
python -m explainableAV.attention.attention --data_path 'explainableAV/Amazon/attention_most_influence.json' --attention_type 'value_zeroing' --model_name 'StyleDistance' --plot_type 'text_plot' --datapoint 246
python -m explainableAV.attention.attention --data_path 'explainableAV/Amazon/attention_most_influence.json' --attention_type 'value_zeroing' --model_name 'LUAR' --plot_type 'text_plot' --datapoint 338
python -m explainableAV.attention.attention --data_path 'explainableAV/Amazon/attention_most_influence.json' --attention_type 'value_zeroing' --model_name 'ModernBERT' --plot_type 'text_plot' --datapoint 338
python -m explainableAV.attention.attention --data_path 'explainableAV/Amazon/attention_most_influence.json' --attention_type 'value_zeroing' --model_name 'StyleDistance' --plot_type 'text_plot' --datapoint 338
```
The plots are stored in explainableAV/results/attention

## Attention Ablation
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
