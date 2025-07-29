# Attributional (Attention)
This file contains information on how to run the attributional experiments from the thesis.

## Creating the datasets
To create the smaller datasets based on the most (and least) influenced datapoints in the behavioral experiments, run the following:
```sh
python -m explainableAV.data_prep.data_split_attention --data_name 'amazon' --experiment 'both' --mask_type 'asterisk' # arguments used in thesis, but can be altered
```
The subdatasets are stored as: \
explainableAV/Amazon/attention_most_influence.json \
explainableAV/Amazon/attention_most_influence_asterisk.json \
explainableAV/Amazon/attention_least_influence.json \
explainableAV/Amazon/attention_least_influence_asterisk.json

## Faithfulness Evaluation
The faithfulness evaluation uses the most influenced datapoints and measures the faithfulness for raw attention, attention rollout, and value zeroing for each of the AV models.
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
The results are stored as explainableAV/results/attention_faithfulness.json with the following structure:
```sh
{
    Attention technique: { # raw, rollout, value_zeroing
        Model: { # LUAR, ModernBERT, StyleDistance
            Faithfulness metric: { # comp, incomp, suff, insuff
                Top-k: { # 0.01, 0.1, 0.25
                    0:
                    1: }
                }
            }
        }
    }
}
```

## Attention Scores for Topic Words
To obtain the attention scores (relative topic-attention ratio and topic coverage), run the following:
```sh
python -m explainableAV.attention.attention --data_path 'explainableAV/Amazon/attention_most_influence.json' --attention_type 'rollout' --model_name 'LUAR' --topic_words_attention
python -m explainableAV.attention.attention --data_path 'explainableAV/Amazon/attention_most_influence.json' --attention_type 'value_zeroing' --model_name 'ModernBERT' --topic_words_attention 
python -m explainableAV.attention.attention --data_path 'explainableAV/Amazon/attention_most_influence.json' --attention_type 'raw' --model_name 'StyleDistance' --topic_words_attention 
```
Results are stored as: \
explainableAV/results/attention_top_LUAR_rollout.json \
explainableAV/results/attention_top_ModernBERT_value_zeroing.json \
explainableAV/results/attention_top_StyleDistance_raw.json

The files have the following structure:
```sh
{
    top-k: [[...]]
    ratio: [[...]]
}
```

### Plots
To plot the topic coverage and relative topic-attention ratio
```sh
python -m explainableAV.results.attention.plot

# Additionally, you can manually set the paths to your results when using different names through: --luar_results_path, --modernbert_results_path, --styledistance_results_path
```
Plots are stored as explainableAV/results/attention/top_k.pdf and explainableAV/results/attention/top_ratio.pdf

## Attention Distribution
The attention distribution is immediately plotted as the original text with attention scores highlighted.

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
For the attention ablation, this includes all necessary commands:
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
