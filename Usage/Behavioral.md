# Behavioral (Input-Output Relations)
This file provides information of how to run all experiments concerning the behavioral experiments of the thesis. This includes the models' baseline performance, and performance on the perturbed texts with and without random perturbations.

## Baseline Model Performance
To get the original/baseline model performance, run the following:
```sh
# LUAR model, Amazon data
python -m explainableAV.models.test --data_path "explainableAV/Amazon/SS_test.json" --model_name "LUAR" --data_split "SS" --dataset_name "amazon"
python -m explainableAV.models.test --data_path "explainableAV/Amazon/SD_test.json" --model_name "LUAR" --data_split "SD" --dataset_name "amazon"
python -m explainableAV.models.test --data_path "explainableAV/Amazon/DS_test.json" --model_name "LUAR" --data_split "DS" --dataset_name "amazon"
python -m explainableAV.models.test --data_path "explainableAV/Amazon/DD_test.json" --model_name "LUAR" --data_split "DD" --dataset_name "amazon"

# ModernBERT model, Amazon data
python -m explainableAV.models.test --data_path "explainableAV/Amazon/SS_test.json" --model_name "ModernBERT" --data_split "SS" --dataset_name "amazon"
python -m explainableAV.models.test --data_path "explainableAV/Amazon/SD_test.json" --model_name "ModernBERT" --data_split "SD" --dataset_name "amazon"
python -m explainableAV.models.test --data_path "explainableAV/Amazon/DS_test.json" --model_name "ModernBERT" --data_split "DS" --dataset_name "amazon"
python -m explainableAV.models.test --data_path "explainableAV/Amazon/DD_test.json" --model_name "ModernBERT" --data_split "DD" --dataset_name "amazon"

# StyleDistance model, Amazon data
python -m explainableAV.models.test --data_path "explainableAV/Amazon/SS_test.json" --model_name "StyleDistance" --data_split "SS" --dataset_name "amazon"
python -m explainableAV.models.test --data_path "explainableAV/Amazon/SD_test.json" --model_name "StyleDistance" --data_split "SD" --dataset_name "amazon"
python -m explainableAV.models.test --data_path "explainableAV/Amazon/DS_test.json" --model_name "StyleDistance" --data_split "DS" --dataset_name "amazon"
python -m explainableAV.models.test --data_path "explainableAV/Amazon/DD_test.json" --model_name "StyleDistance" --data_split "DD" --dataset_name "amazon"
```
Set --threshold if your threshold differs from the one in the thesis. \
Replace the dataset_name to obtain all results for **PAN20**. \
Additionally, run all commands again with --perturb_second to ensure that future results can be computed for **dual perturbations**. \
Results are stored in: \
explainableAV/results/predictions/amazon_LUAR_predictions_mask_both.json for dual perturbation \
explainableAV/results/predictions/amazon_LUAR_predictions_mask_first.json for single-sided perturbation \
explainableAV/results/predictions/amazon_ModernBERT_predictions_mask_both.json for dual perturbation \
explainableAV/results/predictions/amazon_ModernBERT_predictions_mask_first.json for single-sided perturbation \
explainableAV/results/predictions/amazon_StyleDistance_predictions_mask_both.json for dual perturbation \
explainableAV/results/predictions/amazon_StyleDistance_predictions_mask_first.json for single-sided perturbation

explainableAV/results/predictions/pan20_LUAR_predictions_mask_both.json for dual perturbation \
explainableAV/results/predictions/pan20_LUAR_predictions_mask_first.json for single-sided perturbation \
explainableAV/results/predictions/pan20_ModernBERT_predictions_mask_both.json for dual perturbation \
explainableAV/results/predictions/pan20_ModernBERT_predictions_mask_first.json for single-sided perturbation \
explainableAV/results/predictions/pan20_StyleDistance_predictions_mask_both.json for dual perturbation \
explainableAV/results/predictions/pan20_StyleDistance_predictions_mask_first.json for single-sided perturbation

## Perturbed Texts
To obtain the results for the perturbed texts, run the same file with different arguments, ensure you have perturbed texts and created the perturbation-specific baselines before running this:
```sh
# Amazon data, LUAR model, Asterisk, single-sided perturbation
python -m explainableAV.models.test --data_path "explainableAV/Amazon/SS_test.json" --extra_data_path "explainableAV/change_topic/Amazon/amazon_lda_SS_asterisk_True_False.json" --model_name "LUAR" --mask_type 'asterisk' --data_split "SS" --dataset_name "amazon" 
python -m explainableAV.models.test --data_path "explainableAV/Amazon/SD_test.json" --extra_data_path "explainableAV/change_topic/Amazon/amazon_lda_SD_asterisk_True_False.json" --model_name "LUAR" --mask_type 'asterisk' --data_split "SD" --dataset_name "amazon" 
python -m explainableAV.models.test --data_path "explainableAV/Amazon/DS_test.json" --extra_data_path "explainableAV/change_topic/Amazon/amazon_lda_DS_asterisk_True_False.json" --model_name "LUAR" --mask_type 'asterisk' --data_split "DS" --dataset_name "amazon" 
python -m explainableAV.models.test --data_path "explainableAV/Amazon/DD_test.json" --extra_data_path "explainableAV/change_topic/Amazon/amazon_lda_DD_asterisk_True_False.json" --model_name "LUAR" --mask_type 'asterisk' --data_split "DD" --dataset_name "amazon" 

# Amazon data, LUAR model, POS tag, single-sided perturbation
python -m explainableAV.models.test --data_path "explainableAV/Amazon/SS_test.json" --extra_data_path "explainableAV/change_topic/Amazon/amazon_lda_SS_pos tag_True_False.json" --model_name "LUAR" --mask_type 'pos tag' --data_split "SS" --dataset_name "amazon" 
python -m explainableAV.models.test --data_path "explainableAV/Amazon/SD_test.json" --extra_data_path "explainableAV/change_topic/Amazon/amazon_lda_SD_pos tag_True_False.json" --model_name "LUAR" --mask_type 'pos tag' --data_split "SD" --dataset_name "amazon" 
python -m explainableAV.models.test --data_path "explainableAV/Amazon/DS_test.json" --extra_data_path "explainableAV/change_topic/Amazon/amazon_lda_DS_pos tag_True_False.json" --model_name "LUAR" --mask_type 'pos tag' --data_split "DS" --dataset_name "amazon" 
python -m explainableAV.models.test --data_path "explainableAV/Amazon/DD_test.json" --extra_data_path "explainableAV/change_topic/Amazon/amazon_lda_DD_pos tag_True_False.json" --model_name "LUAR" --mask_type 'pos tag' --data_split "DD" --dataset_name "amazon" 

# Amazon data, LUAR model, One word, single-sided perturbation
python -m explainableAV.models.test --data_path "explainableAV/Amazon/SS_test.json" --extra_data_path "explainableAV/change_topic/Amazon/amazon_lda_SS_one word_True_False.json" --model_name "LUAR" --mask_type 'one word' --data_split "SS" --dataset_name "amazon" 
python -m explainableAV.models.test --data_path "explainableAV/Amazon/SD_test.json" --extra_data_path "explainableAV/change_topic/Amazon/amazon_lda_SD_one word_True_False.json" --model_name "LUAR" --mask_type 'one word' --data_split "SD" --dataset_name "amazon" 
python -m explainableAV.models.test --data_path "explainableAV/Amazon/DS_test.json" --extra_data_path "explainableAV/change_topic/Amazon/amazon_lda_DS_one word_True_False.json" --model_name "LUAR" --mask_type 'one word' --data_split "DS" --dataset_name "amazon" 
python -m explainableAV.models.test --data_path "explainableAV/Amazon/DD_test.json" --extra_data_path "explainableAV/change_topic/Amazon/amazon_lda_DD_one word_True_False.json" --model_name "LUAR" --mask_type 'one word' --data_split "DD" --dataset_name "amazon" 

# Amazon data, LUAR model, Change topic, single-sided perturbation
python -m explainableAV.models.test --data_path "explainableAV/Amazon/SS_test.json" --extra_data_path "explainableAV/change_topic/Amazon/amazon_lda_SS_change topic_True_True.json" --model_name "LUAR" --mask_type 'change topic' --data_split "SS" --dataset_name "amazon" 
python -m explainableAV.models.test --data_path "explainableAV/Amazon/SD_test.json" --extra_data_path "explainableAV/change_topic/Amazon/amazon_lda_SD_change topic_True_False.json" --model_name "LUAR" --mask_type 'change topic' --data_split "SD" --dataset_name "amazon" 
python -m explainableAV.models.test --data_path "explainableAV/Amazon/DS_test.json" --extra_data_path "explainableAV/change_topic/Amazon/amazon_lda_DS_change topic_True_True.json" --model_name "LUAR" --mask_type 'change topic' --data_split "DS" --dataset_name "amazon" 
python -m explainableAV.models.test --data_path "explainableAV/Amazon/DD_test.json" --extra_data_path "explainableAV/change_topic/Amazon/amazon_lda_DD_change topic_True_False.json" --model_name "LUAR" --mask_type 'change topic' --data_split "DD" --dataset_name "amazon" 

# Amazon data, LUAR model, Asterisk, single-sided perturbation
python -m explainableAV.models.test --data_path "explainableAV/Amazon/SS_test.json" --extra_data_path "explainableAV/change_topic/Amazon/amazon_llama_SS_cleaned.json" --model_name "LUAR" --mask_type 'llm' --data_split "SS" --dataset_name "amazon" 
python -m explainableAV.models.test --data_path "explainableAV/Amazon/SD_test.json" --extra_data_path "explainableAV/change_topic/Amazon/amazon_llama_SD_cleaned.json" --model_name "LUAR" --mask_type 'llm' --data_split "SD" --dataset_name "amazon" 
python -m explainableAV.models.test --data_path "explainableAV/Amazon/DS_test.json" --extra_data_path "explainableAV/change_topic/Amazon/amazon_llama_DS_cleaned.json" --model_name "LUAR" --mask_type 'llm' --data_split "DS" --dataset_name "amazon" 
python -m explainableAV.models.test --data_path "explainableAV/Amazon/DD_test.json" --extra_data_path "explainableAV/change_topic/Amazon/amazon_llama_DD_cleaned.json" --model_name "LUAR" --mask_type 'llm' --data_split "DD" --dataset_name "amazon" 



# Amazon data, LUAR model, Asterisk Baseline, single-sided perturbation
python -m explainableAV.models.test --data_path "explainableAV/Amazon/SS_test.json" --extra_data_path "explainableAV/change_topic/Amazon_baseline/amazon_new_baseline_lda_SS_asterisk_True_False.json" --model_name "LUAR" --mask_type 'asterisk_baseline' --data_split "SS" --dataset_name "amazon" 
python -m explainableAV.models.test --data_path "explainableAV/Amazon/SD_test.json" --extra_data_path "explainableAV/change_topic/Amazon_baseline/amazon_new_baseline_lda_SD_asterisk_True_False.json" --model_name "LUAR" --mask_type 'asterisk_baseline' --data_split "SD" --dataset_name "amazon" 
python -m explainableAV.models.test --data_path "explainableAV/Amazon/DS_test.json" --extra_data_path "explainableAV/change_topic/Amazon_baseline/amazon_new_baseline_lda_DS_asterisk_True_False.json" --model_name "LUAR" --mask_type 'asterisk_baseline' --data_split "DS" --dataset_name "amazon" 
python -m explainableAV.models.test --data_path "explainableAV/Amazon/DD_test.json" --extra_data_path "explainableAV/change_topic/Amazon_baseline/amazon_new_baseline_lda_DD_asterisk_True_False.json" --model_name "LUAR" --mask_type 'asterisk_baseline' --data_split "DD" --dataset_name "amazon" 

# Amazon data, LUAR model, POS tag Baseline, single-sided perturbation
python -m explainableAV.models.test --data_path "explainableAV/Amazon/SS_test.json" --extra_data_path "explainableAV/change_topic/Amazon_baseline/amazon_new_baseline_lda_SS_pos tag_True_False.json" --model_name "LUAR" --mask_type 'pos tag_baseline' --data_split "SS" --dataset_name "amazon" 
python -m explainableAV.models.test --data_path "explainableAV/Amazon/SD_test.json" --extra_data_path "explainableAV/change_topic/Amazon_baseline/amazon_new_baseline_lda_SD_pos tag_True_False.json" --model_name "LUAR" --mask_type 'pos tag_baseline' --data_split "SD" --dataset_name "amazon" 
python -m explainableAV.models.test --data_path "explainableAV/Amazon/DS_test.json" --extra_data_path "explainableAV/change_topic/Amazon_baseline/amazon_new_baseline_lda_DS_pos tag_True_False.json" --model_name "LUAR" --mask_type 'pos tag_baseline' --data_split "DS" --dataset_name "amazon" 
python -m explainableAV.models.test --data_path "explainableAV/Amazon/DD_test.json" --extra_data_path "explainableAV/change_topic/Amazon_baseline/amazon_new_baseline_lda_DD_pos tag_True_False.json" --model_name "LUAR" --mask_type 'pos tag_baseline' --data_split "DD" --dataset_name "amazon" 

# Amazon data, LUAR model, One word Baseline, single-sided perturbation
python -m explainableAV.models.test --data_path "explainableAV/Amazon/SS_test.json" --extra_data_path "explainableAV/change_topic/Amazon_baseline/amazon_new_baseline_lda_SS_one word_True_False.json" --model_name "LUAR" --mask_type 'one word_baseline' --data_split "SS" --dataset_name "amazon" 
python -m explainableAV.models.test --data_path "explainableAV/Amazon/SD_test.json" --extra_data_path "explainableAV/change_topic/Amazon_baseline/amazon_new_baseline_lda_SD_one word_True_False.json" --model_name "LUAR" --mask_type 'one word_baseline' --data_split "SD" --dataset_name "amazon" 
python -m explainableAV.models.test --data_path "explainableAV/Amazon/DS_test.json" --extra_data_path "explainableAV/change_topic/Amazon_baseline/amazon_new_baseline_lda_DS_one word_True_False.json" --model_name "LUAR" --mask_type 'one word_baseline' --data_split "DS" --dataset_name "amazon" 
python -m explainableAV.models.test --data_path "explainableAV/Amazon/DD_test.json" --extra_data_path "explainableAV/change_topic/Amazon_baseline/amazon_new_baseline_lda_DD_one word_True_False.json" --model_name "LUAR" --mask_type 'one word_baseline' --data_split "DD" --dataset_name "amazon" 

# Amazon data, LUAR model, Change topic Baseline, single-sided perturbation
python -m explainableAV.models.test --data_path "explainableAV/Amazon/SS_test.json" --extra_data_path "explainableAV/change_topic/Amazon_baseline/amazon_new_baseline_lda_SS_change topic_True_True.json" --model_name "LUAR" --mask_type 'change topic_baseline' --data_split "SS" --dataset_name "amazon" 
python -m explainableAV.models.test --data_path "explainableAV/Amazon/SD_test.json" --extra_data_path "explainableAV/change_topic/Amazon_baseline/amazon_new_baseline_lda_SD_change topic_True_False.json" --model_name "LUAR" --mask_type 'change topic_baseline' --data_split "SD" --dataset_name "amazon" 
python -m explainableAV.models.test --data_path "explainableAV/Amazon/DS_test.json" --extra_data_path "explainableAV/change_topic/Amazon_baseline/amazon_new_baseline_lda_DS_change topic_True_True.json" --model_name "LUAR" --mask_type 'change topic_baseline' --data_split "DS" --dataset_name "amazon" 
python -m explainableAV.models.test --data_path "explainableAV/Amazon/DD_test.json" --extra_data_path "explainableAV/change_topic/Amazon_baseline/amazon_new_baseline_lda_DD_change topic_True_False.json" --model_name "LUAR" --mask_type 'change topic_baseline' --data_split "DD" --dataset_name "amazon" 
```
To run for different models, replace 'LUAR' with 'ModernBERT' or 'StyleDistance' \
For **dual perturbation** add --perturb_second (only for Asterisk, POS tag, and One word) \
For **PAN20** alter the data paths and dataset_name \
Optionally, set your own threshold with --threshold \
Results are stored in explainableAV/results/predictions, in the same files as the baseline performance \
The files are structured as follows:
```sh
{
    Pair type: { # SS, SD, DS, or DD
        Perturbation technique: { # asterisk, pos tag, one word, swap, llm, original, asterisk_baseline, pos tag_baseline, one word_baseline, swap_baseline
            accuracy:
            predictions: [...]
            confidences: [...] }
        }
    }
}
```

### Plots
To plot the results of the behavioral experiment:
```sh
# Confusion plot, plotting the TPs, TNs, FPs, and FNs
python -m explainableAV.results.predictions.plot --plot_type 'confusion' --experiment 'first' --dataset_name 'amazon' --baseline # Confusion plot, single-sided perturbation, Amazon data
python -m explainableAV.results.predictions.plot --plot_type 'confusion' --experiment 'both' --dataset_name 'amazon' --baseline # Confusion plot, dual perturbation, Amazon data
python -m explainableAV.results.predictions.plot --plot_type 'confusion' --experiment 'first' --dataset_name 'pan20' --baseline # Confusion plot, single-sided perturbation, PAN20 data
python -m explainableAV.results.predictions.plot --plot_type 'confusion' --experiment 'both' --dataset_name 'pan20' --baseline # Confusion plot, dual perturbation, PAN20 data

# Heatmaps 
python -m explainableAV.results.predictions.plot --plot_type 'heatmaps' --experiment 'first' --dataset_name 'amazon' --baseline # Heatmap plot, single-sided perturbation, Amazon data
python -m explainableAV.results.predictions.plot --plot_type 'heatmaps' --experiment 'both' --dataset_name 'amazon' --baseline # Heatmap plot, dual perturbation, Amazon data
python -m explainableAV.results.predictions.plot --plot_type 'heatmaps' --experiment 'first' --dataset_name 'pan20' --baseline # Heatmap plot, single-sided perturbation, PAN20 data
python -m explainableAV.results.predictions.plot --plot_type 'heatmaps' --experiment 'both' --dataset_name 'pan20' --baseline # Heatmap plot, dual perturbation, PAN20 data

# Additionally, you can manually set the paths to your results when using different names through: --luar_results_path, --modernbert_results_path, --styledistance_results_path
```
Plots are stored i explainableAV/results/predictions
