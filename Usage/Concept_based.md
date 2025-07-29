# Concept-Based (Probing)
This file contains information on how to run the concept-based experiments, which consists of probing with slightly different models or data.

## Probing
To probe the hidden states of the model, run:
```sh
# Fine-tuned test (including cross-validation)
python -m explainableAV.probes.probing --model_name 'LUAR' --data_path 'explainableAV/Amazon/test_set_15000x4.json'
python -m explainableAV.probes.probing --model_name 'ModernBERT' --data_path 'explainableAV/Amazon/test_set_15000x4.json'
python -m explainableAV.probes.probing --model_name 'StyleDistance' --data_path 'explainableAV/Amazon/test_set_15000x4.json'

# Pre-trained test
python -m explainableAV.probes.probing --model_name 'LUAR' --data_path 'explainableAV/Amazon/test_set_15000x4.json' --pretrained_model
python -m explainableAV.probes.probing --model_name 'ModernBERT' --data_path 'explainableAV/Amazon/test_set_15000x4.json' --pretrained_model
python -m explainableAV.probes.probing --model_name 'StyleDistance' --data_path 'explainableAV/Amazon/test_set_15000x4.json' --pretrained_model

# Fine-tuned masked
python -m explainableAV.probes.probing --model_name 'LUAR' --data_path 'explainableAV/change_topic/Amazon/amazon_lda_SS_asterisk_False_False.json' --masked_data
python -m explainableAV.probes.probing --model_name 'LUAR' --data_path 'explainableAV/change_topic/Amazon/amazon_lda_SD_asterisk_False_False.json' --masked_data
python -m explainableAV.probes.probing --model_name 'LUAR' --data_path 'explainableAV/change_topic/Amazon/amazon_lda_DS_asterisk_False_False.json' --masked_data
python -m explainableAV.probes.probing --model_name 'LUAR' --data_path 'explainableAV/change_topic/Amazon/amazon_lda_DD_asterisk_False_False.json' --masked_data

python -m explainableAV.probes.probing --model_name 'ModernBERT' --data_path 'explainableAV/change_topic/Amazon/amazon_lda_SS_asterisk_False_False.json' --masked_data
python -m explainableAV.probes.probing --model_name 'ModernBERT' --data_path 'explainableAV/change_topic/Amazon/amazon_lda_SD_asterisk_False_False.json' --masked_data
python -m explainableAV.probes.probing --model_name 'ModernBERT' --data_path 'explainableAV/change_topic/Amazon/amazon_lda_DS_asterisk_False_False.json' --masked_data
python -m explainableAV.probes.probing --model_name 'ModernBERT' --data_path 'explainableAV/change_topic/Amazon/amazon_lda_DD_asterisk_False_False.json' --masked_data

python -m explainableAV.probes.probing --model_name 'StyleDistance' --data_path 'explainableAV/change_topic/Amazon/amazon_lda_SS_asterisk_False_False.json' --masked_data
python -m explainableAV.probes.probing --model_name 'StyleDistance' --data_path 'explainableAV/change_topic/Amazon/amazon_lda_SD_asterisk_False_False.json' --masked_data
python -m explainableAV.probes.probing --model_name 'StyleDistance' --data_path 'explainableAV/change_topic/Amazon/amazon_lda_DS_asterisk_False_False.json' --masked_data
python -m explainableAV.probes.probing --model_name 'StyleDistance' --data_path 'explainableAV/change_topic/Amazon/amazon_lda_DD_asterisk_False_False.json' --masked_data
```
*If you use a different size test set, change --data_path to .../test_set_yoursizex4.json where yoursize corresponds with the test size of one pair type in your data for Fine-tuned test and Pre-trained test.* \
Repeat the fine-tuned masked experiments for ModernBERT and StyleDistance \
Results are stored as: \
explainableAV/results/probing_metrics/probing_metrics_LUAR.json \
explainableAV/results/probing_metrics/probing_metrics_ModernBERT.json \
explainableAV/results/probing_metrics/probing_metrics_StyleDistance.json \
Results of the probing losses are stored in explainableAV/results/probing_losses

The probing_metrics files follow the following structure:
```sh
{
    Metric: { # Precision, Recall, F1-scores, rollout, value_zeroing
        Layer_number: }
}
```

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
python -m explainableAV.results.probing_losses.plot --model_name 'LUAR' --results_path 'explainableAV/results/probing_losses/probing_losses_LUAR.json'
python -m explainableAV.results.probing_losses.plot --model_name 'ModernBERT' --results_path 'explainableAV/results/probing_losses/probing_losses_ModernBERT.json'
python -m explainableAV.results.probing_losses.plot --model_name 'StyleDistance' --results_path 'explainableAV/results/probing_losses/probing_losses_StyleDistance.json'
```
Plots are stored as: \
explainableAV/results/probing_metrics/probing_accuracy.pdf \
explainableAV/results/probing_metrics/heatmaps_probing_LUAR.pdf \
explainableAV/results/probing_metrics/heatmaps_probing_ModernBERT.pdf \
explainableAV/results/probing_metrics/heatmaps_probing_StyleDistance.pdf \
explainableAV/results/probing_metrics/heatmaps_probing_f1_only.pdf
