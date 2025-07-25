# Concept-Based (Probing)
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
Repeat the fine-tuned masked experiments for ModernBERT and StyleDistance
Results are stored in explainableAV/results/probing_metrics/
Results of the probing losses are stored in explainableAV/results/probing_losses

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
Plots are stored in explainableAV/results/probing_metrics
