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
