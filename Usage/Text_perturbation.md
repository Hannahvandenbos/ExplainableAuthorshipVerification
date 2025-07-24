# Text Perturbations
In order to perturb the texts, we first have to extract topic words with Guided LDA, and then perturb the texts in various ways.

## Extract Topic Words
In order to perturb the texts, we first have to extract the topic words with Guided lda. If you want to use the topic words as used in the thesis, do:
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
The dataset with topic words is stored in explainableAV/extract_topic

To evaluate multiple numbers of topic words with Guided LDA run:
```sh
# Amazon
python -m explainableAV.extract_topic.guided_lda --data_path "explainableAV/Amazon/test_set_15000x4.json" --evaluate
python -m explainableAV.extract_topic.guided_lda_evaluation --data_path "explainableAV/Amazon/test_set_15000x4.json" --data_name 'amazon' --evaluate_masks --evaluate # evaluation
python -m explainableAV.extract_topic.guided_lda_evaluation --data_path "explainableAV/Amazon/test_set_15000x4.json" --data_name 'amazon' --inter_distance --evaluate # evaluation

# PAN20
python -m explainableAV.extract_topic.guided_lda --data_path "explainableAV/PAN20/test_set_2500x4.json" --data_name 'pan20' --evaluate
python -m explainableAV.extract_topic.guided_lda_evaluation --data_path "explainableAV/PAN20/test_set_2500x4.json" --data_name 'pan20' --inter_distance --evaluate # evaluation
python -m explainableAV.extract_topic.guided_lda_evaluation --data_path "explainableAV/PAN20/test_set_2500x4.json" --data_name 'pan20' --evaluate_masks --evaluate # evaluation
```
The datasets with topic words are stored in explainableAV/extract_topic
The evaluation results are stored in: 

