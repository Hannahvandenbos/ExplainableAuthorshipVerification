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
The evaluation results are stored in: explainableAV/extract_topic/{data_name}_evaluate_mask_percentage.json and explainableAV/extract_topic/{data_name}_evaluate_inter_topic_distance.json

### Results
The evaluation plot can be plotted by using the following command:
```sh
# Amazon
python -m explainableAV.extract_topic.guided_lda_evaluation --data_path "explainableAV/Amazon/test_set_15000x4.json" --data_name 'amazon' --plot

# PAN20
python -m explainableAV.extract_topic.guided_lda_evaluation --data_path "explainableAV/PAN20/test_set_2500x4.json" --data_name 'pan20' --plot
```
The plot is stored in explainableAV/extract_topic

## Perturbations
To create the perturbed texts (Asterisk, POS tag, One words, and Swap), for single-sided perturbation, you can run the following commands for the SS test set on the Amazon data: 
```sh
# Amazon data, asterisk, single-sided perturbation
python -m explainableAV.change_topic.mask_words --data_path "explainableAV/Amazon/SS_test.json" --topic_related_path "explainableAV/extract_topic/amazon_topic_related_8400_filtered.json" --mask_type "asterisk" --save --mask_one_text --pair_type 'SS'
python -m explainableAV.change_topic.mask_words --data_path "explainableAV/Amazon/SD_test.json" --topic_related_path "explainableAV/extract_topic/amazon_topic_related_8400_filtered.json" --mask_type "asterisk" --save --mask_one_text --pair_type 'SD'
python -m explainableAV.change_topic.mask_words --data_path "explainableAV/Amazon/DS_test.json" --topic_related_path "explainableAV/extract_topic/amazon_topic_related_8400_filtered.json" --mask_type "asterisk" --save --mask_one_text --pair_type 'DS'
python -m explainableAV.change_topic.mask_words --data_path "explainableAV/Amazon/DD_test.json" --topic_related_path "explainableAV/extract_topic/amazon_topic_related_8400_filtered.json" --mask_type "asterisk" --save --mask_one_text --pair_type 'DD'

# Amazon data, pos tag, single-sided perturbation
python -m explainableAV.change_topic.mask_words --data_path "explainableAV/Amazon/SS_test.json" --topic_related_path "explainableAV/extract_topic/amazon_topic_related_8400_filtered.json" --mask_type "pos tag" --save --mask_one_text --pair_type 'SS'
python -m explainableAV.change_topic.mask_words --data_path "explainableAV/Amazon/SD_test.json" --topic_related_path "explainableAV/extract_topic/amazon_topic_related_8400_filtered.json" --mask_type "pos tag" --save --mask_one_text --pair_type 'SD'
python -m explainableAV.change_topic.mask_words --data_path "explainableAV/Amazon/DS_test.json" --topic_related_path "explainableAV/extract_topic/amazon_topic_related_8400_filtered.json" --mask_type "pos tag" --save --mask_one_text --pair_type 'DS'
python -m explainableAV.change_topic.mask_words --data_path "explainableAV/Amazon/DD_test.json" --topic_related_path "explainableAV/extract_topic/amazon_topic_related_8400_filtered.json" --mask_type "pos tag" --save --mask_one_text --pair_type 'DD'

# Amazon data, one word, single-sided perturbation
python -m explainableAV.change_topic.mask_words --data_path "explainableAV/Amazon/SS_test.json" --topic_related_path "explainableAV/extract_topic/amazon_topic_related_8400_filtered.json" --mask_type "one word" --save --mask_one_text --pair_type 'SS'
python -m explainableAV.change_topic.mask_words --data_path "explainableAV/Amazon/SD_test.json" --topic_related_path "explainableAV/extract_topic/amazon_topic_related_8400_filtered.json" --mask_type "one word" --save --mask_one_text --pair_type 'SD'
python -m explainableAV.change_topic.mask_words --data_path "explainableAV/Amazon/DS_test.json" --topic_related_path "explainableAV/extract_topic/amazon_topic_related_8400_filtered.json" --mask_type "one word" --save --mask_one_text --pair_type 'DS'
python -m explainableAV.change_topic.mask_words --data_path "explainableAV/Amazon/DD_test.json" --topic_related_path "explainableAV/extract_topic/amazon_topic_related_8400_filtered.json" --mask_type "one word" --save --mask_one_text --pair_type 'DD'

# Amazon data, swap, single-sided perturbation
python -m explainableAV.change_topic.mask_words --data_path "explainableAV/Amazon/SS_test.json" --topic_related_path "explainableAV/extract_topic/amazon_topic_related_8400_filtered.json" --mask_type "change topic" --save --mask_one_text --different --pair_type 'SS'
python -m explainableAV.change_topic.mask_words --data_path "explainableAV/Amazon/SD_test.json" --topic_related_path "explainableAV/extract_topic/amazon_topic_related_8400_filtered.json" --mask_type "change topic" --save --mask_one_text --pair_type 'SD'
python -m explainableAV.change_topic.mask_words --data_path "explainableAV/Amazon/DS_test.json" --topic_related_path "explainableAV/extract_topic/amazon_topic_related_8400_filtered.json" --mask_type "change topic" --save --mask_one_text --different --pair_type 'DS'
python -m explainableAV.change_topic.mask_words --data_path "explainableAV/Amazon/DD_test.json" --topic_related_path "explainableAV/extract_topic/amazon_topic_related_8400_filtered.json" --mask_type "change topic" --save --mask_one_text --pair_type 'DD'
```
For **dual perturbation** remove --mask_one_text. For swap, there is no dual perturbation.
To create the **perturbation-specific baselines** for the behavioral experiment, add --baseline to each command

To run on the **PAN20 data**, replace the --data_path with "explainableAV/PAN20/XX_test.json" with XX the corresponding pair type file and replace --topic_related_path with "explainableAV/extract_topic/pan20_topic_related_all_nouns_filtered.json" and add the argument --data_name "pan20"

The datasets are stored in explainableAV/change_topic/Amazon, explainableAV/change_topic/PAN20, explainableAV/change_topic/Amazon_baseline, explainableAV/change_topic/PAN20_baseline


To create the LLM perturbation (Amazon only), run:
```sh
python -m explainableAV.change_topic.llm_perturbations --data_path "explainableAV/Amazon/SS_test.json" --save "explainableAV/change_topic/Amazon/amazon_llama_SS.json"
python -m explainableAV.change_topic.llm_perturbations --data_path "explainableAV/Amazon/SD_test.json" --save "explainableAV/change_topic/Amazon/amazon_llama_SD.json"
python -m explainableAV.change_topic.llm_perturbations --data_path "explainableAV/Amazon/DS_test.json" --save "explainableAV/change_topic/Amazon/amazon_llama_DS.json"
python -m explainableAV.change_topic.llm_perturbations --data_path "explainableAV/Amazon/DD_test.json" --save "explainableAV/change_topic/Amazon/amazon_llama_DD.json"

# To clean the perturbation afterwards (remove some artifacts from LLMs), run:
python -m explainableAV.change_topic.llm_clean --llm_data_path "explainableAV/change_topic/Amazon/amazon_llama_SS.json" --original_data_path "explainableAV/Amazon/SS_test.json" --save "explainableAV/change_topic/Amazon/amazon_llama_SS_cleaned.json"
python -m explainableAV.change_topic.llm_clean --llm_data_path "explainableAV/change_topic/Amazon/amazon_llama_SD.json" --original_data_path "explainableAV/Amazon/SS_test.json" --save "explainableAV/change_topic/Amazon/amazon_llama_SD_cleaned.json"
python -m explainableAV.change_topic.llm_clean --llm_data_path "explainableAV/change_topic/Amazon/amazon_llama_DS.json" --original_data_path "explainableAV/Amazon/SS_test.json" --save "explainableAV/change_topic/Amazon/amazon_llama_DS_cleaned.json"
python -m explainableAV.change_topic.llm_clean --llm_data_path "explainableAV/change_topic/Amazon/amazon_llama_DD.json" --original_data_path "explainableAV/Amazon/SS_test.json" --save "explainableAV/change_topic/Amazon/amazon_llama_DD_cleaned.json"
```
The datasets are stored in explainableAV/change_topic/Amazon

## Perturbation Quality
To compute the mask quality, run commands like the following:
```sh
# Amazon data, Asterisk
python -m explainableAV.change_topic.mask_quality --data_path_SS "explainableAV/Amazon/SS_test.json" --data_path_SD "explainableAV/Amazon/SD_test.json" --data_path_DS "explainableAV/Amazon/DS_test.json" --data_path_DD "explainableAV/Amazon/DD_test.json" --masked_data_path_SS "explainableAV/change_topic/Amazon/amazon_lda_SS_asterisk_True_False.json" --masked_data_path_SD "explainableAV/change_topic/Amazon/amazon_lda_SD_asterisk_True_False.json" --masked_data_path_DS "explainableAV/change_topic/Amazon/amazon_lda_DS_asterisk_True_False.json" --masked_data_path_DD "explainableAV/change_topic/Amazon/amazon_lda_DD_asterisk_True_False.json" --mask_one_text --mask_type 'asterisk'

# Amazon data, Pos tag
python -m explainableAV.change_topic.mask_quality --data_path_SS "explainableAV/Amazon/SS_test.json" --data_path_SD "explainableAV/Amazon/SD_test.json" --data_path_DS "explainableAV/Amazon/DS_test.json" --data_path_DD "explainableAV/Amazon/DD_test.json" --masked_data_path_SS "explainableAV/change_topic/Amazon/amazon_lda_SS_pos tag_True_False.json" --masked_data_path_SD "explainableAV/change_topic/Amazon/amazon_lda_SD_pos tag_True_False.json" --masked_data_path_DS "explainableAV/change_topic/Amazon/amazon_lda_DS_pos tag_True_False.json" --masked_data_path_DD "explainableAV/change_topic/Amazon/amazon_lda_DD_pos tag_True_False.json" --mask_one_text --mask_type 'pos tag'

# Amazon data, One word
python -m explainableAV.change_topic.mask_quality --data_path_SS "explainableAV/Amazon/SS_test.json" --data_path_SD "explainableAV/Amazon/SD_test.json" --data_path_DS "explainableAV/Amazon/DS_test.json" --data_path_DD "explainableAV/Amazon/DD_test.json" --masked_data_path_SS "explainableAV/change_topic/Amazon/amazon_lda_SS_one word_True_False.json" --masked_data_path_SD "explainableAV/change_topic/Amazon/amazon_lda_SD_one word_True_False.json" --masked_data_path_DS "explainableAV/change_topic/Amazon/amazon_lda_DS_one word_True_False.json" --masked_data_path_DD "explainableAV/change_topic/Amazon/amazon_lda_DD_one word_True_False.json" --mask_one_text --mask_type 'one word'

# Amazon data, Swap
python -m explainableAV.change_topic.mask_quality --data_path_SS "explainableAV/Amazon/SS_test.json" --data_path_SD "explainableAV/Amazon/SD_test.json" --data_path_DS "explainableAV/Amazon/DS_test.json" --data_path_DD "explainableAV/Amazon/DD_test.json" --masked_data_path_SS "explainableAV/change_topic/Amazon/amazon_lda_SS_change topic_True_True.json" --masked_data_path_SD "explainableAV/change_topic/Amazon/amazon_lda_SD_change topic_True_False.json" --masked_data_path_DS "explainableAV/change_topic/Amazon/amazon_lda_DS_change topic_True_True.json" --masked_data_path_DD "explainableAV/change_topic/Amazon/amazon_lda_DD_change topic_True_False.json" --mask_one_text --mask_type 'change topic'

# Amazon data, LLM
python -m explainableAV.change_topic.mask_quality --data_path_SS "explainableAV/Amazon/SS_test.json" --data_path_SD "explainableAV/Amazon/SD_test.json" --data_path_DS "explainableAV/Amazon/DS_test.json" --data_path_DD "explainableAV/Amazon/DD_test.json" --masked_data_path_SS "explainableAV/change_topic/Amazon/amazon_llama_SS_cleaned.json" --masked_data_path_SD "explainableAV/change_topic/Amazon/amazon_llama_SD_cleaned.json" --masked_data_path_DS "explainableAV/change_topic/Amazon/amazon_llama_DS_cleaned.json" --masked_data_path_DD "explainableAV/change_topic/Amazon/amazon_llama_DD_cleaned.json" --mask_one_text --mask_type 'llm'
```
Replace the files to match the PAN20 dataset.
--mask_one_text ensures a fair comparison between all perturbation techniques, but can be removed for 'Asterisk', 'POS tag', and 'One word'.
The results are stored in explainableAV/change_topic

### Plot
To plot the results from the perturbation quality, run the following:
```sh
# Amazon
python -m explainableAV.change_topic.perturbation_quality_plot

# PAN20
python -m explainableAV.change_topic.perturbation_quality_plot --dataset_name "pan20"
```
The plot is stored in explainableAV/change_topic

