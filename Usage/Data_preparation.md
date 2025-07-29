# Data Preparation
This file explains which data was used, how to preprocess the data, and how to create text pairs, ensuring applicability with the experiments. 
We used two datasets: Amazon reviews and Fanfictions from the PAN2020 competition.

## Amazon
The *Amazon Reviews* dataset needs to be downloaded from [Amazon Reviews](https://nijianmo.github.io/amazon/index.html) \
Download the 5-core files for the following categories and extract them to the explainableAV/Amazon folder:
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

### Data Preprocessing
Run the following command to aggregate the data files, set the topics, and filter the data:
```sh
# Get single reviews and set topic
python -m explainableAV.data_prep.reorder_Amazon
```
The processed data is stored in explainableAV/Amazon/amazon_reviews_final.json, consisting of the reviews with their AuthorID and Topic.

To create text pairs, run the following command:
```sh
python -m explainableAV.data_prep.create_pairs --dataset_path "explainableAV/Amazon/amazon_reviews_final.json" --SS_file_path "explainableAV/Amazon/SS.json" --SD_file_path "explainableAV/Amazon/SD.json" --DS_file_path "explainableAV/Amazon/DS.json" --DD_file_path "explainableAV/Amazon/DD.json"
```
This will create four datasets, one per pair type: \
explainableAV/Amazon/SS.json: contains all text pairs that are Same-author Same-topic \
explainableAV/Amazon/SD.json: contains all text pairs that are Same-author Different-topic \
explainableAV/Amazon/DS.json: contains all text pairs that are Different-author Same-topic \
explainableAV/Amazon/DD.json: contains all text pairs that are Different-author Different-topic \

A dataset entry has the following form:
```sh
entry = {
"Label": 0, # 0 for different-author, 1 for same-author
"Topics": (topic1, topic2),
"Pair": (text1, text2)
}
```

Finally, to create the train, test, and validation splits of the text pairs, run the following command:
```sh
python -m explainableAV.data_prep.data_split --samples_per_pair 15000 --SS_file_path "explainableAV/Amazon/SS.json" --SD_file_path "explainableAV/Amazon/SD.json" --DS_file_path "explainableAV/Amazon/DS.json" --DD_file_path "explainableAV/Amazon/DD.json"
```
*For a smaller or larger dataset, change --samples_per_pair to a different number as the commands above are specific to the full data.* \
This will create the following files: \
explainableAV/Amazon/SS_test.json&emsp;explainableAV/Amazon/SS_val.json&emsp;explainableAV/Amazon/SS_train.json \
explainableAV/Amazon/SD_test.json&emsp;explainableAV/Amazon/SD_val.json&emsp;explainableAV/Amazon/SD_train.json \
explainableAV/Amazon/DS_test.json&emsp;explainableAV/Amazon/DS_val.json&emsp;explainableAV/Amazon/DS_train.json \
explainableAV/Amazon/DD_test.json&emsp;explainableAV/Amazon/DD_val.json&emsp;explainableAV/Amazon/DD_train.json \
explainableAV/Amazon/test_set_15000x4.json&emsp;explainableAV/Amazon/val_set_15000x4.json&emsp;explainableAV/Amazon/train_set_15000x4.json \
Where the last line combines all pair types into one test, val or train dataset. \

### Results
To print an overview of the number of pairs per pair type in the data, run:
```sh
python -m explainableAV.data_prep.data_distributions
```

To print an overview of the number of pairs per split per pair type in the data, run:
```sh
python -m explainableAV.data_prep.data_distributions --statistic 'splits' --split_size 15000
```
*Split size should correspond with the test size of one pair type in your data.*

To plot the topic distributions of the train and test dataset, run the following:
```sh
python -m explainableAV.data_prep.data_distributions --statistic 'topic_distribution' --split_size 15000
```
*Split size should correspond with the test size of one pair type in your data.* \
The plot is stored as explainableAV/data_prep/Topic_distribution_Amazon_15000.pdf

## PAN20
The PAN20 dataset needs to be downloaded from [PAN20](https://zenodo.org/records/3724096) \
You can opt for both the large or small version. The small version was used in this research. \
Store the two jsonl files in the explainableAV/PAN20 folder.

### Data Preprocessing
Run the following commands to creating filtered single text entries with AuthorID, Text, and Topic:
```sh
# PAN20 (small)
python -m explainableAV.data_prep.reorder_PAN20 --texts_path "explainableAV/PAN20/.pan20-authorship-verification-training-small.jsonl" --label_path "explainableAV/PAN20/pan20-authorship-verification-training-small-truth.jsonl"

# PAN20 (large)
python -m explainableAV.reorder_PAN20 --texts_path "explainableAV/PAN20/pan20-authorship-verification-training-large.jsonl" --label_path "explainableAV/PAN20/pan20-authorship-verification-training-large-truth.jsonl"
```
The dataset is stored as explainableAV/PAN20/PAN20_filtered.json

To create text pairs, run the following command:
```sh
# PAN20
python -m explainableAV.data_prep.create_pairs --dataset_path "explainableAV/PAN20/PAN20_filtered.json" --SS_file_path "explainableAV/PAN20/SS.json" --SD_file_path "explainableAV/PAN20/SD.json" --DS_file_path "explainableAV/PAN20/DS.json" --DD_file_path "explainableAV/PAN20/DD.json"
```
This will create four datasets, one per pair type: \
explainableAV/PAN20/SS.json: contains all text pairs that are Same-author Same-topic \
explainableAV/PAN20/SD.json: contains all text pairs that are Same-author Different-topic \
explainableAV/PAN20/DS.json: contains all text pairs that are Different-author Same-topic \
explainableAV/PAN20/DD.json: contains all text pairs that are Different-author Different-topic \

A dataset entry has the following form:
```sh
entry = {
"Label": 0, # 0 for different-author, 1 for same-author
"Topics": (topic1, topic2),
"Pair": (text1, text2)
}
```

Finally, to create the train, test, and validation splits of the text pairs, run the following command:
```sh
# PAN20
python -m explainableAV.data_prep.data_split --samples_per_pair 2500 --SS_file_path "explainableAV/PAN20/SS.json" --SD_file_path "explainableAV/PAN20/SD.json" --DS_file_path "explainableAV/PAN20/DS.json" --DD_file_path "explainableAV/PAN20/DD.json"
```
This will create the following files: \
explainableAV/PAN20/SS_test.json&emsp;explainableAV/PAN20/SS_val.json&emsp;explainableAV/PAN20/SS_train.json \
explainableAV/PAN20/SD_test.json&emsp;explainableAV/PAN20/SD_val.json&emsp;explainableAV/PAN20/SD_train.json \ 
explainableAV/PAN20/DS_test.json&emsp;explainableAV/PAN20/DS_val.json&emsp;explainableAV/PAN20/DS_train.json \ 
explainableAV/PAN20/DD_test.json&emsp;explainableAV/PAN20/DD_val.json&emsp;explainableAV/PAN20/DD_train.json \
explainableAV/PAN20/test_set_2500x4.json&emsp;explainableAV/PAN20/val_set_2500x4.json&emsp;explainableAV/PAN20/train_set_2500x4.json \
Where the last line combines all pair types into one test, val or train dataset \

### Results

To print an overview of the number of pairs per pair type in the data, run:
```sh
python -m explainableAV.data_prep.data_distributions --data_name "PAN20"
```

To print an overview of the number of pairs per split per pair type in the data, run:
```sh
# PAN20
python -m explainableAV.data_prep.data_distributions --statistic 'splits' --data_name "PAN20" --split_size 2500
```

To plot the topic distributions of the train and test dataset, run the following:
```sh
python -m explainableAV.data_prep.data_distributions --statistic 'topic_distribution' --data_name "PAN20" --split_size 2500
```
The plot is stored as explainableAV/data_prep/Topic_distribution_PAN20_2500.pdf
