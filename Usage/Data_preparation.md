# Data Preparation
This file explains which data we used and how to create the pairs and preprocess the data in general, ensuring applicability with the experiments. 
We use two datasets: Amazon reviews and Fanfictions from the PAN2020 competition.

## Amazon
The *Amazon Reviews* dataset needs to be downloaded from [Amazon Reviews](https://nijianmo.github.io/amazon/index.html) \
Download the 5-core files for the following categories and extract them to the *explainableAV/Amazon* folder:
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
Run the following command to filter and reorder the datasets:
```sh
# Get single reviews and set topic
python -m explainableAV.data_prep.reorder_Amazon
```

To create all text pairs (SS, SD, DS, DD), run the following command:
```sh
python -m explainableAV.data_prep.create_pairs --dataset_path "explainableAV/Amazon/amazon_reviews_final.json" --SS_file_path "explainableAV/Amazon/SS.json" --SD_file_path "explainableAV/Amazon/SD.json" --DS_file_path "explainableAV/Amazon/DS.json" --DD_file_path "explainableAV/Amazon/DD.json"
```

Finally, to create the train, test, and validation splits of the text pairs, run the following command:
```sh
python -m explainableAV.data_prep.data_split --samples_per_pair 15000 --SS_file_path "explainableAV/Amazon/SS.json" --SD_file_path "explainableAV/Amazon/SD.json" --DS_file_path "explainableAV/Amazon/DS.json" --DD_file_path "explainableAV/Amazon/DD.json"
```
For a smaller or larger dataset, change --samples_per_pair to a different number as the commands above are specific to the full data.

### Results
To print an overview of the number of pairs per pair type in the data, run:
```sh
python -m explainableAV.data_prep.data_distributions
```

To print an overview of the number of pairs per split per pair type in the data, run:
```sh
python -m explainableAV.data_prep.data_distributions --statistic 'splits' --split_size 15000
```
Split size should correspond with the test size of one pair type in your data.

To plot the topic distributions of the train and test dataset, run the following:
```sh
python -m explainableAV.data_prep.data_distributions --statistic 'topic_distribution' --split_size 15000
```
Split size should correspond with the test size of one pair type in your data.
Plots are stored in explainableAV/data_prep

## PAN20
The PAN20 dataset needs to be downloaded from [PAN20](https://zenodo.org/records/3724096) \
You can opt for both the large or small version. The small version was used in this research. \
Store the two jsonl files in the *explainableAV/PAN20* folder.

### Data Preprocessing
Run the following commands to filter and reorder the datasets:
```sh
# PAN20 (small)
python -m explainableAV.data_prep.reorder_PAN20 --texts_path "explainableAV/PAN20/.pan20-authorship-verification-training-small.jsonl" --label_path "explainableAV/PAN20/pan20-authorship-verification-training-small-truth.jsonl"

# PAN20 (large)
python -m explainableAV.reorder_PAN20 --texts_path "explainableAV/PAN20/pan20-authorship-verification-training-large.jsonl" --label_path "explainableAV/PAN20/pan20-authorship-verification-training-large-truth.jsonl"
```

To create all text pairs (SS, SD, DS, DD), run the following command:
```sh
# PAN20
python -m explainableAV.data_prep.create_pairs --dataset_path "explainableAV/PAN20/PAN20_filtered.json" --SS_file_path "explainableAV/PAN20/SS.json" --SD_file_path "explainableAV/PAN20/SD.json" --DS_file_path "explainableAV/PAN20/DS.json" --DD_file_path "explainableAV/PAN20/DD.json"
```

Finally, to create the train, test, and validation splits of the text pairs, run the following command:
```sh
# PAN20
python -m explainableAV.data_prep.data_split --samples_per_pair 2500 --SS_file_path "explainableAV/PAN20/SS.json" --SD_file_path "explainableAV/PAN20/SD.json" --DS_file_path "explainableAV/PAN20/DS.json" --DD_file_path "explainableAV/PAN20/DD.json"
```

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
Plots are stored in explainableAV/data_prep
