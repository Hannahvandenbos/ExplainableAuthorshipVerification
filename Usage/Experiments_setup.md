# Experiments Setup
This file provides information on how to obtain the model classification thresholds.

## Thresholds
To find the optimal thresholds for the AV models, according to the minimum standard deviation between the accuracies over the pair types, run the following code whereby the plots over the thresholds are plotted as well:
```sh
python -m explainableAV.models.find_thresholds # Amazon data, LUAR model
python -m explainableAV.models.find_thresholds --model_name 'ModernBERT' # Amazon data, ModernBERT model
python -m explainableAV.models.find_thresholds --model_name 'StyleDistance' # Amazon data, StyleDistance model

python -m explainableAV.models.find_thresholds --SS_val_path "explainableAV/PAN20/SS_val.json" --SD_val_path "explainableAV/PAN20/SD_val.json" --DS_val_path "explainableAV/PAN20/DS_val.json" --DD_val_path "explainableAV/PAN20/DD_val.json" --dataset_name "pan20"  # PAN20 data, LUAR model
python -m explainableAV.models.find_thresholds --SS_val_path "explainableAV/PAN20/SS_val.json" --SD_val_path "explainableAV/PAN20/SD_val.json" --DS_val_path "explainableAV/PAN20/DS_val.json" --DD_val_path "explainableAV/PAN20/DD_val.json" --dataset_name "pan20" --model_name 'ModernBERT' # PAN20 data, ModernBERT model
python -m explainableAV.models.find_thresholds --SS_val_path "explainableAV/PAN20/SS_val.json" --SD_val_path "explainableAV/PAN20/SD_val.json" --DS_val_path "explainableAV/PAN20/DS_val.json" --DD_val_path "explainableAV/PAN20/DD_val.json" --dataset_name "pan20" --model_name 'StyleDistance' # PAN20 data, StyleDistance model
```
The optimal thresholds are printed, the plots are stored in explainableAV/models/results
