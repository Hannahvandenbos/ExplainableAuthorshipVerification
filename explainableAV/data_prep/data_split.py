import json
import argparse
import random
from explainableAV.utils.utils import load_dataset, create_dataset
from sklearn.model_selection import train_test_split

def individual_split(data_name, class_name, data, test_size=0.2, seed=0):
    '''
    Create a train-test split for the given data
    Inputs:
        data_name: 'Amazon' or 'PAN20'
        class_name: 'SS', 'SD', 'DS', or 'DD'
        data: text pairs to split
        test_size: percentage of texts for the test set
        seed: seed
    Output:
        train and test split
    '''
    train, test = train_test_split(data, test_size=test_size, random_state=seed)
    create_dataset(f"explainableAV/{data_name}/{class_name}_train_filtered.json", train)
    create_dataset(f"explainableAV/{data_name}/{class_name}_test_filtered.json", test)
    return train, test

def manual_split(datas, name, data_names, samples_per_pair=15000):
    '''
    Create a test set of `samples_per_pair` datapoints per class and 
    use the remaining data for training
    Inputs:
        datas: all types of datasets
        name: data name 'Amazon' or 'PAN20'
        data_names: list of names from the data ['SS', 'SD', 'DS', 'DD']
        samples_per_pair: number of text pairs for the test set
    Output:
        train data set, test data set, validation data sets
    '''
    final_train_data = []
    final_test_data = []
    final_val_data = []

    for data, class_name in zip(datas, data_names):
        test_subset = random.sample(data, samples_per_pair)
        
        create_dataset(f"explainableAV/{name}/{class_name}_test.json", test_subset)
        final_test_data += test_subset

        remaining_data = [item for item in data if item not in test_subset]

        val_size = samples_per_pair // 2
        val_subset = random.sample(remaining_data, val_size)
        create_dataset(f"explainableAV/{name}/{class_name}_val.json", val_subset)
        final_val_data += val_subset

        train_subset = [item for item in remaining_data if item not in val_subset]
        create_dataset(f"explainableAV/{name}/{class_name}_train.json", train_subset)
        final_train_data += train_subset

    return final_train_data, final_test_data, final_val_data

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data_name', type=str, default="Amazon", help='Dataset name: "PAN20" or "Amazon"')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test_size as a float')
    parser.add_argument('--samples_per_pair', type=int, default=15000)
    parser.add_argument('--SS_file_path', type=str, default="explainableAV/Amazon/SS.json", help="Path to load same-author same-topic data")
    parser.add_argument('--SD_file_path', type=str, default="explainableAV/Amazon/SD.json", help="Path to load same-author different-topic data")
    parser.add_argument('--DS_file_path', type=str, default="explainableAV/Amazon/DS.json", help="Path to load different-author same-topic data")
    parser.add_argument('--DD_file_path', type=str, default="explainableAV/Amazon/DD.json", help="Path to load different-author different-topic data")  
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = argument_parser()
    random.seed(args.seed)
    SS = load_dataset(args.SS_file_path)
    SD = load_dataset(args.SD_file_path)
    DS = load_dataset(args.DS_file_path)
    DD = load_dataset(args.DD_file_path)

    # # create individual train, test splits
    # SS_train, SS_test = individual_split(args.data_name, "SS", SS, test_size=args.test_size, seed=args.seed)
    # SD_train, SD_test = individual_split(args.data_name, "SD", SD, test_size=args.test_size, seed=args.seed)
    # DS_train, DS_test = individual_split(args.data_name, "DS", DS, test_size=args.test_size, seed=args.seed)
    # DD_train, DD_test = individual_split(args.data_name, "DD", DD, test_size=args.test_size, seed=args.seed)

    # # create full train, test split
    # full_train = SS_train + SD_train + DS_train + DD_train
    # full_test = SS_test + SD_test + DS_test + DD_test
    # create_dataset(f"{args.data_name}/test.json", full_test)
    # create_dataset(f"{args.data_name}/train.json", full_train)

    # create manual inference set
    data_names = ["SS", "SD", "DS", "DD"]
    final_train_data, final_test_data, final_val_data = manual_split([SS, SD, DS, DD], args.data_name, data_names, samples_per_pair=args.samples_per_pair)
    create_dataset(f"explainableAV/{args.data_name}/test_set_{args.samples_per_pair}x4.json", final_test_data)
    create_dataset(f"explainableAV/{args.data_name}/train_set_{args.samples_per_pair}x4.json", final_train_data)
    create_dataset(f"explainableAV/{args.data_name}/val_set_{args.samples_per_pair}x4.json", final_val_data)
