import json

def load_dataset(file_name):
    '''Load json file'''
    with open(file_name, "r") as f:
        data = json.load(f)
    return data

def load_multiple_datasets(file_names):
    '''Load and combine multiple json file'''
    data = []
    for file in file_names:
        data += load_dataset(file)
    return data

def load_dataset_jsonl(file_name):
    '''Load jsonl file'''
    data = [json.loads(line) for line in open(file_name, 'r')]
    return data

def create_dataset(file_name, data):
    '''Save data as json file'''
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
