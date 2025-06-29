import argparse
import re
import copy
import random
from explainableAV.utils.utils import load_dataset, create_dataset

def clean_perturbed_text(original, perturbed):
    """
    Clean LLM perturbed text by removing artifacts and ensuring it ends at the first full sentence
    after reaching the original text's length
    Inputs:
        original: original input text
        perturbed: LLM perturbed text
    Output:
        cleaned perturbed text
    """
    # remove artificats
    perturbed = re.sub(r"\[/?INST\]", "", perturbed).strip()
    sentences = re.findall(r'([^\n.!?]*[.!?]|\n+)', perturbed)

    cleaned_text = ""
    for sentence in sentences:
        cleaned_text += sentence
        if len(cleaned_text.strip()) >= len(original) - 10:  # stop at full sentence after original length
            break

    return cleaned_text.strip()

def clean_text(data, generated_data):
    '''
    Clean LLM generated data
    Inputs:
        data: original data
        generated_data: LLM data to clean
    Output:
        cleaned LLM data
    '''
    changed_data = copy.deepcopy(generated_data)
    for line, generated in zip(data, changed_data):
        generated["Pair"][0] = clean_perturbed_text(line["Pair"][0], generated["Pair"][0])
    return changed_data

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm_data_path', required=True, type=str) 
    parser.add_argument('--original_data_path', required=True, type=str)
    parser.add_argument('--save', type=str, help="file path to save the cleaned version if None it is not saved") 
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = argument_parser()
    random.seed(args.seed)
    data = load_dataset(args.original_data_path)
    generated_data = load_dataset(args.llm_data_path)
    cleaned_data = clean_text(data, generated_data)
    if args.save:
        create_dataset(args.save, cleaned_data)
