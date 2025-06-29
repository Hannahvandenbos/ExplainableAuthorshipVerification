import json
import argparse
import statistics
from collections import defaultdict
from explainableAV.utils.utils import load_dataset, load_dataset_jsonl, create_dataset

def aggregate_dataset(texts, labels, final_dataset_name):
    '''
    Aggregate the two PAN files and store as final dataset
    Inputs:
        texts: file that contains the text pairs
        labels: file that contains the labels of the pairs
        final_dataset_name: path to store the final preprocessed dataset (before creating the pairs)
    '''
    data = []
    seen_entries = set()
    for label, text in zip(labels, texts):
        if label['id'] == text['id']: # just to be sure
            new_entry = (
                text['pair'][0],
            )
            new_entry2 = (
                text['pair'][1],
            )

        if new_entry not in seen_entries:
            seen_entries.add(new_entry)
            data.append({
                "authorID": label['authors'][0],
                "Text": text['pair'][0],
                "Topic": text['fandoms'][0]
            })
        if new_entry2 not in seen_entries:
            seen_entries.add(new_entry2)
            data.append({
                "authorID": label['authors'][1],
                "Text": text['pair'][1],
                "Topic": text['fandoms'][1]
            })

    create_dataset(final_dataset_name, data)


def number_of_topics_filter(topics_file_name, new_topics_file_name):
    '''
    Can be used to get statistics of final dataset
    Inputs:
        topics_file_name: final dataset after aggregating
        new_topics_file_name: name to save filtered data
    '''
    data = load_dataset(topics_file_name)

    num_topics = defaultdict(int)
    for fanfiction in data:
        num_topics[fanfiction["Topic"]] += 1

    filtered_data = [
        fanfiction for fanfiction in data
        if num_topics[fanfiction["Topic"]] >= 50
    ]

    filtered_num_topics = defaultdict(int)
    num_authors = defaultdict(int)
    for fanfiction in filtered_data:
        filtered_num_topics[fanfiction["Topic"]] += 1
        num_authors[fanfiction["authorID"]] += 1

    print("The number of documents per topic are: ", filtered_num_topics)
    print("The number of documents are: ", len(filtered_data))
    print("The number of topics are: ", len(filtered_num_topics))
    print("The number of authors are: ", len(num_authors))

    create_dataset(new_topics_file_name, filtered_data)


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--final_dataset_name', type=str, default="explainableAV/PAN20/PAN20.json", help="Path to store final preprocessed PAN20 data")
    parser.add_argument('--final_dataset_name_filtered', type=str, default="explainableAV/PAN20/PAN20_filtered.json", help="Path to store final preprocessed PAN20 data")
    parser.add_argument('--texts_path', type=str, required=True, help="Path to jsonl file that contains the text pairs")
    parser.add_argument('--label_path', type=str, required=True, help="Path to jsonl file that contains the labels")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = argument_parser()
    texts_path = args.texts_path
    labels_path = args.label_path
    new_file_name = args.final_dataset_name
    texts = load_dataset_jsonl(texts_path)
    labels = load_dataset_jsonl(labels_path)

    aggregate_dataset(texts, labels, new_file_name)
    number_of_topics_filter(args.final_dataset_name, args.final_dataset_name_filtered)
