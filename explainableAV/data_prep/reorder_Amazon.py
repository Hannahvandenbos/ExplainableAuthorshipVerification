import json
import argparse
from collections import defaultdict
from explainableAV.utils.utils import load_dataset, create_dataset

def aggregated_dataset(file_names, topics):
    '''
    Aggregate the 5-core amazon datasets
    Inputs:
        file_names: files to aggregate
        topics: names of the topics that correspond with the file_names
    Output:
        aggregated Amazon data
    '''
    data = []
    seen_entries = set()
    for i, file_name in enumerate(file_names):
        with open(file_name, "r", encoding="utf-8") as f:
            for line in f:
                review = json.loads(line.strip())
                new_entry = (
                    review.get("reviewText"),
                )
                if new_entry not in seen_entries:
                    seen_entries.add(new_entry)
                    data.append({
                        "authorID": review.get("reviewerID"),
                        "Text": review.get("reviewText"),
                        "Topic": topics[i]
                    })
    return data

def multi_topic_authors(aggregated_data, num_topics=7):
    '''
    Obtain dataset, where each author has written about at least num_topics topics
    Inputs:
        aggregated_data: aggregated data
        num_topics: minimum number of topics an author should have
    Output:
        filtered data
    '''
    data = aggregated_data

    reviewer_topics = defaultdict(set)

    for review in data:
        reviewer = review["authorID"]
        topic = review["Topic"]
        reviewer_topics[reviewer].add(topic)

    qualified_reviewers = {reviewer for reviewer, topics in reviewer_topics.items() if len(topics) >= num_topics}
    filtered_data = [review for review in data if review["authorID"] in qualified_reviewers]

    return filtered_data

def number_of_topics(multi_author_data):
    '''
    Get the number of documents per topic
    Inputs:
        multi_author_data: data after aggregating docs and filtering authors
    Output:
        number of documents per topic
    '''
    data = multi_author_data

    num_topics = defaultdict(int)
    for review in data:
        num_topics[review["Topic"]] += 1
    return num_topics

def remove_topics_with_few_documents(multi_author_data, final_dataset_name, num_docs=1000, num_tops=7):
    '''
    Remove topics that have too few documents (less than num_docs)
    Inputs:
        mulit_author_data: data after aggregating docs and filtering authors
        final_dataset_name: file to store the final preprocessed dataset
        num_docs: minimum number of documents per topic
        num_tops: minimum number of topics per author
    '''
    num_topics = number_of_topics(multi_author_data)
    valid_topics = {topic for topic, count in num_topics.items() if count >= num_docs}

    data = multi_author_data

    reviewer_topics = defaultdict(set)
    for review in data:
        if review["Topic"] in valid_topics:
            reviewer_topics[review["authorID"]].add(review["Topic"])

    # each author still needs at least 7 topics
    valid_reviewers = {reviewer for reviewer, topics in reviewer_topics.items() if len(topics) >= num_tops}
    filtered_data = [review for review in data if review["authorID"] in valid_reviewers and review["Topic"] in valid_topics]

    create_dataset(final_dataset_name, filtered_data)

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--final_dataset_name', type=str, default="explainableAV/Amazon/amazon_reviews_final.json", help="Path to store final preprocessed amazon review data")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = argument_parser()
    file_names = ["explainableAV/Amazon/AMAZON_FASHION_5.json", "explainableAV/Amazon/All_Beauty_5.json", "explainableAV/Amazon/Appliances_5.json", "explainableAV/Amazon/Arts_Crafts_and_Sewing_5.json", 
    "explainableAV/Amazon/Automotive_5.json", "explainableAV/Amazon/CDs_and_Vinyl_5.json", "explainableAV/Amazon/Cell_Phones_and_Accessories_5.json", "explainableAV/Amazon/Clothing_Shoes_and_Jewelry_5.json", 
    "explainableAV/Amazon/Digital_Music_5.json", "explainableAV/Amazon/Gift_Cards_5.json", "explainableAV/Amazon/Grocery_and_Gourmet_Food_5.json", "explainableAV/Amazon/Home_and_Kitchen_5.json", 
    "explainableAV/Amazon/Industrial_and_Scientific_5.json", "explainableAV/Amazon/Prime_Pantry_5.json", "explainableAV/Amazon/Software_5.json", "explainableAV/Amazon/Video_Games_5.json"]
    topics = ["fashion", "beauty", "appliances", "arts, crafts and sewing", "automotive", "cds and vinyl", "cell phones and accessories", "clothing, shoe and jewelry",
    "digital music", "gift cards", "grocery and gourmet food", "home and kitchen", "industrial and scientific", "prime pantry", "software", "video games"]
    final_dataset_name = args.final_dataset_name

    aggregated_data = aggregated_dataset(file_names, topics)
    multi_author_data = multi_topic_authors(aggregated_data)
    remove_topics_with_few_documents(multi_author_data, final_dataset_name)
