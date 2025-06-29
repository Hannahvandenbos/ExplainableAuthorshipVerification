import json
import random
import copy
import argparse
from collections import defaultdict
from explainableAV.utils.utils import load_dataset, create_dataset

def group_by_authors(data):
    '''
    Reorder data such that per author we get all texts with topic
    Input:
        data: loaded file with the reordered data (see reorder_Amazon.py or reorder_PAN20.py)
    Output:
        dictionary ordered by authors
    '''
    authors = {}
    for review in data:
        reviewer = review["authorID"]
        if reviewer not in authors.keys():
            authors[reviewer] = [(review.get("Text"), review.get("Topic"))]
        else:
            authors[reviewer].append((review.get("Text"), review.get("Topic")))
    return authors

def group_by_topic_within_author(authors):
    '''
    If you have data per author, orden the texts per topic as well
    Input:
        author: dataset with texts and topics grouped per author
    Output:
        dictionary ordered by authors then by topics
    '''
    authors_topics = {}
    for author in authors.keys():
        topics = {}
        for text, topic in authors[author]:
            if topic not in topics.keys():
                topics[topic] = [text]
            else:
                topics[topic].append(text)
        authors_topics[author] = topics
    return authors_topics

def same_topic_pairs(topic, author_per_topic):
    '''
    Create same topic pairs given texts from one author and topic
    Inputs:
        topic: topic to create pairs for
        author_per_topic: dataset structured as obtained through group_by_topic_within_author for one author
    Output:
        same-author same-topic pairs
    '''
    data = []
    texts = copy.deepcopy(author_per_topic[topic])
    while texts:
        text1 = texts.pop(0)

        if texts:

            text2 = random.choice(texts)
            texts.remove(text2)
            new_entry = {
                            "Label": 1,
                            "Topics": (topic, topic),
                            "Pair": (text1, text2)
                        }
            data.append(new_entry)
    return data

def cross_topic_pairs(topics, dictionary):
    '''
    Create cross topic pairs given texts from one author and topic
    Inputs:
        topics: topics to create pairs for
        dictionary: dataset structured as obtained through group_by_topic_within_author for one author
    Output:
        same-author different-topic pairs
    '''
    data = []
    topic_dict = copy.deepcopy(dictionary)
    while len(topics) > 1: # need at least two topics for cross-topic
        topic1 = random.choice(topics)
        text1 = random.choice(topic_dict[topic1])
        topic_dict[topic1].remove(text1) # should not be in any other pair

        topics_new = [t for t in topics if t != topic1] # cannot choose the same topic
        topic2 = random.choice(topics_new)
        text2 = random.choice(topic_dict[topic2])
        topic_dict[topic2].remove(text2) # should not be in any other pair

        new_entry = {
                        "Label": 1,
                        "Topics": (topic1, topic2),
                        "Pair": (text1, text2)
                    }
        data.append(new_entry)

        topics = [t for t in topic_dict.keys() if len(topic_dict[t]) > 0]
    return data

def same_author(author_topic, SS_file_name, SD_file_name):
    '''
    Creates the same author pairs
    Inputs:
        author_topic: dataset structured as obtained through group_by_topic_within_author
        SS_file_name: path to store the same-author same-topic data
        SD_file_name: path to store the same-author cross-topic data 
    '''
    SS_data = []
    SD_data = []
    author_topic_copy = copy.deepcopy(author_topic)
    for author in author_topic_copy.keys():
        topic_dict = author_topic_copy[author] 
        for topic in topic_dict.keys():
            # get same author, same topic pairs
            SS_data += same_topic_pairs(topic, topic_dict)
        
        # get same author, different topic pairs
        topics = [t for t in topic_dict.keys() if len(topic_dict[t]) > 0]
        SD_data += cross_topic_pairs(topics, topic_dict)

    create_dataset(SS_file_name, SS_data)
    create_dataset(SD_file_name, SD_data)
    print("Number of same-author same-topic pairs: ",len(SS_data))
    print("Number of same-author cross-topic pairs: ",len(SD_data))

def diff_author_same_topic(author_topic, DD_file_name):
    '''
    Creates and stores the different author same topic pairs
    Inputs:
        author_topic: dataset structured as obtained through group_by_topic_within_author
        DD_file_name: path to store the different-author same-topic data
    '''
    data = []
    author_topic_copy = copy.deepcopy(author_topic)
    authorIDs = list(author_topic_copy.keys())
    while len(authorIDs) > 1: # need at least two authors to create pair
        author1 = random.choice(authorIDs)
        topics = list(author_topic_copy[author1].keys())
        if not topics: # if author has no topics continue and remove author 
            authorIDs.remove(author1)
            continue

        topic = random.choice(topics)
        text1 = random.choice(author_topic_copy[author1][topic])
        
        author_topic_copy[author1][topic].remove(text1) # cannot choose text again

        authorIDs_new = [a for a in author_topic_copy if a != author1 and topic in author_topic_copy[a].keys()] # author must not be the same and have the topic

        if not authorIDs_new:  # if no author with same topic
            del author_topic_copy[author1][topic] # there is no two authors with same topic so remove topic
            continue

        if not author_topic_copy[author1][topic]: # remove topic if no more texts left
            del author_topic_copy[author1][topic]

        author2 = random.choice(authorIDs_new)
        text2 = random.choice(author_topic_copy[author2][topic])
        
        author_topic_copy[author2][topic].remove(text2) # cannot choose text again
        if not author_topic_copy[author2][topic]: # remove topic if no more texts left
            del author_topic_copy[author2][topic]

        new_entry = {
                        "Label": 0,
                        "Topics": (topic, topic),
                        "Pair": (text1, text2)
                    }
        data.append(new_entry)
        
    create_dataset(DD_file_name, data)
    print("Number of cross-author same-topic pairs: ",len(data))

def diff_author_diff_topic(author_topic, DD_file_name):
    '''
    Creates and saves the different author cross-topic pairs
    Inputs:
        author_topic: dataset structured as obtained through group_by_topic_within_author
        DD_file_name: path to store the different-author cross-topic data
    '''
    data = []
    author_topic_copy = copy.deepcopy(author_topic)
    authorIDs = list(author_topic_copy.keys())
    while len(authorIDs) > 1: # need at least two authors to create pair
        author1 = random.choice(authorIDs)
        topics = list(author_topic_copy[author1].keys())  # Get list of topics for author1
        if not topics: # if author has no topics continue and remove author 
            authorIDs.remove(author1)
            continue

        topic1 = random.choice(topics)
        text1 = random.choice(author_topic_copy[author1][topic1])
        
        author_topic_copy[author1][topic1].remove(text1) # cannot choose text again
        if not author_topic_copy[author1][topic1]: # remove topic if no more texts left
            del author_topic_copy[author1][topic1]

        authorIDs_new = [a for a in author_topic_copy if a != author1] # author must not be the same
        author2 = random.choice(authorIDs_new)
        topics2 = [t for t in author_topic_copy[author2] if t != topic1] # topic should be different
        if not topics2: # if author has no topics that are different continue
            continue
        topic2 = random.choice(topics2)
        text2 = random.choice(author_topic_copy[author2][topic2])
        
        author_topic_copy[author2][topic2].remove(text2) # cannot choose text again
        if not author_topic_copy[author2][topic2]: # remove topic if no more texts left
            del author_topic_copy[author2][topic2]

        new_entry = {
                        "Label": 0,
                        "Topics": (topic1, topic2),
                        "Pair": (text1, text2)
                    }
        data.append(new_entry)
        
    create_dataset(DD_file_name, data)
    print("Number of cross-author cross-topic pairs: ",len(data))

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset_path', type=str, default="Amazon/amazon_reviews_final.json", help='Path to dataset that has reviewerID, Topic and reviewText')
    parser.add_argument('--SS_file_path', type=str, default="Amazon/same_author_same_topic_pairs.json", help="Path to store same-author same-topic data")
    parser.add_argument('--SD_file_path', type=str, default="Amazon/same_author_diff_topic_pairs.json", help="Path to store same-author different-topic data")
    parser.add_argument('--DS_file_path', type=str, default="Amazon/diff_author_same_topic.json", help="Path to store different-author same-topic data")
    parser.add_argument('--DD_file_path', type=str, default="Amazon/diff_author_diff_topic.json", help="Path to store different-author different-topic data")    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = argument_parser()
    random.seed(args.seed)
    SS = args.SS_file_path
    SD = args.SD_file_path
    DS = args.DS_file_path
    DD = args.DD_file_path
    data = load_dataset(args.dataset_path) # load data
    authors = group_by_authors(data) # order by author
    author_topic = group_by_topic_within_author(authors) # order by topic per author
    same_author(author_topic, SS, SD)
    diff_author_same_topic(author_topic, DS)
    diff_author_diff_topic(author_topic, DD)
