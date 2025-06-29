from collections import defaultdict
import copy

def get_individual_texts_per_topic(data):
    '''
    Return dictionary with list of texts per topic
    '''
    topic_texts = defaultdict(list)
    for line in data:
        text1 = line["Pair"][0]
        text2 = line["Pair"][1]
        topic1 = line["Topics"][0]
        topic2 = line["Topics"][1]
        topic_texts[topic1].append(text1)
        topic_texts[topic2].append(text2)
    return topic_texts

def mask_first_text(data, masked_data, perturb_second):
    new_data = copy.deepcopy(data)
    for line, masked_line in zip(new_data, masked_data):
        line["Pair"][0] = masked_line["Pair"][0]
        if perturb_second:
            line["Pair"][1] = masked_line["Pair"][1]
    return new_data
