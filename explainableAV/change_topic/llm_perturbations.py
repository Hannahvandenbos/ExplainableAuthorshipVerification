import argparse
import torch
import re
import copy
import random
import time
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from explainableAV.utils.utils import load_dataset, create_dataset

def change_topic_llama(text, new_topic, model, tokenizer, device):
    '''
    Change topic of a text with Llama
    Inputs:
        text: input text
        new_topic: topic to which to change the text
        model: llama model
        tokenizer: model tokenizer
        device: cuda or cpu
    Output:
        generated text
    '''
    prompt = f"""[INST]
    Change the topic of the following text to {new_topic}, without changing any stylistic features, and output nothing but that new text: "{text}"
    [/INST]"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    start = time.time()

    # Llama model
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=len(inputs.input_ids[0]) + 10,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    end = time.time()
    print("Time: ", end - start)
    if "[/INST]" in response: # remove instructions from generated text
        response = response.split("[/INST]", 1)[-1].strip()

    response = re.sub(r"Change the topic of the following text.*?:", "", response, flags=re.DOTALL).strip()

    return response

def llm_perturbations(data):
    '''
    Generate LLM perturbations with Llama
    Input:
        data: texts to perturb
    Output:
        perturbed texts
    '''
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load Llama
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto"
    )

    # get pool to select new topic
    unique_topics = []
    for line in data:
        unique_topics.append(line["Topics"][0])
        unique_topics.append(line["Topics"][1])
    unique_topics = set(unique_topics)

    changed_data = copy.deepcopy(data)

    for line in tqdm(changed_data, desc="Processing LLM Perturbations", unit="sample"):
        topic1, topic2 = line["Topics"]

        if topic1 != topic2:  # change to the same topic
            line["Pair"][0] = change_topic_llama(line["Pair"][0], topic2, model, tokenizer, device)
            line["Topics"][0] = topic2
            print(topic2)
        else:  # change to a different topic
            new_topic = random.Random(0).choice(sorted(unique_topics - {topic1}))  # pick from all topics except current
            line["Pair"][0] = change_topic_llama(line["Pair"][0], new_topic, model, tokenizer, device)
            line["Topics"][0] = new_topic
            print(new_topic)

    return changed_data

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True, type=str) 
    parser.add_argument('--save', type=str, help="file path to save the changed topic version if None it is not saved") 
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = argument_parser()
    random.seed(args.seed)
    data = load_dataset(args.data_path)
    # for prompt testing
    random.shuffle(data)
    indices = [1, 4, 9, 15, 16, 17, 18, 22, 24, 32]
    data_sample = [data[i] for i in indices]
    # --------------------

    changed_data = llm_perturbations(data_sample)
    if args.save:
        create_dataset(args.save, changed_data)
