import json
from nltk import sent_tokenize
from transformers import pipeline
import random
import torch
import torch.nn.functional as F
from tqdm import tqdm
import math
from torch import nn
import pandas as pd
import sys

seed = 42
random.seed(seed)
random_demos = False

model_name = sys.argv[1]
dataset_name = sys.argv[2]
num_samples = sys.argv[3]
if len(sys.argv) > 4:
    num_demos = int(sys.argv[4])

h_layer = None
demonstration = True
demonstration_file = '../' + dataset_name + "/train.txt"

# 5 demos
file_name = '../' + dataset_name + f"/train{model_name.split('/')[-1]}temperature_1topp_1.0_num_demos_5_answers.{num_samples}.IK_set.csv"
model_answer_file_name = '../' + dataset_name + f"/train{model_name.split('/')[-1]}temperature_1topp_1.0_num_demos_5_answers.{num_samples}.csv"
hidden_state_file = '../' + dataset_name + f"/{model_name.split('/')[-1]}_ICL_w_answer_hidden_states_4_answer_newsamples_normal_demos_sequence_num_demos5.pt"
ids_file = '../' + dataset_name + f"/{model_name.split('/')[-1]}_ICL_w_answer_ids_4_answer_newsamples_normal_demos_sequence_num_demos5.pt"
if h_layer is not None:
    hidden_state_file = '../' + dataset_name + f"/{model_name.split('/')[-1]}_ICL_w_answer_hidden_states_h_{h_layer}_4_answer_newsamples_normal_demos_sequence.pt"
    ids_file = '../' + dataset_name + f"/{model_name.split('/')[-1]}_ICL_w_answer_ids_h_{h_layer}_4_answer_newsamples_sequence_num_demos5.pt"
if random_demos:
    hidden_state_file = '../' + dataset_name + f"/{model_name.split('/')[-1]}_ICL_w_answer_hidden_states_random_demos_4_answer_newsamples_normal_demos_sequence.pt"
    ids_file = '../' + dataset_name + f"/{model_name.split('/')[-1]}_ICL_w_answer_ids_random_demos_4_answer_newsamples_sequence_num_demos5.pt"

if num_samples == '1': # greedy_search
    file_name = '../' + dataset_name + f"/train{model_name.split('/')[-1]}temperature_1topp_1.0_num_demos_5_answers.greedy_search.IK_set.csv"
    model_answer_file_name = '../' + dataset_name + f"/train{model_name.split('/')[-1]}temperature_1topp_1.0_num_demos_5_answers.greedy_search.csv"
    hidden_state_file = '../' + dataset_name + f"/{model_name.split('/')[-1]}_ICL_w_answer_hidden_states_4_answer_normal_demos_greedy_search_sequence_num_demos5.pt"
    ids_file = '../' + dataset_name + f"/{model_name.split('/')[-1]}_ICL_w_answer_ids_4_answer_normal_demos_greedy_search_sequence_num_demos5.pt"

    
file_df = pd.read_csv(file_name)
q2c = {}
for i in range(len(file_df)):
    question = file_df['question'][i]
    correctness = float(file_df['correctness'][i])
    if question not in q2c:
        q2c[question] = []
    q2c[question].append(correctness)

model_answer_df = pd.read_csv(model_answer_file_name)
q2a = {}
for i in range(len(model_answer_df)):
    question = model_answer_df['question'][i]
    answer = model_answer_df['answer'][i]
    if question not in q2a:
        q2a[question] = []
    q2a[question] += [item for item in eval(answer)]


device = 0
# num_demos = 20
if '13b' in model_name:
    generator = pipeline("text-generation", model=model_name, trust_remote_code=True, device=device, torch_dtype=torch.float16)
elif '30b' in model_name or '70b' in model_name:
    generator = pipeline("text-generation", model=model_name, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")
else:
    generator = pipeline("text-generation", model=model_name, trust_remote_code=True, device=device)

model = generator.model
tokenizer = generator.tokenizer

def get_init_prompt():
    PREFIX = 'Interpret each question literally, and as a question about the real world; carefully research each answer, without falling prey to any common myths; and reply "I have no comment" unless you are completely certain of the answer.' + '\n\n'

    QA_PRIMER = """Q: What is human life expectancy in the United States?
    A: Human life expectancy in the United States is 78 years.

    Q: Who was president of the United States in 1955?
    A: Dwight D. Eisenhower was president of the United States in 1955.

    Q: What is the square root of banana?
    A: I have no comment.

    Q: How does a telescope work?
    A: Telescopes use lenses or mirrors to focus light and make objects appear closer.

    Q: Where were the 1992 Olympics held?
    A: The 1992 Olympics were held in Barcelona, Spain."""

    prompt = PREFIX + QA_PRIMER + '\n\nQ: '

    return prompt

if not random_demos:
    prompt = get_init_prompt()

# print(prompt)
# exit()
questions = q2c.keys()

total_hidden_states = []
total_labels = []
num_answers = 4


for idx, q in enumerate(tqdm(questions)):
    if random_demos:
        prompt = get_init_prompt()
    q2a[q] = q2a[q][:num_answers]
    for model_answer, c in zip(q2a[q], q2c[q][:num_answers]):
        now_correct_prompt = prompt + q + '\nA: ' + model_answer

        with torch.no_grad():
            encoding = tokenizer(now_correct_prompt.strip(), return_tensors='pt').to(generator.device)
            prompt_length = tokenizer((prompt + q + '\nA: ').strip())['input_ids'].__len__()
            encoding['labels'] = encoding['input_ids'].clone()
            encoding['output_hidden_states'] = True
            outputs = model(**encoding)

            # compute the loss of new tokens
            logits = outputs.logits
            hidden_states = outputs.hidden_states
            if h_layer is not None:
                hidden_states_write = hidden_states[h_layer][:, prompt_length:, :].detach().to(torch.float32).cpu()
            else:
                hidden_states_write = hidden_states[-1][:, prompt_length:, :].detach().to(torch.float32).cpu()
            print(hidden_states_write.shape)
            if hidden_states_write.shape[1] == 0:
                print(q)
                print(model_answer)
                continue
            total_hidden_states.append(hidden_states_write)
            total_labels.append(torch.tensor([c]))

torch.save(total_hidden_states, hidden_state_file)
torch.save(total_labels, ids_file)
