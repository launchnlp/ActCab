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

# model_name = "meta-llama/Llama-2-7b-hf"
model_name = sys.argv[1]
dataset_name = sys.argv[2]
num_samples = sys.argv[3]
if len(sys.argv) > 4:
    num_demos = int(sys.argv[4])

h_layer = None
demonstration = True
demonstration_file = '../' + dataset_name + "/train.txt"

file_name = '../' + dataset_name + f"/train{model_name.split('/')[-1]}temperature_1topp_1.0_num_demos_5_answers.{num_samples}.IK_set.csv"
model_answer_file_name = '../' + dataset_name + f"/train{model_name.split('/')[-1]}temperature_1topp_1.0_num_demos_5_answers.{num_samples}.csv"
hidden_state_file = '../' + dataset_name + f"/{model_name.split('/')[-1]}_multi-layer-logits/{model_name.split('/')[-1]}_ICL_w_answer_hidden_states_4_answer_newsamples_normal_demos_sequence_num_demos5.pt"
ids_file = '../' + dataset_name + f"/{model_name.split('/')[-1]}_multi-layer-logits/{model_name.split('/')[-1]}_ICL_w_answer_ids_4_answer_newsamples_normal_demos_sequence_num_demos5.pt"
if h_layer is not None:
    hidden_state_file = '../' + dataset_name + f"/{model_name.split('/')[-1]}_multi-layer-logits/{model_name.split('/')[-1]}_ICL_w_answer_hidden_states_h_{h_layer}_4_answer_newsamples_normal_demos_sequence.pt"
    ids_file = '../' + dataset_name + f"/multi-layer-logits/{model_name.split('/')[-1]}_ICL_w_answer_ids_h_{h_layer}_4_answer_newsamples_sequence_num_demos5.pt"
if random_demos:
    hidden_state_file = '../' + dataset_name + f"/{model_name.split('/')[-1]}_multi-layer-logits/{model_name.split('/')[-1]}_ICL_w_answer_hidden_states_random_demos_4_answer_newsamples_normal_demos_sequence.pt"
    ids_file = '../' + dataset_name + f"/{model_name.split('/')[-1]}_multi-layer-logits/{model_name.split('/')[-1]}_ICL_w_answer_ids_random_demos_4_answer_newsamples_sequence_num_demos5.pt"

if num_samples == '1': # greedy_search
    file_name = '../' + dataset_name + f"/train{model_name.split('/')[-1]}temperature_1topp_1.0_num_demos_5_answers.greedy_search.IK_set.csv"
    model_answer_file_name = '../' + dataset_name + f"/train{model_name.split('/')[-1]}temperature_1topp_1.0_num_demos_5_answers.greedy_search.csv"
    hidden_state_file = '../' + dataset_name + f"/{model_name.split('/')[-1]}_multi-layer-logits/{model_name.split('/')[-1]}_ICL_w_answer_hidden_states_4_answer_normal_demos_greedy_search_sequence_num_demos5.pt"
    ids_file = '../' + dataset_name + f"/{model_name.split('/')[-1]}_multi-layer-logits/{model_name.split('/')[-1]}_ICL_w_answer_ids_4_answer_normal_demos_greedy_search_sequence_num_demos5.pt"


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

num_layers = model.config.num_hidden_layers

def get_init_prompt():
    prompt = """SYSTEM: You are an AI research assistant. You use a tone that is technical and scientific.
USER: Hello, who are you?
ASSISTANT: Greeting! I am an AI research assistant. How can I help you today?
USER: """

    if demonstration:
        demo_questions = [item.split('\n')[0] for item in open(demonstration_file).read().split('\n\n') if len(item.split('\n')) > 1]
        demo_answers = [item.split('\n')[1] for item in open(demonstration_file).read().split('\n\n') if len(item.split('\n')) > 1]
        # randomly select num_demos questions and corrisponding answers from the demonstration file
        qa_pairs = list(zip(demo_questions, demo_answers))
        random.shuffle(qa_pairs)
        qa_pairs = qa_pairs[:num_demos]
        # random.shuffle(qa_pairs)
        for qa_pair in qa_pairs:
            prompt += qa_pair[0] + '\nASSISTANT: ' + qa_pair[1] + '\nUSER: '

    return prompt

if not random_demos:
    prompt = get_init_prompt()

questions = q2c.keys()

total_hidden_states = []
total_labels = []
num_answers = 4


for idx, q in enumerate(tqdm(questions)):
    if random_demos:
        prompt = get_init_prompt()
    q2a[q] = q2a[q][:num_answers]
    for model_answer, c in zip(q2a[q], q2c[q][:num_answers]):
        now_correct_prompt = prompt + q + '\nASSISTANT: ' + model_answer

        with torch.no_grad():
            encoding = tokenizer(now_correct_prompt.strip(), return_tensors='pt').to(generator.device)
            prompt_length = tokenizer((prompt + q + '\nASSISTANT:').strip())['input_ids'].__len__()
            encoding['labels'] = encoding['input_ids'].clone()
            encoding['output_hidden_states'] = True
            outputs = model(**encoding)

            # compute the loss of new tokens
            logits = outputs.logits
            hidden_states = outputs.hidden_states
            hidden_states_write = []
            for h_layer in range(num_layers):
                hidden_states_write.append(hidden_states[h_layer][:, prompt_length:, :].detach().to(torch.float32).cpu())

            if hidden_states_write[0].shape[1] == 0:
                print(q)
                print(model_answer)
                # exit()
                continue
            total_hidden_states.append(hidden_states_write)
            total_labels.append(torch.tensor([c]))

for h_layer in range(num_layers):
    total_hidden_states_cur_layer = []
    for item in total_hidden_states:
        total_hidden_states_cur_layer.append(item[h_layer])

    file_name = hidden_state_file.replace('.pt', f'.h_{h_layer}.pt')
    id_file_name = ids_file.replace('.pt', f'.h_{h_layer}.pt')

    torch.save(total_hidden_states_cur_layer, file_name)
    torch.save(total_labels, id_file_name)
