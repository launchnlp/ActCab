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
h_layer = int(sys.argv[3])
demonstration = True
demonstration_file = "../" + dataset_name + "/train.txt"
correctness_file_name = sys.argv[4]
model_answer_file_name = sys.argv[5]
num_demos = int(sys.argv[6])

    
file_df = pd.read_csv(correctness_file_name)
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
    if "[" in answer:
        q2a[question] += [item for item in eval(answer)]
    else:
        q2a[question] += [answer]



device = 0
if "13b" not in model_name:
    generator = pipeline("text-generation", model=model_name, device=0)
else:
    generator = pipeline("text-generation", model=model_name, trust_remote_code=True, device=device, torch_dtype=torch.float16)
model = generator.model
tokenizer = generator.tokenizer

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

questions = [q for q in q2c.keys()]

total_hidden_states = []
total_labels = []
num_answers = 1

stored_answers = []
confidence_list = []
label_list = []
question_list = []
class Calibrator(torch.nn.Module):
    def __init__(self, hidden_size):
        super(Calibrator, self).__init__()
        self.linear = torch.nn.Linear(hidden_size, 1, bias=False)
        

    def forward(self, hidden_states):
        l1 = self.linear(hidden_states)
        return nn.functional.sigmoid(l1), None

if "13b" not in model_name:
    calibrator = Calibrator(4096).cuda()
else:
    calibrator = Calibrator(5120).cuda()
    
stored_state_dict = torch.load('../' + dataset_name + "/multiple_layer_ckpt/" + model_name.split('/')[-1] + "_calibrator_intenvention_seq_avg_greedy_search_num_demos" + str(num_demos) + ".h_" + str(h_layer) + ".pt")
calibrator.load_state_dict(stored_state_dict)
if "13b" in model_name:
    calibrator = calibrator.half()

for idx, q in enumerate(tqdm(questions)):
    if random_demos:
        prompt = get_init_prompt()
    q2a[q] = q2a[q][:num_answers]
    for model_answer, c in zip(q2a[q], q2c[q][:num_answers]):
        now_correct_prompt = prompt + q + '\nASSISTANT: ' + model_answer

        with torch.no_grad():
            encoding = tokenizer(now_correct_prompt.strip(), return_tensors='pt').to(generator.device)
            encoding['labels'] = encoding['input_ids'].clone()
            encoding['output_hidden_states'] = True
            outputs = model(**encoding)

            prompt_length = tokenizer((prompt + q + '\nASSISTANT:').strip())['input_ids'].__len__()

            # compute the loss of new tokens
            logits = outputs.logits
            hidden_states = outputs.hidden_states
            if h_layer is not None:
                hidden_states_write = hidden_states[h_layer][:, prompt_length:, :].detach().cpu()
            else:
                hidden_states_write = hidden_states[-1][:, prompt_length:, :].detach().cpu()
            total_hidden_states.append(hidden_states_write)
        
        stored_answers.append(model_answer)
        if h_layer is not None:
            confidence = calibrator(hidden_states[h_layer][:, prompt_length:, :].mean(dim=1, keepdim=True))[0].cpu().item()
        else:
            confidence = calibrator(hidden_states[-1][:, prompt_length:, :].mean(dim=1, keepdim=True))[0].cpu().item()
        confidence_list.append(confidence)
        
        label_list.append(c)
        
    question_list.append(q)
    # break

data_df = pd.DataFrame()
data_df['question'] = question_list
data_df['answer'] = stored_answers
data_df['confidence'] = confidence_list
data_df['correctness'] = label_list

data_df.to_csv(correctness_file_name.replace('.csv', '.confidence_seq_avg_greedy_linear.csv'), index=False)