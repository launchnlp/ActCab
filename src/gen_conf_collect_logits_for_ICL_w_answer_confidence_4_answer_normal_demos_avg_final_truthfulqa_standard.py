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
    
stored_state_dict = torch.load('../' + dataset_name + "/ICL_cls_ckpt/" + model_name.split('/')[-1] + "_calibrator_intenvention_seq_avg_num_demos5.pt")
calibrator.load_state_dict(stored_state_dict)
if "13b" in model_name:
    calibrator = calibrator.half()

for idx, q in enumerate(tqdm(questions)):
    if random_demos:
        prompt = get_init_prompt()
    q2a[q] = q2a[q][:num_answers]
    for model_answer in q2a[q]:
        now_correct_prompt = prompt + q + '\nA: ' + model_answer

        with torch.no_grad():
            encoding = tokenizer(now_correct_prompt.strip(), return_tensors='pt').to(generator.device)
            encoding['labels'] = encoding['input_ids'].clone()
            encoding['output_hidden_states'] = True
            outputs = model(**encoding)

            prompt_length = tokenizer((prompt + q + '\nA:').strip())['input_ids'].__len__()

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
        
    q2c[q] = q2c[q][:num_answers]
    for c in q2c[q]:
        # total_labels.append(torch.tensor([c]))
        label_list.append(c)
    question_list.append(q)
    # break

data_df = pd.DataFrame()
data_df['question'] = question_list
data_df['answer'] = stored_answers
data_df['confidence'] = confidence_list
data_df['correctness'] = label_list

data_df.to_csv(correctness_file_name.replace('.csv', '.confidence_seq_avg_greedy_linear.csv'), index=False)