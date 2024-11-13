from transformers import pipeline
import pandas as pd
from nltk.tokenize import sent_tokenize
import random
import sys
import torch
from torch import nn
import time

seed = 42
random.seed(seed)

model_name = sys.argv[1]
dataset_name = sys.argv[2]
h_layer = int(sys.argv[3])

demonstration = True
question_file = "../" + dataset_name + "/test.txt"
demonstration_file = "../" + dataset_name + "/train.txt"
device = 0
num_return_sequences = int(sys.argv[4])
if len(sys.argv) > 4:
    num_demos = int(sys.argv[5])
else:
    num_demos = 20
do_sample = True if num_return_sequences > 1 else False
# do_sample = True
temperature = 1
top_p=1.0
total_num_questions = 2000

if '13b' in model_name:
    generator = pipeline("text-generation", model=model_name, trust_remote_code=True, device=device, torch_dtype=torch.float16)
elif '30b' in model_name or '70b' in model_name:
    generator = pipeline("text-generation", model=model_name, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")
else:
    generator = pipeline("text-generation", model=model_name, trust_remote_code=True, device=device, torch_dtype=torch.float16)

if question_file.endswith('.txt'):
    questions = [item.split('\n')[0] for item in open(question_file).read().split('\n\n')]
elif question_file.endswith('.csv'):
    data_df = pd.read_csv(question_file)
    questions = data_df['question'].tolist()
else:
    print('question file format must be txt or csv')
    exit()

answers = []
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

# class Calibrator(torch.nn.Module):
#     def __init__(self, hidden_size):
#         super(Calibrator, self).__init__()
#         self.linear = torch.nn.Linear(hidden_size, 1, bias=False)
        

#     def forward(self, hidden_states):
#         l1 = self.linear(hidden_states)
#         return nn.functional.sigmoid(l1), None

class Calibrator(torch.nn.Module):
    def __init__(self, hidden_size):
        super(Calibrator, self).__init__()
        self.linear = torch.nn.Linear(hidden_size, 1, bias=False)
        self.layer_n = h_layer

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
calibrator = calibrator.half()

questions_for_writing = []
tokenizer = generator.tokenizer
stop_id = tokenizer.encode('\n')[-1]
start_time = time.time()
for question in questions:
    if question == '':
        continue
    print('-----------------------------------')
    print(question)
    gen_answers = []
    for i in range(num_return_sequences):
        gen_answers.extend(generator(prompt + question + '\nASSISTANT:', max_new_tokens=50, do_sample=do_sample, temperature=temperature, top_p=top_p, eos_token_id=stop_id, fact_guided_greedy_search=True, calibrator=calibrator))
        # gen_answers.extend(generator(prompt + question + '\nASSISTANT:', max_new_tokens=50, do_sample=do_sample, temperature=temperature, top_p=top_p, eos_token_id=stop_id, fact_guided_greedy_search=True, calibrator=None))
    now_answers = []
    for gen_answer in gen_answers:
        answer = gen_answer['generated_text']
        answer = answer[len(prompt + question + '\nASSISTANT:'):]
        answer = answer.split('\nUSER: ')[0].strip().replace('\n', ' ').replace('USER', '').split('User:')[0].strip()
        print(answer)
        now_answers.append(answer)
    if len(now_answers) == 1 and now_answers[0] == '':
        continue
    answers.append(now_answers)
    questions_for_writing.append(question)
    if len(answers) == total_num_questions:
        break
    if len(questions_for_writing) == 50:
        end_time = time.time()
        print('Time taken:', end_time - start_time)
        total_num_tokens = 0
        for ans in answers:
            num_tokens = len(generator.tokenizer(ans[0])['input_ids'])
            total_num_tokens += num_tokens
        print('Average token per second:', total_num_tokens / (end_time - start_time))
        exit()
    

df = pd.DataFrame({'question': questions_for_writing, 'answer': answers})
if do_sample:
    df.to_csv(question_file.replace('.txt', model_name.split('/')[-1] + 'temperature_' + str(temperature) + 'topp_' + str(top_p) + '_num_demos_' + str(num_demos) +'_answers.' + str(num_return_sequences) +'.fact_guided.csv'), index=False)
else:
    df.to_csv(question_file.replace('.txt', model_name.split('/')[-1]+ '.num_demos_' + str(num_demos) + '.greedy_search.fact_guided_seq_avg_greedy_linear.csv'), index=False)