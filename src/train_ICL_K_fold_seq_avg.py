import json
from nltk import sent_tokenize
from transformers import pipeline
import random
import torch
import torch.nn.functional as F
from tqdm import tqdm
import math
from torch import nn
import pandas as pd7
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import brier_score_loss
import scipy.stats
import time

seed = 42
random.seed(seed)

h_name_postfix = "_ICL_w_answer_hidden_states_4_answer_newsamples_normal_demos_sequence_true.pt"
i_name_postfix = "_ICL_w_answer_ids_4_answer_newsamples_normal_demos_sequence_true.pt"

model_name = sys.argv[1]
dataset_name = sys.argv[2]
h_layer = None
sample_or_greedy = "sample"
if len(sys.argv) > 3:
    sample_or_greedy = sys.argv[3]
    if sample_or_greedy == "greedy":
        h_name_postfix = "_ICL_w_answer_hidden_states_4_answer_normal_demos_greedy_search_sequence_true.pt"
        i_name_postfix = "_ICL_w_answer_ids_4_answer_normal_demos_greedy_search_sequence_true.pt"
    if sample_or_greedy == "merge":
        h_name_postfix = "_ICL_w_answer_hidden_states_4_answer_normal_demos_merge_sequence_true.pt"
        i_name_postfix = "_ICL_w_answer_ids_4_answer_normal_demos_merge_sequence_true.pt"
if len(sys.argv) > 4:
    num_demos = int(sys.argv[4])
    if sample_or_greedy == "sample":
        h_name_postfix = "_ICL_w_answer_hidden_states_4_answer_newsamples_normal_demos_sequence_num_demos" + str(num_demos) + ".pt"
        i_name_postfix = "_ICL_w_answer_ids_4_answer_newsamples_normal_demos_sequence_num_demos" + str(num_demos) + ".pt"
    if sample_or_greedy == "greedy":
        h_name_postfix = "_ICL_w_answer_hidden_states_4_answer_normal_demos_greedy_search_sequence_num_demos" + str(num_demos) + ".pt"
        i_name_postfix = "_ICL_w_answer_ids_4_answer_normal_demos_greedy_search_sequence_num_demos" + str(num_demos) + ".pt"
    if sample_or_greedy == "merge":
        h_name_postfix = "_ICL_w_answer_hidden_states_4_answer_normal_demos_merge_sequence_num_demos" + str(num_demos) + ".pt"
        i_name_postfix = "_ICL_w_answer_ids_4_answer_normal_demos_merge_sequence_num_demos" + str(num_demos) + ".pt"
if len(sys.argv) > 5:
    h_layer = int(sys.argv[5])
    if sample_or_greedy == "sample":
        h_name_postfix = "_ICL_w_answer_hidden_states_4_answer_newsamples_normal_demos_sequence_num_demos" + str(num_demos) + ".h_" + str(h_layer) + ".pt"
        i_name_postfix = "_ICL_w_answer_ids_4_answer_newsamples_normal_demos_sequence_num_demos" + str(num_demos) + ".h_" + str(h_layer) + ".pt"
    if sample_or_greedy == "greedy":
        h_name_postfix = "_ICL_w_answer_hidden_states_4_answer_normal_demos_greedy_search_sequence_num_demos" + str(num_demos) + ".h_" + str(h_layer) + ".pt"
        i_name_postfix = "_ICL_w_answer_ids_4_answer_normal_demos_greedy_search_sequence_num_demos" + str(num_demos) + ".h_" + str(h_layer) + ".pt"
    if sample_or_greedy == "merge":
        h_name_postfix = "_ICL_w_answer_hidden_states_4_answer_normal_demos_merge_sequence_num_demos" + str(num_demos) + ".h_" + str(h_layer) + ".pt"
        i_name_postfix = "_ICL_w_answer_ids_4_answer_normal_demos_merge_sequence_num_demos" + str(num_demos) + ".h_" + str(h_layer) + ".pt"

if ' ' not in dataset_name:
    hidden_states_path =  '../' + dataset_name + "/" + model_name.split('/')[-1] + h_name_postfix
    ids_path = '../' + dataset_name + "/" + model_name.split('/')[-1] + i_name_postfix
    if h_layer is not None:
        hidden_states_path =  '../' + dataset_name + "/" + f"/{model_name.split('/')[-1]}_multi-layer-logits-sampling/" + model_name.split('/')[-1] + h_name_postfix
        ids_path = '../' + dataset_name + "/" + f"/{model_name.split('/')[-1]}_multi-layer-logits-sampling/" + model_name.split('/')[-1] + i_name_postfix
else:
    dataset_names = dataset_name.split(' ')
    hidden_states_path = []
    ids_path = []
    for dn in dataset_names:
        hidden_states_path.append('../' + dn + "/" + model_name.split('/')[-1] + h_name_postfix)
        ids_path.append('../' + dn + "/" + model_name.split('/')[-1] + i_name_postfix)

device = 0
eval_interval = 30
if dataset_name == "truthfulqa":
    eval_interval = 10
loss_print_interval = 30
    
class Calibrator(torch.nn.Module):
    def __init__(self, hidden_size):
        super(Calibrator, self).__init__()
        self.linear = torch.nn.Linear(hidden_size, 1, bias=False)
        

    def forward(self, hidden_states):
        l1 = self.linear(hidden_states)
        return nn.functional.sigmoid(l1), None


init_temp = 1.0

# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
save_interval = 1000

if type(hidden_states_path) != list:
    c_w_hidden_states = torch.load(hidden_states_path)
    c_w_ids = torch.load(ids_path)
else:
    c_w_hidden_states = []
    c_w_ids = []
    for hsp, ip in zip(hidden_states_path, ids_path):
        c_w_hidden_states.append(torch.load(hsp))
        c_w_ids.append(torch.load(ip))

print(len(c_w_hidden_states))
print(len(c_w_ids))
# exit()
if type(hidden_states_path) != list:
    train_c_w_hidden_states = c_w_hidden_states[:int(len(c_w_hidden_states)*0.8)]
    train_c_w_ids = c_w_ids[:int(len(c_w_ids)*0.8)]
    val_c_w_hidden_states = c_w_hidden_states[int(len(c_w_hidden_states)*0.8):]
    val_c_w_ids = c_w_ids[int(len(c_w_ids)*0.8):]
else:
    train_c_w_hidden_states = []
    train_c_w_ids = []
    val_c_w_hidden_states = []
    val_c_w_ids = []
    for hsp, ip in zip(c_w_hidden_states, c_w_ids):
        train_c_w_hidden_states.extend(hsp[:int(len(hsp)*0.8)])
        train_c_w_ids.extend(ip[:int(len(ip)*0.8)])
        val_c_w_hidden_states.extend(hsp[int(len(hsp)*0.8):])
        val_c_w_ids.extend(ip[int(len(ip)*0.8):])

ref_val_c_w_hidden_states = torch.load(hidden_states_path)
ref_score_ids = torch.load(ids_path)
val_c_w_hidden_states = ref_val_c_w_hidden_states[int(len(ref_val_c_w_hidden_states)*0.8):]
val_c_w_ids = ref_score_ids[int(len(ref_score_ids)*0.8):]

batch_size = 128
eval_batch_size = 32

cur_idx = 0

best_calibrator_state_dict = None

huber_loss = torch.nn.HuberLoss(reduction='mean', delta=0.05)

# split the data into N slices
train_c_w_hidden_states_slices = []
train_c_w_ids_slices = []
num_fold = 10
for i in range(num_fold):
    train_c_w_hidden_states_slices.append(train_c_w_hidden_states[i*len(train_c_w_hidden_states)//num_fold:(i+1)*len(train_c_w_hidden_states)//num_fold])
    train_c_w_ids_slices.append(train_c_w_ids[i*len(train_c_w_ids)//num_fold:(i+1)*len(train_c_w_ids)//num_fold])

ece_loss_ids = []
first_factor_values = []
second_factor_values = []
# K fold training and inference
for slice_idx, (train_c_w_hidden_states, train_c_w_ids) in enumerate(zip(train_c_w_hidden_states_slices, train_c_w_ids_slices)):
    if "13b" not in model_name:
        calibrator = Calibrator(4096).cuda()
    else:
        calibrator = Calibrator(5120).cuda()
    optimizer = torch.optim.AdamW(calibrator.parameters(), lr=0.00001)
    calibrator.train()
    eval_c_w_hidden_states = train_c_w_hidden_states
    eval_c_w_ids = train_c_w_ids

    train_c_w_hidden_states = [train_c_w_hidden_states_slices[i] for i in range(num_fold) if i != slice_idx]
    train_c_w_hidden_states = [item for sublist in train_c_w_hidden_states for item in sublist]
    train_c_w_ids = [train_c_w_ids_slices[i] for i in range(num_fold) if i != slice_idx]
    train_c_w_ids = [item for sublist in train_c_w_ids for item in sublist]

    for cur_epoch in tqdm(range(2)):
        print("Epoch: ", cur_epoch)
        batched_hidden_states = []
        batched_ids = []

        lower_bound_loss = 9999

        # shuffle training data
        train_c_w_hidden_states, train_c_w_ids = zip(*random.sample(list(zip(train_c_w_hidden_states, train_c_w_ids)), len(train_c_w_hidden_states)))

        for idx in range(len(train_c_w_hidden_states)):
            hidden_states = [train_c_w_hidden_states[idx].to(device).mean(dim=1, keepdim=True).float()]
            ids = [train_c_w_ids[idx].float()]
            batched_hidden_states.extend(hidden_states)
            batched_ids.extend(ids)
            if len(batched_hidden_states) == batch_size:
                cur_idx += 1
                batched_hidden_states = torch.cat(batched_hidden_states, dim=0).to(device)
                batched_ids = torch.cat(batched_ids, dim=0).to(device)
                optimizer.zero_grad()
                logits, _ = calibrator(batched_hidden_states)
                # mse loss
                loss = F.mse_loss(logits.view(-1), batched_ids.view(-1))
                loss.backward()
                optimizer.step()
                # scheduler.step()
                batched_hidden_states = []
                batched_ids = []
                if cur_idx % loss_print_interval == 0:
                    print("Loss: ", loss)
                if cur_idx % eval_interval == 0:
                    total_eval_loss = 0
                    count = 0 
                    with torch.no_grad():
                        for val_idx in range(len(val_c_w_hidden_states)):
                            hidden_states = [val_c_w_hidden_states[val_idx].to(device).mean(dim=1, keepdim=True).float()]
                            ids = [val_c_w_ids[val_idx].float()]
                            batched_hidden_states.extend(hidden_states)
                            batched_ids.extend(ids)
                            if len(batched_hidden_states) == eval_batch_size:
                                batched_hidden_states = torch.cat(batched_hidden_states, dim=0).to(device)
                                batched_ids = torch.cat(batched_ids, dim=0).to(device)
                                logits, _ = calibrator(batched_hidden_states)
                                loss = F.mse_loss(logits.view(-1), batched_ids.view(-1))
                                # loss = huber_loss(logits.view(-1), batched_ids.view(-1))
                                
                                total_eval_loss += loss
                                count += 1
                                batched_hidden_states = []
                                batched_ids = []
                        if len(batched_hidden_states) > 0:
                            batched_hidden_states = torch.cat(batched_hidden_states, dim=0).to(device)
                            batched_ids = torch.cat(batched_ids, dim=0).to(device)
                            logits, _ = calibrator(batched_hidden_states)
                            loss = F.mse_loss(logits.view(-1), batched_ids.view(-1))
                            # loss = huber_loss(logits.view(-1), batched_ids.view(-1))
                            total_eval_loss += loss
                            count += 1
                            batched_hidden_states = []
                            batched_ids = []
                            print("Eval Loss: ", total_eval_loss/count)
                        if loss < lower_bound_loss:
                            lower_bound_loss = loss
                            # copy the best model
                            best_calibrator_state_dict = calibrator.state_dict()
                            # torch.save(calibrator.state_dict(), dataset_name + "/ICL_cls_ckpt/" + model_name.split('/')[-1] + "_calibrator_best.pt")
                # if cur_idx % save_interval == 0:
                    # print("Saving model: ", cur_idx)
                    # torch.save(calibrator.state_dict(), dataset_name + "/ICL_cls_ckpt/" + model_name.split('/')[-1] + "_calibrator_" + str(cur_idx) + ".pt")
        if len(batched_hidden_states) > 0:
            cur_idx += 1
            batched_hidden_states = torch.cat(batched_hidden_states, dim=0).to(device)
            batched_ids = torch.cat(batched_ids, dim=0).to(device)
            optimizer.zero_grad()
            logits, _ = calibrator(batched_hidden_states)
            loss = F.mse_loss(logits.view(-1), batched_ids.view(-1))
            loss.backward()
            optimizer.step()
            batched_hidden_states = []
            batched_ids = []
            if cur_idx % loss_print_interval == 0:
                print("Loss: ", loss)
        
    
    new_ids = []
    # do inference on training data
    with torch.no_grad():
        batched_hidden_states = []
        batched_ids = []
        for val_idx in range(len(eval_c_w_hidden_states)):
            hidden_states = [eval_c_w_hidden_states[val_idx].to(device).mean(dim=1, keepdim=True).float()]
            ids = [eval_c_w_ids[val_idx].float()]
            batched_hidden_states.extend(hidden_states)
            batched_ids.extend(ids)
        batched_hidden_states = torch.cat(batched_hidden_states, dim=0).to(device)
        batched_ids = torch.cat(batched_ids, dim=0).to(device)

        logits, _ = calibrator(batched_hidden_states)
        logits = logits.view(-1)
        batched_ids = batched_ids.view(-1)
        predictions = logits.cpu().tolist()
        scores = batched_ids.cpu().tolist()
        labels = [1 if random.random() < score else 0 for score in scores]
        first_factor_values.extend(predictions)
        second_factor_values.extend(labels)

idx2bins_id = []

# equal width binning
bin_size = 0.02
bins = [0.0]
while bins[-1] < 1.0:
    bins.append(bins[-1] + bin_size)
bins[-1] = 1.0
bin2score = {}
for bin_idx in range(len(bins)-1):
    bin2score[bin_idx] = []
for score, label in zip(first_factor_values, second_factor_values):
    bin_idx = int(score/bin_size)
    bin2score[bin_idx].append(label)
    idx2bins_id.append(bin_idx)
bin2acc = {}
for bin_idx in bin2score:
    bin2acc[bin_idx] = sum(bin2score[bin_idx])/len(bin2score[bin_idx]) if len(bin2score[bin_idx]) > 0 else 0
    print(bin2acc[bin_idx])


new_ids = [bin2acc[idx2bins_id[i]] for i in range(len(first_factor_values))]
train_c_w_ids = [torch.Tensor([id]) for id in new_ids]
ece_loss_ids.extend(train_c_w_ids)

print("*****************************Training*****************************")
time.sleep(2)
if "13b" not in model_name:
    calibrator = Calibrator(4096).cuda()
else:
    calibrator = Calibrator(5120).cuda()
optimizer = torch.optim.AdamW(calibrator.parameters(), lr=0.00001)
calibrator.train()
train_c_w_hidden_states = [item for sublist in train_c_w_hidden_states_slices for item in sublist]
train_c_w_ids = ece_loss_ids

ground_truth_ids = [item for sublist in train_c_w_ids_slices for item in sublist]
# adjusted_train_c_w_ids = []
# for idx, (ece_id, gt_id) in enumerate(zip(train_c_w_ids, ground_truth_ids)):
#     if ece_id < 0.5 and gt_id > 0.5:
#         ece_id = torch.Tensor([0.5])
#         # ece_id = (ece_id + gt_id) / 2
#     if ece_id > 0.5 and gt_id < 0.5:
#         # ece_id = (ece_id + gt_id) / 2
#         ece_id = torch.Tensor([0.5])
#     # ece_id = (ece_id + gt_id) / 2
#     adjusted_train_c_w_ids.append(ece_id)
# train_c_w_ids = adjusted_train_c_w_ids

for cur_epoch in tqdm(range(100)):
    print("Epoch: ", cur_epoch)
    batched_hidden_states = []
    batched_ids = []

    lower_bound_loss = 9999

    # shuffle training data
    train_c_w_hidden_states, train_c_w_ids = zip(*random.sample(list(zip(train_c_w_hidden_states, train_c_w_ids)), len(train_c_w_hidden_states)))

    for idx in range(len(train_c_w_hidden_states)):
        hidden_states = [train_c_w_hidden_states[idx].to(device).mean(dim=1, keepdim=True).float()]
        ids = [train_c_w_ids[idx].float()]
        batched_hidden_states.extend(hidden_states)
        batched_ids.extend(ids)
        if len(batched_hidden_states) == batch_size:
            cur_idx += 1
            batched_hidden_states = torch.cat(batched_hidden_states, dim=0).to(device)
            batched_ids = torch.cat(batched_ids, dim=0).to(device)
            optimizer.zero_grad()
            logits, _ = calibrator(batched_hidden_states)
            # mse loss
            loss = F.mse_loss(logits.view(-1), batched_ids.view(-1))
            loss.backward()
            optimizer.step()
            # scheduler.step()
            batched_hidden_states = []
            batched_ids = []
            if cur_idx % loss_print_interval == 0:
                print("Loss: ", loss)
            if cur_idx % eval_interval == 0:
                total_eval_loss = 0
                count = 0 
                with torch.no_grad():
                    for val_idx in range(len(val_c_w_hidden_states)):
                        hidden_states = [val_c_w_hidden_states[val_idx].to(device).mean(dim=1, keepdim=True).float()]
                        ids = [val_c_w_ids[val_idx].float()]
                        batched_hidden_states.extend(hidden_states)
                        batched_ids.extend(ids)
                        if len(batched_hidden_states) == eval_batch_size:
                            batched_hidden_states = torch.cat(batched_hidden_states, dim=0).to(device)
                            batched_ids = torch.cat(batched_ids, dim=0).to(device)
                            logits, _ = calibrator(batched_hidden_states)
                            loss = F.mse_loss(logits.view(-1), batched_ids.view(-1))
                            # loss = huber_loss(logits.view(-1), batched_ids.view(-1))
                            
                            total_eval_loss += loss
                            count += 1
                            batched_hidden_states = []
                            batched_ids = []
                    if len(batched_hidden_states) > 0:
                        batched_hidden_states = torch.cat(batched_hidden_states, dim=0).to(device)
                        batched_ids = torch.cat(batched_ids, dim=0).to(device)
                        logits, _ = calibrator(batched_hidden_states)
                        loss = F.mse_loss(logits.view(-1), batched_ids.view(-1))
                        # loss = huber_loss(logits.view(-1), batched_ids.view(-1))
                        total_eval_loss += loss
                        count += 1
                        batched_hidden_states = []
                        batched_ids = []
                        print("Eval Loss: ", total_eval_loss/count)
                    if loss < lower_bound_loss:
                        lower_bound_loss = loss
                        # copy the best model
                        best_calibrator_state_dict = calibrator.state_dict()

    if len(batched_hidden_states) > 0:
        cur_idx += 1
        batched_hidden_states = torch.cat(batched_hidden_states, dim=0).to(device)
        batched_ids = torch.cat(batched_ids, dim=0).to(device)
        optimizer.zero_grad()
        logits, _ = calibrator(batched_hidden_states)
        loss = F.mse_loss(logits.view(-1), batched_ids.view(-1))
        loss.backward()
        optimizer.step()
        batched_hidden_states = []
        batched_ids = []
        if cur_idx % loss_print_interval == 0:
            print("Loss: ", loss)


if num_demos == 20:
    if sample_or_greedy == 'sample':
        torch.save(calibrator.state_dict(), '../' + dataset_name + "/ICL_cls_ckpt/" + model_name.split('/')[-1] + "_calibrator_" + "intenvention_seq_avg" + ".K_fold.pt")
    elif sample_or_greedy == 'greedy':
        torch.save(calibrator.state_dict(), '../' + dataset_name + "/ICL_cls_ckpt/" + model_name.split('/')[-1] + "_calibrator_" + "intenvention_seq_avg_greedy_search" + ".K_fold.pt")
    elif sample_or_greedy == 'merge':
        torch.save(calibrator.state_dict(), '../' + dataset_name + "/ICL_cls_ckpt/" + model_name.split('/')[-1] + "_calibrator_" + "intenvention_seq_avg_merge" + ".K_fold.pt")
    else:
        print("Invalid sample_or_greedy")
        exit()
else:
    if sample_or_greedy == 'sample':
        torch.save(calibrator.state_dict(), '../' + dataset_name + "/ICL_cls_ckpt/" + model_name.split('/')[-1] + "_calibrator_" + "intenvention_seq_avg_num_demos" + str(num_demos) + ".K_fold.pt")
    elif sample_or_greedy == 'greedy':
        torch.save(calibrator.state_dict(), '../' + dataset_name + "/ICL_cls_ckpt/" + model_name.split('/')[-1] + "_calibrator_" + "intenvention_seq_avg_greedy_search_num_demos" + str(num_demos) + ".K_fold.pt")
    elif sample_or_greedy == 'merge':
        torch.save(calibrator.state_dict(), '../' + dataset_name + "/ICL_cls_ckpt/" + model_name.split('/')[-1] + "_calibrator_" + "intenvention_seq_avg_merge_num_demos" + str(num_demos) + ".K_fold.pt")
    else:
        print("Invalid sample_or_greedy")
        exit()

print("*****************************Evaluating*****************************")
# calibrator.load_state_dict(best_calibrator_state_dict)
batched_hidden_states = []
batched_ids = []
with torch.no_grad():
    for val_idx in range(len(val_c_w_hidden_states)):
        hidden_states = [val_c_w_hidden_states[val_idx].to(device).mean(dim=1, keepdim=True).float()]
        ids = [val_c_w_ids[val_idx].float()]
        batched_hidden_states.extend(hidden_states)
        batched_ids.extend(ids)
    batched_hidden_states = torch.cat(batched_hidden_states, dim=0).to(device)
    batched_ids = torch.cat(batched_ids, dim=0).to(device)
    logits, plot_data = calibrator(batched_hidden_states)
    logits = logits.view(-1)
    batched_ids = batched_ids.view(-1)
    for p, l in zip(logits.cpu().tolist(), batched_ids.cpu().tolist()):
        print(p, l)


# compute ece
predictions = logits.cpu().tolist()
scores = batched_ids.cpu().tolist()
labels = [1 if random.random() < score else 0 for score in scores]
first_factor_values = predictions
second_factor_values = labels

print("Computing ECE...")
bin_size = 0.1
bins = [0.0]
while bins[-1] < 1.0:
    bins.append(bins[-1] + bin_size)
bins[-1] = 1.0
bin2score = {}
for bin_idx in range(len(bins)-1):
    bin2score[bin_idx] = []
for score, label in zip(first_factor_values, second_factor_values):
    bin_idx = int(score/bin_size)
    bin2score[bin_idx].append(label)
bin2acc = {}
for bin_idx in bin2score:
    bin2acc[bin_idx] = sum(bin2score[bin_idx])/len(bin2score[bin_idx]) if len(bin2score[bin_idx]) > 0 else 0
    print(bin2acc[bin_idx])
ece = 0
for bin_idx in bin2acc:
    ece += abs(bin2acc[bin_idx] - (bins[bin_idx] + bins[bin_idx+1])/2) * len(bin2score[bin_idx])
ece /= len(first_factor_values)
print('ece: ', ece)

for bin_idx in bin2acc:
    print(len(bin2score[bin_idx]))

auroc = roc_auc_score(second_factor_values, first_factor_values)
print("auroc: ", auroc)

brier_score = brier_score_loss(second_factor_values, first_factor_values)
print("brier_score: ", brier_score)
