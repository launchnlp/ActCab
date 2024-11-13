from datasets import load_dataset
import json

dataset = load_dataset("wiki_qa")

correct_output = "validation_correct_answers.json"
wrong_output = "validation_wrong_answers.json"

split = 'validatiown'
with open(correct_output, "w") as c_f, open(wrong_output, "w") as w_f:
    q2correct_answers = {}
    q2wrong_answers = {}
    for example in dataset[split]:
        if example["label"] == 1:
            if example["question"] not in q2correct_answers:
                q2correct_answers[example["question"]] = []
            q2correct_answers[example["question"]].append(example["answer"])
        else:
            if example["question"] not in q2wrong_answers:
                q2wrong_answers[example["question"]] = []
            q2wrong_answers[example["question"]].append(example["answer"])
    for q in q2correct_answers:
        c_f.write(json.dumps({"question": q, "correct_answer": q2correct_answers[q]}) + "\n")
    for q in q2wrong_answers:
        w_f.write(json.dumps({"question": q, "wrong_answer": q2wrong_answers[q]}) + "\n")