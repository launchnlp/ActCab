# this file is used to process the train.csv and test.csv files to train.txt and test.txt files

import json
import pandas as pd

train_file = "train.csv"
test_file = "test.csv"

# process test data
test_df = pd.read_csv(test_file)
questions = test_df['Question'].tolist()
answers = test_df['Answer'].tolist()
with open("test.txt", "w") as f:
    for i in range(len(questions)):
        f.write(questions[i] + "\n" + answers[i] + "\n\n")

# process train data
train_df = pd.read_csv(train_file)
questions = train_df['Question'].tolist()
correct_answers = train_df['Answer'].tolist()
false_answers = train_df['false_answer'].tolist()
with open("train.txt", "w") as f:
    for i in range(len(questions)):
        f.write(questions[i] + "\n" + correct_answers[i].split(';')[0] + "\n\n")

questions_write = questions
answers_write = []
correctness_write = []
for correct_answer, false_answer in zip(correct_answers, false_answers):
    cor_answer = correct_answer.split(';') + false_answer.split(';')
    answers_write.append(cor_answer)
    correctness_write.append([1] * len(correct_answer.split(';')) + [0] * len(false_answer.split(';')))
df = pd.DataFrame({'question': questions_write, 'answer': answers_write})
df.to_csv('trainLlama-2-7b-hftemperature_1topp_1.0_num_demos_5_answers.4.csv', index=False)

question_write_for_correctness_file = []
correctness_write_for_correctness_file = []
for question, correctness in zip(questions_write, correctness_write):
    for correct in correctness:
        question_write_for_correctness_file.append(question)
        correctness_write_for_correctness_file.append(correct)
df = pd.DataFrame({'question': question_write_for_correctness_file, 'correctness': correctness_write_for_correctness_file})
df.to_csv('trainLlama-2-7b-hftemperature_1topp_1.0_num_demos_5_answers.4.IK_set.csv', index=False)