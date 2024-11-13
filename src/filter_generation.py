import pandas as pd
import sys

greedy_search_file = sys.argv[1]
generation_file = sys.argv[2]
ground_truth_file = sys.argv[3]


greedy_data_df = pd.read_csv(greedy_search_file)
greedy_confidence = greedy_data_df['confidence']
greedy_correctness = greedy_data_df['correctness']
questions = greedy_data_df['question']
greedy_answer = greedy_data_df['answer']
q2gre_a = {}
for question, answer in zip(questions, greedy_answer):
    q2gre_a[question] = answer


generation_data_df = pd.read_csv(generation_file)
generation_confidence = generation_data_df['confidence']
generation_correctness = generation_data_df['correctness']
questions = generation_data_df['question']
generation_answer = generation_data_df['answer']
q2gen_a = {}
for question, answer in zip(questions, generation_answer):
    q2gen_a[question] = answer

filtered_correctness = []
our_ans = []
for ques, gre_conf, gre_cor, gen_conf, gen_cor in zip(questions, greedy_confidence, greedy_correctness, generation_confidence, generation_correctness):
    if gen_conf > gre_conf:
        filtered_correctness.append(gen_cor)
        our_ans.append(q2gen_a[ques])
    else:
        filtered_correctness.append(gre_cor)
        our_ans.append(q2gre_a[ques])

print(sum(filtered_correctness)/len(filtered_correctness))
print(sum(filtered_correctness), len(filtered_correctness))

q2a = {}
with open(ground_truth_file, 'r') as f:
    answers = [item for item in f.read().split('\n\n') if item != '']
    for answer in answers:
        qa = answer.split('\n')
        q = qa[0]
        a = qa[1]
        q2a[q] = a
ground_truths = []
for question in questions:
    ground_truths.append(q2a[question])

output_file = generation_file.replace('.csv', '.filtered.csv')
df = pd.DataFrame({'question': questions, 'answer': our_ans, 'ground_truth': ground_truths, 'correctness': filtered_correctness})
df.to_csv(output_file, index=False)