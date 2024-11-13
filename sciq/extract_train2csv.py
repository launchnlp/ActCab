import pandas as pd

file_path = "./train.txt"
items = open(file_path).read().split('\n\n')

questions = []
answers = []
num_items = 2000    
for item in items:
    if len(item.split('\n')) < 2:
        continue
    if len(questions) >= num_items:
        break
    question = item.split('\n')[0]
    answer = [item.split('\n')[1]]
    questions.append(question)
    answers.append(answer)

df = pd.DataFrame({'question': questions, 'answer': answers})
df.to_csv("./train.csv", index=False)