import sys
import pandas as pd

file_path = sys.argv[1]
model_key = sys.argv[2]
# llama2_7B, llama2_13B, llama3_8B
df_truthfulqa = pd.read_csv(file_path)

# question,answer
questions = df_truthfulqa['question'].tolist()
if model_key in df_truthfulqa.columns:
    truthfulqa_ans = df_truthfulqa[model_key].tolist()
else:
    truthfulqa_ans = df_truthfulqa['answer'].tolist()
# print(truthfulqa_ans)
truthfulqa_ans = []
for item in df_truthfulqa[model_key].tolist():
    if item != item:
        truthfulqa_ans.append('')
        continue
    if '[' in item:
        try:
            truthfulqa_ans.append(eval(item)[0])
        except:
            truthfulqa_ans.append(item)
    else:
        truthfulqa_ans.append(item)
temp_ans = []
for ans in truthfulqa_ans:
    if ';' in ans:
        ans = ans.split(';')[0]
        temp_ans.append(ans)
    else:
        temp_ans.append(ans)
truthfulqa_ans = temp_ans
        
# create a new dataframe
df = pd.DataFrame({'Question': questions, model_key: truthfulqa_ans})
output_file_path = file_path.replace('.csv', '.formatted.csv')
df.to_csv(output_file_path, index=False)