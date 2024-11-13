import pandas as pd
import sys

file_path = sys.argv[1]
model_key = sys.argv[2]

informativeness = 0
truthfulness = 0
df = pd.read_csv(file_path)

truth_key = model_key + " GPT-judge acc"
info_key = model_key + " GPT-info acc"
for truth, info in zip(df[truth_key], df[info_key]):
    informativeness += info
    truthfulness += truth

print("Informativeness: ", informativeness / len(df))
print("Truthfulness: ", truthfulness / len(df))
print("Truth*Info: ", informativeness / len(df) * truthfulness / len(df))