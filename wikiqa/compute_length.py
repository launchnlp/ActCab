from datasets import load_dataset
from nltk.tokenize import word_tokenize

dataset = load_dataset('nq_open')
total_count = 0
total_length = 0


for split in dataset.keys():
    for example in dataset[split]:
        answer = example["answer"][0]
        tokens = word_tokenize(answer)
        total_count += 1
        total_length += len(tokens)
        if total_count > 100000:
            break

print(total_length / total_count)