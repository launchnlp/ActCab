from datasets import load_dataset

dataset = load_dataset('trivia_qa', 'rc.nocontext')

for split in dataset.keys():
    with open(f"{split}_aliases.txt", "w") as f:
        for example in dataset[split]:
            if len(example['answer']['aliases']) > 0:
                f.write(example["question"] + "\n" + ';'.join(example['answer']['aliases']) + "\n\n")