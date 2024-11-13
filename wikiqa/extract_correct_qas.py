from datasets import load_dataset

dataset = load_dataset("wiki_qa")

for split in dataset.keys():
    with open(f"data/{split}.txt", "w") as f:
        for example in dataset[split]:
            if example["label"] == 1:
                f.write(example["question"] + "\n" + example["answer"] + "\n\n")