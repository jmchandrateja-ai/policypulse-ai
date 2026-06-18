from datasets import load_dataset

print("Step 1: Loading dataset...")
dataset = load_dataset("stanfordnlp/sst2")
print(dataset)
print()
print("Sample:", dataset["train"][0])
print()
print("Label names: 0=negative, 1=positive")
