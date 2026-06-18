import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import torch
from torch.utils.data import Dataset

print("Step 1: Loading dataset...")
df = pd.read_csv("data/master_dataset.csv")
df = df.dropna(subset=["text", "sentiment"])
df["sentiment"] = df["sentiment"].astype(int)
print(f"Total samples: {len(df)}")
print(f"Sentiment distribution: {df['sentiment'].value_counts().to_dict()}")

# Split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["sentiment"])
print(f"Train: {len(train_df)} | Test: {len(test_df)}")

print("\nStep 2: Loading tokenizer...")
MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class PolicyDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(
            list(texts), truncation=True, padding=True, max_length=128
        )
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

train_dataset = PolicyDataset(train_df["text"], train_df["sentiment"])
test_dataset  = PolicyDataset(test_df["text"],  test_df["sentiment"])

print("\nStep 3: Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=3
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"f1": f1}

print("\nStep 4: Training...")
training_args = TrainingArguments(
    output_dir="models/sentiment",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_steps=10,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

print("\nStep 5: Evaluating...")
predictions = trainer.predict(test_dataset)
preds = np.argmax(predictions.predictions, axis=-1)
print(classification_report(test_df["sentiment"], preds,
      target_names=["negative", "neutral", "positive"]))

print("\nStep 6: Saving model...")
model.save_pretrained("models/sentiment_model")
tokenizer.save_pretrained("models/sentiment_model")
print("Model saved to models/sentiment_model")
