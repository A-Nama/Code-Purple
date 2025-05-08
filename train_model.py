import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import os

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the dataset
df = pd.read_csv("data/bias_dataset.csv")

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Dataset Class
class BiasDataset(Dataset):
    def __init__(self, paragraphs, scores):
        self.encodings = tokenizer(paragraphs, truncation=True, padding=True, return_tensors="pt")
        self.labels = torch.tensor(scores, dtype=torch.float)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

# Train-test split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["paragraph_text"].tolist(),
    df["bias_score"].tolist(),
    test_size=0.2,
    random_state=42
)

train_dataset = BiasDataset(train_texts, train_labels)
val_dataset = BiasDataset(val_texts, val_labels)

# Model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=1)
model.to(device)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# Optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Training Loop
epochs = 4
model.train()
for epoch in range(epochs):
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"Epoch {epoch + 1} loss: {loss.item()}")

# Evaluation
model.eval()
predictions, true_vals = [], []
with torch.no_grad():
    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        preds = outputs.logits.squeeze().cpu().numpy()
        labels = batch["labels"].cpu().numpy()
        predictions.extend(preds)
        true_vals.extend(labels)

rmse = np.sqrt(mean_squared_error(true_vals, predictions))
print(f"Validation RMSE: {rmse:.4f}")

# Save model
os.makedirs("model/bias_bert_model", exist_ok=True)
model.save_pretrained("model/bias_bert_model")
tokenizer.save_pretrained("model/bias_bert_model")
