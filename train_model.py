import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from torch.optim import AdamW


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
df = pd.read_csv("bias_dataset.csv")

# Step 1: Group and order paragraphs by doc_id and para_id
df_sorted = df.sort_values(by=["doc_id", "para_id"])
grouped = df_sorted.groupby("doc_id")

# Step 2: Concatenate paragraphs and keep one score per document (average or max score)
docs = []
labels = []
for doc_id, group in grouped:
    full_text = " ".join(group["paragraph_text"].tolist())
    avg_score = group["bias_score"].mean()  # or use max(group["bias_score"])
    docs.append(full_text)
    labels.append(avg_score)

# Step 3: Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class DocumentBiasDataset(Dataset):
    def __init__(self, texts, scores):
        self.encodings = tokenizer(texts, truncation=True, padding=True, return_tensors="pt", max_length=512)
        self.labels = torch.tensor(scores, dtype=torch.float)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

# Step 4: Split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    docs, labels, test_size=0.2, random_state=42
)

train_dataset = DocumentBiasDataset(train_texts, train_labels)
val_dataset = DocumentBiasDataset(val_texts, val_labels)

# Step 5: Model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=1)
model.to(device)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

optimizer = AdamW(model.parameters(), lr=2e-5)

# Step 6: Training loop
model.train()
for epoch in range(3):
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"Epoch {epoch + 1} loss: {loss.item()}")

# Step 7: Validation
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

# Step 8: Save model
model.save_pretrained("./bias_bert_model")
tokenizer.save_pretrained("./bias_bert_model")
