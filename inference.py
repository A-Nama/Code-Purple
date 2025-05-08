import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = "model/bias_bert_model" 

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

def predict_bias(paragraphs, max_length=512):
    inputs = tokenizer(paragraphs, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        scores = torch.sigmoid(outputs.logits).squeeze().numpy()
        return scores  # Between 0 and 1 now
