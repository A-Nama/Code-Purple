import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv
import os

# Load HF token from .env
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
hf_model_id = "aishanama/Code-Purple"  

# Load model and tokenizer from Hugging Face using auth token
tokenizer = AutoTokenizer.from_pretrained(hf_model_id, use_auth_token=hf_token)
model = AutoModelForSequenceClassification.from_pretrained(hf_model_id, use_auth_token=hf_token)
model.eval()

def predict_bias(paragraphs, max_length=512):
    inputs = tokenizer(paragraphs, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        scores = torch.sigmoid(outputs.logits).squeeze().numpy()
        return scores

def interpret_bias_score(score):
    if score < 0.4:
        return "✅ Very Inclusive"
    elif score < 0.6:
        return "🟡 Fairly Neutral"
    elif score < 0.8:
        return "🟠 Likely Biased"
    else:
        return "🔴 Highly Biased / Exclusionary"
