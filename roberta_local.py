#!/usr/bin/env python3

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("roberta-base-openai-detector")
model = AutoModelForSequenceClassification.from_pretrained("roberta-base-openai-detector")

def classify_text(s : str):
    inputs = tokenizer(s, return_tensors='pt')
    with torch.no_grad():
        logits = model(**inputs).logits
    pc = model.config.id2label[logits.argmax().item()]
    conf = max(torch.softmax(logits, dim=1).tolist()[0])
    if pc == 'Real':
        return ('Human', conf)
    return ('AI', conf)
