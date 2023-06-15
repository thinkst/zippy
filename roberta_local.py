#!/usr/bin/env python3

from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

if torch.cuda.is_available():
    DEVICE = 'cuda:0'
else:
    DEVICE = 'cpu'

tokenizer = AutoTokenizer.from_pretrained("roberta-base-openai-detector")
model = AutoModelForSequenceClassification.from_pretrained("roberta-base-openai-detector")
pipe = pipeline('text-classification', model=model, tokenizer=tokenizer, device=DEVICE)

def classify_text(s : List[str]) -> List[Tuple[str, float]]: 
    res = pipe(s)
    out = []
    for r in res:
        label = r['label']
        conf = r['score']
        if label == 'Real':
            out.append(('Human', conf))
        out.append(('AI', conf))
    return out

