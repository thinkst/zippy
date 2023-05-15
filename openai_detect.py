#!/usr/bin/env python3

import os, requests
from typing import Optional, Dict, Tuple

MODEL_NAME = 'model-detect-v2'
API_KEY = os.getenv('OPENAI_API_KEY')
API_URL = 'https://api.openai.com/v1/completions'

def make_req(text : str) -> Optional[Dict]:
    if len(text) < 1000:
        print("Input too short for OpenAI to classify")
        return None
    headers = {
        'authorization': 'Bearer ' + API_KEY,
        'origin': 'https://platform.openai.com',
        'openai-organization': 'org-gxAZne8U4jJ8pb632XJBLH1i'
    }
    data = {
        'prompt': text + '<disc_score|>',
        'max_tokens': 1,
        'temperature': 1,
        'top_p': 1,
        'n': 1,
        'model': MODEL_NAME,
        'stream': False,
        'stop': '\\n',
        'logprobs': 5
    }
    res = requests.post(API_URL, headers=headers, json=data)
    return res.json().get('choices', [None])[0]

def run_on_file(fn : str) -> Optional[Tuple[str, float]]:
    with open(fn, 'r') as fp:
        contents = fp.read()
    res = make_req(contents)
    if res is None:
        print("Unable to classify!")
        return None
    else:
        #print(res)
        if res.get('text') == '"':
            return ('AI', abs(res.get('logprobs').get('token_logprobs')[0]))
        elif res.get('text') == '!':
            return ('Human', abs(res.get('logprobs').get('token_logprobs')[0]))
        return None #res.get('text')
