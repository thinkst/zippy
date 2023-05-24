#!/usr/bin/env python3

import os, requests, re
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
    #print(str(res.status_code) + ' ' + res.text)
    res = res.json().get('choices', [None])[0]
    if res is None:
        return None
    if res.get('text') == '"':
        return ('AI', abs(res.get('logprobs').get('token_logprobs')[0]))
    elif res.get('text') == '!':
        return ('Human', abs(res.get('logprobs').get('token_logprobs')[0]))
    return None #res.get('text')

def run_on_file_chunked(fn : str, chunk_size : int = 4096) -> Optional[Tuple[str, float]]:
    with open(fn, 'r') as fp:
        contents = fp.read()
    return run_on_text_chunked(contents)

def run_on_text_chunked(contents : str, chunk_size : int = 4096) -> Optional[Tuple[str, float]]:
    	# Remove extra spaces and duplicate newlines.
	contents = re.sub(' +', ' ', contents)
	contents = re.sub('\t', '', contents)
	contents = re.sub('\n+', '\n', contents)
	contents = re.sub('\n ', '\n', contents)

	start = 0
	end = 0
	chunks = []
	while start + chunk_size < len(contents) and end != -1:
		end = contents.rfind(' ', start, start + chunk_size + 1)
		chunks.append(contents[start:end])
		start = end + 1
	chunks.append(contents[start:])
	scores = []
	for c in chunks:
		scores.append(make_req(c))
	ssum : float = 0.0
	for s in scores:
		if s is None:
			continue
		if s[0] == 'AI':
			ssum -= s[1]
		else:
			ssum += s[1]
	sa : float = ssum / len(scores)
	if sa < 0:
		return ('AI', abs(sa))
	else:
		return ('Human', abs(sa))

