#!/usr/bin/env python3

import pytest, os, jsonlines
from warnings import warn
from roberta_detect import run_on_file_chunked, run_on_text_chunked

AI_SAMPLE_DIR = 'samples/llm-generated/'
HUMAN_SAMPLE_DIR = 'samples/human-generated/'

MIN_LEN = 150

ai_files = os.listdir(AI_SAMPLE_DIR)
human_files = os.listdir(HUMAN_SAMPLE_DIR)

CONFIDENCE_THRESHOLD : float = 0.00 # What confidence to treat as error vs warning

def test_training_file():
    (classification, score) = run_on_file_chunked('ai-generated.txt')
    assert classification == 'AI', 'The training corpus should always be detected as AI-generated... since it is (score: ' + str(round(score, 8)) + ')'

@pytest.mark.parametrize('f', human_files)
def test_human_samples(f):
    (classification, score) = run_on_file_chunked(HUMAN_SAMPLE_DIR + f)
    if score > CONFIDENCE_THRESHOLD:
        assert classification == 'Human', f + ' is a human-generated file, misclassified as AI-generated with confidence ' + str(round(score, 8))
    else:
        if classification != 'Human':
            warn("Misclassified " + f + " with score of: " + str(round(score, 8)))
        else:
            warn("Unable to confidently classify: " + f)

@pytest.mark.parametrize('f', ai_files)
def test_llm_sample(f):
   (classification, score) = run_on_file_chunked(AI_SAMPLE_DIR + f)
   if score > CONFIDENCE_THRESHOLD:
       assert classification == 'AI', f + ' is an LLM-generated file, misclassified as human-generated with confidence ' + str(round(score, 8))
   else:
       if classification != 'AI':
           warn("Misclassified " + f + " with score of: " + str(round(score, 8)))
       else:
           warn("Unable to confidently classify: " + f)

HUMAN_JSONL_FILE = 'samples/webtext.test.jsonl'
human_samples = []
with jsonlines.open(HUMAN_JSONL_FILE) as reader:
    for obj in reader:
        human_samples.append(obj)

@pytest.mark.parametrize('i', human_samples[0:250])
def test_human_jsonl(i):
    (classification, score) = run_on_text_chunked(i.get('text', ''))
    assert classification == 'Human', HUMAN_JSONL_FILE + ':' + str(i.get('id')) + ' (len: ' + str(i.get('length', -1)) + ') is a human-generated sample, misclassified as AI-generated with confidence ' + str(round(score, 8))

AI_JSONL_FILE = 'samples/xl-1542M.test.jsonl'
ai_samples = []
with jsonlines.open(AI_JSONL_FILE) as reader:
    for obj in reader:
        ai_samples.append(obj)

@pytest.mark.parametrize('i', ai_samples[0:250])
def test_llm_jsonl(i):
    (classification, score) = run_on_text_chunked(i.get('text', ''))
    assert classification == 'AI', AI_JSONL_FILE + ':' + str(i.get('id')) + ' (text: ' + i.get('text', "").replace('\n', ' ')[:50] + ') is an LLM-generated sample, misclassified as human-generated with confidence ' + str(round(score, 8))

GPT3_JSONL_FILE = 'samples/GPT-3-175b_samples.jsonl'
gpt3_samples = []
with jsonlines.open(GPT3_JSONL_FILE) as reader:
    for o in reader:
        for l in o.split('<|endoftext|>'):
            if len(l) >= MIN_LEN:
                gpt3_samples.append(l)

@pytest.mark.parametrize('i', gpt3_samples)
def test_gpt3_jsonl(i):
    (classification, score) = run_on_text_chunked(i)
    assert classification == 'AI', GPT3_JSONL_FILE + ' is an LLM-generated sample, misclassified as human-generated with confidence ' + str(round(score, 8))

NEWS_JSONL_FILE = 'samples/news.jsonl'
news_samples = []
with jsonlines.open(NEWS_JSONL_FILE) as reader:
    for obj in reader:
        news_samples.append(obj)

@pytest.mark.parametrize('i', news_samples[0:250])
def test_humannews_jsonl(i):
    (classification, score) = run_on_text_chunked(i.get('human', ''))
    assert classification == 'Human', NEWS_JSONL_FILE + ' is a human-generated sample, misclassified as AI-generated with confidence ' + str(round(score, 8))

@pytest.mark.parametrize('i', news_samples[0:250])
def test_chatgptnews_jsonl(i):
    (classification, score) = run_on_text_chunked(i.get('chatgpt', ''))
    assert classification == 'AI', NEWS_JSONL_FILE + ' is a AI-generated sample, misclassified as human-generated with confidence ' + str(round(score, 8))
