#!/usr/bin/env python3

import pytest, os, jsonlines
from warnings import warn
from lzma_detect import run_on_file_chunked, run_on_text_chunked, PRELUDE_STR, LzmaLlmDetector

AI_SAMPLE_DIR = 'samples/llm-generated/'
HUMAN_SAMPLE_DIR = 'samples/human-generated/'

ai_files = os.listdir(AI_SAMPLE_DIR)
human_files = os.listdir(HUMAN_SAMPLE_DIR)

FUZZINESS = 3
CONFIDENCE_THRESHOLD : float = 0.00 # What confidence to treat as error vs warning

PRELUDE_RATIO = LzmaLlmDetector(prelude_str=PRELUDE_STR).prelude_ratio

def test_training_file():
    assert run_on_file_chunked('ai-generated.txt')[0] == 'AI', 'The training corpus should always be detected as AI-generated... since it is'

@pytest.mark.parametrize('f', human_files)
def test_human_samples(f):
    (classification, score) = run_on_file_chunked(HUMAN_SAMPLE_DIR + f, fuzziness=FUZZINESS, prelude_ratio=PRELUDE_RATIO)
    if score > CONFIDENCE_THRESHOLD:
        assert classification == 'Human', f + ' is a human-generated file, misclassified as AI-generated with confidence ' + str(round(score, 8))
    else:
        if classification != 'Human':
            warn("Misclassified " + f + " with score of: " + str(round(score, 8)))
        else:
            warn("Unable to confidently classify: " + f)

@pytest.mark.parametrize('f', ai_files)
def test_llm_sample(f):
   (classification, score) = run_on_file_chunked(AI_SAMPLE_DIR + f, fuzziness=FUZZINESS, prelude_ratio=PRELUDE_RATIO)
   if score > CONFIDENCE_THRESHOLD:
       assert classification == 'AI', f + ' is an LLM-generated file, misclassified as human-generated with confidence ' + str(round(score, 8))
   else:
       if classification != 'AI':
           warn("Misclassified " + f + " with score of: " + str(round(score, 8)))
       else:
           warn("Unable to confidently classify: " + f)

MIN_LEN = 150

HUMAN_JSONL_FILE = 'samples/webtext.test.jsonl'
human_samples = []
with jsonlines.open(HUMAN_JSONL_FILE) as reader:
    for obj in reader:
        if obj.get('length', 0) >= MIN_LEN:
            human_samples.append(obj)

@pytest.mark.parametrize('i', human_samples[0:250])
def test_human_jsonl(i):
    (classification, score) = run_on_text_chunked(i.get('text', ''), fuzziness=FUZZINESS, prelude_ratio=PRELUDE_RATIO)
    assert classification == 'Human', HUMAN_JSONL_FILE + ':' + str(i.get('id')) + ' (len: ' + str(i.get('length', -1)) + ') is a human-generated sample, misclassified as AI-generated with confidence ' + str(round(score, 8))

AI_JSONL_FILE = 'samples/xl-1542M.test.jsonl'
ai_samples = []
with jsonlines.open(AI_JSONL_FILE) as reader:
    for obj in reader:
        if obj.get('length', 0) >= MIN_LEN:
            ai_samples.append(obj)

@pytest.mark.parametrize('i', ai_samples[0:250])
def test_llm_jsonl(i):
    (classification, score) = run_on_text_chunked(i.get('text', ''), fuzziness=FUZZINESS, prelude_ratio=PRELUDE_RATIO)
    assert classification == 'AI', AI_JSONL_FILE + ':' + str(i.get('id')) + ' (text: ' + i.get('text', "").replace('\n', ' ')[:50] + ') is an LLM-generated sample, misclassified as human-generated with confidence ' + str(round(score, 8))
