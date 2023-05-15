#!/usr/bin/env python3

import pytest, os
from warnings import warn
from openai_detect import run_on_file

AI_SAMPLE_DIR = 'samples/llm-generated/'
HUMAN_SAMPLE_DIR = 'samples/human-generated/'

ai_files = os.listdir(AI_SAMPLE_DIR)
ai_files = filter(lambda f: os.path.getsize(AI_SAMPLE_DIR + f) >= 1000, ai_files)
human_files = os.listdir(HUMAN_SAMPLE_DIR)
human_files = filter(lambda f: os.path.getsize(HUMAN_SAMPLE_DIR + f) >= 1000, human_files)

def test_training_file():
    assert run_on_file('ai-generated.txt')[0] == 'AI', 'The training corpus should always be detected as AI-generated... since it is'

@pytest.mark.parametrize('f', human_files)
def test_human_samples(f):
    (classification, score) = run_on_file(HUMAN_SAMPLE_DIR + f)
    assert classification == 'Human', f + ' is a human-generated file, misclassified as AI-generated with confidence ' + str(round(score, 8))

@pytest.mark.parametrize('f', ai_files)
def test_llm_sample(f):
    (classification, score) = run_on_file(AI_SAMPLE_DIR + f)
    assert classification == 'AI', f + ' is an LLM-generated file, misclassified as human-generated with confidence ' + str(round(score, 8))
