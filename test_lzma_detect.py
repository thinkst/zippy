#!/usr/bin/env python3

import pytest, os
from warnings import warn
from lzma_detect import run_on_file_chunked

AI_SAMPLE_DIR = 'samples/llm-generated/'
HUMAN_SAMPLE_DIR = 'samples/human-generated/'

ai_files = os.listdir(AI_SAMPLE_DIR)
human_files = os.listdir(HUMAN_SAMPLE_DIR)

FUZZINESS = 3
CONFIDENCE_THRESHOLD : float = 0.00 # What confidence to treat as error vs warning

def test_training_file():
    assert run_on_file_chunked('ai-generated.txt')[0] == 'AI', 'The training corpus should always be detected as AI-generated... since it is'

@pytest.mark.parametrize('f', human_files)
def test_human_samples(f):
    (classification, score) = run_on_file_chunked(HUMAN_SAMPLE_DIR + f, fuzziness=FUZZINESS)
    if score > CONFIDENCE_THRESHOLD:
        assert classification == 'Human', f + ' is a human-generated file, misclassified as AI-generated with confidence ' + str(round(score, 8))
    else:
        if classification != 'Human':
            warn("Misclassified " + f + " with score of: " + str(round(score, 8)))
        else:
            warn("Unable to confidently classify: " + f)

@pytest.mark.parametrize('f', ai_files)
def test_llm_sample(f):
   (classification, score) = run_on_file_chunked(AI_SAMPLE_DIR + f, fuzziness=FUZZINESS)
   if score > CONFIDENCE_THRESHOLD:
       assert classification == 'AI', f + ' is an LLM-generated file, misclassified as human-generated with confidence ' + str(round(score, 8))
   else:
       if classification != 'AI':
           warn("Misclassified " + f + " with score of: " + str(round(score, 8)))
       else:
           warn("Unable to confidently classify: " + f)