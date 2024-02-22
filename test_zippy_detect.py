#!/usr/bin/env python3

import pytest, os, jsonlines, csv
from warnings import warn
from zippy.zippy import Zippy, EnsembledZippy, PRELUDE_STR, LzmaLlmDetector, BrotliLlmDetector, ZlibLlmDetector, CompressionEngine
import zippy.zippy

AI_SAMPLE_DIR = 'samples/llm-generated/'
HUMAN_SAMPLE_DIR = 'samples/human-generated/'

MIN_LEN = 150
NUM_JSONL_SAMPLES = 500

ai_files = os.listdir(AI_SAMPLE_DIR)
human_files = os.listdir(HUMAN_SAMPLE_DIR)

CONFIDENCE_THRESHOLD : float = 0.00 # What confidence to treat as error vs warning

# Bool on whether to ensemble the models or run a single model
ENSEMBLE = False

if not ENSEMBLE:
    # What compression engine to use for the test
    ENGINE = CompressionEngine.LZMA

    if os.environ.get('ZIPPY_PRESET') != None:
        if ENGINE == CompressionEngine.LZMA:
            PRELUDE_RATIO = LzmaLlmDetector(prelude_str=PRELUDE_STR, preset=int(os.environ.get('ZIPPY_PRESET'))).prelude_ratio
        elif ENGINE == CompressionEngine.ZLIB:
            PRELUDE_RATIO = ZlibLlmDetector(prelude_str=PRELUDE_STR, preset=int(os.environ.get('ZIPPY_PRESET'))).prelude_ratio
        elif ENGINE == CompressionEngine.BROTLI:
            PRELUDE_RATIO = BrotliLlmDetector(prelude_str=PRELUDE_STR, preset=int(os.environ.get('ZIPPY_PRESET'))).prelude_ratio
    else:
        if ENGINE == CompressionEngine.LZMA:
            PRELUDE_RATIO = LzmaLlmDetector(prelude_str=PRELUDE_STR).prelude_ratio
        elif ENGINE == CompressionEngine.ZLIB:
            PRELUDE_RATIO = ZlibLlmDetector(prelude_str=PRELUDE_STR).prelude_ratio
        elif ENGINE == CompressionEngine.BROTLI:
            PRELUDE_RATIO = BrotliLlmDetector(prelude_str=PRELUDE_STR).prelude_ratio
    if os.environ.get('ZIPPY_PRESET') != None:
        zippy = Zippy(ENGINE, preset=int(os.environ.get('ZIPPY_PRESET'), normalize=True))
    else:
        zippy = Zippy(ENGINE, normalize=True)

else:
    zippy = EnsembledZippy()
    PRELUDE_RATIO = None

def test_training_file(record_property):
    (classification, score) = zippy.run_on_file_chunked('zippy/ai-generated.txt')
    record_property("score", str(score))
    assert classification == 'AI', 'The training corpus should always be detected as AI-generated... since it is'

@pytest.mark.parametrize('f', human_files)
def test_human_samples(f, record_property):
    (classification, score) = zippy.run_on_file_chunked(HUMAN_SAMPLE_DIR + f, prelude_ratio=PRELUDE_RATIO)
    record_property("score", str(score))
    if score > CONFIDENCE_THRESHOLD:
        assert classification == 'Human', f + ' is a human-generated file, misclassified as AI-generated with confidence ' + str(round(score, 8))
    else:
        if classification != 'Human':
            warn("Misclassified " + f + " with score of: " + str(round(score, 8)))
        else:
            warn("Unable to confidently classify: " + f)

@pytest.mark.parametrize('f', ai_files)
def test_llm_sample(f, record_property):
   (classification, score) = zippy.run_on_file_chunked(AI_SAMPLE_DIR + f, prelude_ratio=PRELUDE_RATIO)
   record_property("score", str(score))
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
        if obj.get('length', 0) >= MIN_LEN:
            human_samples.append(obj)

@pytest.mark.parametrize('i', human_samples[0:NUM_JSONL_SAMPLES])
def test_human_jsonl(i, record_property):
    (classification, score) = zippy.run_on_text_chunked(i.get('text', ''), prelude_ratio=PRELUDE_RATIO)
    record_property("score", str(score))
    assert classification == 'Human', HUMAN_JSONL_FILE + ':' + str(i.get('id')) + ' (len: ' + str(i.get('length', -1)) + ') is a human-generated sample, misclassified as AI-generated with confidence ' + str(round(score, 8))

# AI_JSONL_FILE = 'samples/xl-1542M.test.jsonl'
# ai_samples = []
# with jsonlines.open(AI_JSONL_FILE) as reader:
#     for obj in reader:
#         if obj.get('length', 0) >= MIN_LEN:
#             ai_samples.append(obj)

# @pytest.mark.parametrize('i', ai_samples[0:NUM_JSONL_SAMPLES])
# def test_gpt2_jsonl(i, record_property):
#     (classification, score) = run_on_text_chunked(i.get('text', ''), prelude_ratio=PRELUDE_RATIO)
#     record_property("score", str(score))
#     assert classification == 'AI', AI_JSONL_FILE + ':' + str(i.get('id')) + ' (text: ' + i.get('text', "").replace('\n', ' ')[:50] + ') is an LLM-generated sample, misclassified as human-generated with confidence ' + str(round(score, 8))

# GPT3_JSONL_FILE = 'samples/GPT-3-175b_samples.jsonl'
# gpt3_samples = []
# with jsonlines.open(GPT3_JSONL_FILE) as reader:
#     for o in reader:
#         for l in o.split('<|endoftext|>'):
#             if len(l) >= MIN_LEN:
#                 gpt3_samples.append(l)

# @pytest.mark.parametrize('i', gpt3_samples[0:NUM_JSONL_SAMPLES])
# def test_gpt3_jsonl(i, record_property):
#     (classification, score) = run_on_text_chunked(i, prelude_ratio=PRELUDE_RATIO)
#     record_property("score", str(score))
#     assert classification == 'AI', GPT3_JSONL_FILE + ' is an LLM-generated sample, misclassified as human-generated with confidence ' + str(round(score, 8))

NEWS_JSONL_FILE = 'samples/news.jsonl'
news_samples = []
with jsonlines.open(NEWS_JSONL_FILE) as reader:
    for obj in reader:
        news_samples.append(obj)

@pytest.mark.parametrize('i', news_samples[0:NUM_JSONL_SAMPLES])
def test_humannews_jsonl(i, record_property):
    (classification, score) = zippy.run_on_text_chunked(i.get('human', ''), prelude_ratio=PRELUDE_RATIO)
    record_property("score", str(score))
    assert classification == 'Human', NEWS_JSONL_FILE + ' is a human-generated sample, misclassified as AI-generated with confidence ' + str(round(score, 8))

@pytest.mark.parametrize('i', news_samples[0:NUM_JSONL_SAMPLES])
def test_chatgptnews_jsonl(i, record_property):
    (classification, score) = zippy.run_on_text_chunked(i.get('chatgpt', ''), prelude_ratio=PRELUDE_RATIO)
    record_property("score", str(score))
    assert classification == 'AI', NEWS_JSONL_FILE + ' is a AI-generated sample, misclassified as human-generated with confidence ' + str(round(score, 8))

CHEAT_HUMAN_JSONL_FILE = 'samples/ieee-init.jsonl'
ch_samples = []
with jsonlines.open(CHEAT_HUMAN_JSONL_FILE) as reader:
    for obj in reader:
        if len(obj.get('abstract', '')) >= MIN_LEN:
            ch_samples.append(obj)

@pytest.mark.parametrize('i', ch_samples[0:NUM_JSONL_SAMPLES])
def test_cheat_human_jsonl(i, record_property):
    (classification, score) = zippy.run_on_text_chunked(i.get('abstract', ''), prelude_ratio=PRELUDE_RATIO)
    record_property("score", str(score))
    assert classification == 'Human', CHEAT_HUMAN_JSONL_FILE + ':' + str(i.get('id')) + ' [' + str(len(i.get('abstract', ''))) + '] (title: ' + i.get('title', "").replace('\n', ' ')[:15] + ') is a human-generated sample, misclassified as AI-generated with confidence ' + str(round(score, 8))

CHEAT_GEN_JSONL_FILE = 'samples/ieee-chatgpt-generation.jsonl'
cg_samples = []
with jsonlines.open(CHEAT_GEN_JSONL_FILE) as reader:
    for obj in reader:
        if len(obj.get('abstract', '')) >= MIN_LEN:
            cg_samples.append(obj)

@pytest.mark.parametrize('i', cg_samples[0:NUM_JSONL_SAMPLES])
def test_cheat_generation_jsonl(i, record_property):
    (classification, score) = zippy.run_on_text_chunked(i.get('abstract', ''), prelude_ratio=PRELUDE_RATIO)
    record_property("score", str(score))
    assert classification == 'AI', CHEAT_GEN_JSONL_FILE + ':' + str(i.get('id')) + ' (title: ' + i.get('title', "").replace('\n', ' ')[:50] + ') is an LLM-generated sample, misclassified as human-generated with confidence ' + str(round(score, 8))

CHEAT_POLISH_JSONL_FILE = 'samples/ieee-chatgpt-polish.jsonl'
cp_samples = []
with jsonlines.open(CHEAT_POLISH_JSONL_FILE) as reader:
    for obj in reader:
        if len(obj.get('abstract', '')) >= MIN_LEN:
            cp_samples.append(obj)

@pytest.mark.parametrize('i', cp_samples[0:NUM_JSONL_SAMPLES])
def test_cheat_polish_jsonl(i, record_property):
    (classification, score) = zippy.run_on_text_chunked(i.get('abstract', ''), prelude_ratio=PRELUDE_RATIO)
    record_property("score", str(score))
    assert classification == 'AI', CHEAT_POLISH_JSONL_FILE + ':' + str(i.get('id')) + ' (title: ' + i.get('title', "").replace('\n', ' ')[:50] + ') is an LLM-generated sample, misclassified as human-generated with confidence ' + str(round(score, 8))

CHEAT_VICUNAGEN_JSONL_FILE = 'samples/ieee-vicuna-generation.jsonl'
vg_samples = []
with jsonlines.open(CHEAT_VICUNAGEN_JSONL_FILE) as reader:
    for obj in reader:
        if len(obj.get('abstract', '')) >= MIN_LEN:
            vg_samples.append(obj)

@pytest.mark.parametrize('i', vg_samples[0:NUM_JSONL_SAMPLES])
def test_vicuna_generation_jsonl(i, record_property):
    (classification, score) = zippy.run_on_text_chunked(i.get('abstract', ''), prelude_ratio=PRELUDE_RATIO)
    record_property("score", str(score))
    assert classification == 'AI', CHEAT_VICUNAGEN_JSONL_FILE + ':' + str(i.get('id')) + ' (title: ' + i.get('title', "").replace('\n', ' ')[:50] + ') is an LLM-generated sample, misclassified as human-generated with confidence ' + str(round(score, 8))

GPTZERO_EVAL_FILE = 'samples/gptzero_eval.csv'
ge_samples = []
with open(GPTZERO_EVAL_FILE) as fp:
    csvr = csv.DictReader(fp)
    for obj in csvr:
        if len(obj.get('Document', '')) >= MIN_LEN:
            ge_samples.append(obj)

@pytest.mark.parametrize('i', list(filter(lambda x: x.get('Label') == 'Human', ge_samples[0:NUM_JSONL_SAMPLES])))
def test_gptzero_eval_dataset_human(i, record_property):
    (classification, score) = zippy.run_on_text_chunked(i.get('Document', ''), prelude_ratio=PRELUDE_RATIO)
    record_property("score", str(score))
    assert classification == i.get('Label'), GPTZERO_EVAL_FILE + ':' + str(i.get('Index')) + ' was misclassified with confidence ' + str(round(score, 8))

@pytest.mark.parametrize('i', list(filter(lambda x: x.get('Label') == 'AI', ge_samples[0:NUM_JSONL_SAMPLES])))
def test_gptzero_eval_dataset_ai(i, record_property):
    (classification, score) = zippy.run_on_text_chunked(i.get('Document', ''), prelude_ratio=PRELUDE_RATIO)
    record_property("score", str(score))
    assert classification == i.get('Label'), GPTZERO_EVAL_FILE + ':' + str(i.get('Index')) + ' was misclassified with confidence ' + str(round(score, 8))
