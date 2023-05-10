#!/usr/bin/env python3

import pytest, os
from lzma_detect import run_on_file

def test_corpus():
    assert run_on_file('ai-generated.txt')[0] == 'AI', 'The training corpus should always be detected as AI-generated... since it is'