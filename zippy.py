#!/usr/bin/env python3

# Code to attempt to detect AI-generated text [relatively] quickly via compression ratios
# (C) 2023 Thinkst Applied Research, PTY
# Author: Jacob Torrey <jacob@thinkst.com>

import lzma, argparse, os, itertools
from zlib import compressobj, Z_FINISH
import re, sys
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional, Tuple, TypeAlias
from multiprocessing import Pool, cpu_count

Score : TypeAlias = tuple[str, float]

class CompressionEngine(Enum):
    LZMA = 1
    ZLIB = 2

ENGINE : CompressionEngine = CompressionEngine.ZLIB

def clean_text(s : str) -> str:
    '''
    Removes formatting and other non-content data that may skew compression ratios (e.g., duplicate spaces)
    '''
    # Remove extra spaces and duplicate newlines.
    s = re.sub(' +', ' ', s)
    s = re.sub('\t', '', s)
    s = re.sub('\n+', '\n', s)
    s = re.sub('\n ', '\n', s)
    s = re.sub(' \n', '\n', s)

    # Remove non-alphanumeric chars
    s = re.sub('[^0-9A-Za-z,\.\(\) \n]', '', s)#.lower()

    return s

# The prelude file is a text file containing only AI-generated text, it is used to 'seed' the LZMA dictionary
PRELUDE_FILE : str = 'ai-generated.txt'
with open(PRELUDE_FILE, 'r') as fp:
    PRELUDE_STR = clean_text(fp.read())

class AIDetector(ABC):
    '''
    Base class for AI detection
    '''
    @abstractmethod
    def score_text(self, sample : str) -> Optional[Score]:
        pass

class ZlibLlmDetector(AIDetector):
    '''Class providing functionality to attempt to detect LLM/generative AI generated text using the zlib compression algorithm'''
    def __init__(self, prelude_file : Optional[str] = None, prelude_str : Optional[str] = None, prelude_ratio : Optional[float] = None):
        self.PRESET = 9
        self.WBITS = -15
        self.prelude_ratio = 0.0
        if prelude_ratio != None:
            self.prelude_ratio = prelude_ratio
        
        if prelude_file != None:
            with open(prelude_file) as fp:
                self.prelude_str = fp.read()
                self.prelude_ratio = self._compress(self.prelude_str)
    
        if prelude_str != None:
            self.prelude_str = prelude_str
            self.prelude_ratio = self._compress(self.prelude_str)

    def _compress(self, s : str) -> float:
        orig_len = len(s.encode())
        c = compressobj(level=self.PRESET, wbits=self.WBITS, memLevel=9)
        bytes = c.compress(s.encode())
        bytes += c.flush(Z_FINISH)
        c_len = len(bytes)
        #c_len = len(compress(s.encode(), level=self.PRESET, wbits=self.WBITS))
        return c_len / orig_len
    
    def score_text(self, sample: str) -> Score | None:
        '''
        Returns a tuple of a string (AI or Human) and a float confidence (higher is more confident) that the sample was generated 
        by either an AI or human. Returns None if it cannot make a determination
        '''
        if self.prelude_ratio == 0.0:
            return None
        sample_score = self._compress(self.prelude_str + sample)
        #print(str((self.prelude_ratio, sample_score)))
        delta = self.prelude_ratio - sample_score
        determination = 'AI'
        if delta < 0:
            determination = 'Human'

        return (determination, abs(delta * 100))


class LzmaLlmDetector(AIDetector):
    '''Class providing functionality to attempt to detect LLM/generative AI generated text using the LZMA compression algorithm'''
    def __init__(self, prelude_file : Optional[str] = None, prelude_str : Optional[str] = None, prelude_ratio : Optional[float] = None) -> None:
        '''Initializes a compression with the passed prelude file, and optionally the number of digits to round to compare prelude vs. sample compression'''
        self.PRESET : int = 2
        self.comp = lzma.LZMACompressor(preset=self.PRESET)
        self.c_buf : List[bytes] = []
        self.in_bytes : int = 0
        self.prelude_ratio : float = 0.0
        if prelude_ratio != None:
            self.prelude_ratio = prelude_ratio
        self.SHORT_SAMPLE_THRESHOLD : int = 350 # What sample length is considered "short"

        if prelude_file != None:
            # Read it once to get the default compression ratio for the prelude
            with open(prelude_file, 'r') as fp:
                self._compress_str(fp.read())
            self.prelude_ratio = self._finalize()
            #print(prelude_file + ' ratio: ' + str(self.prelude_ratio))
            # Redo this to prime the compressor
            self.comp = lzma.LZMACompressor(preset=self.PRESET)
            with open(prelude_file, 'r') as fp:
                self._compress_str(fp.read())

        if prelude_str != None:
            if self.prelude_ratio == 0.0:
                self._compress_str(prelude_str)
                self.prelude_ratio = self._finalize()
                self.comp = lzma.LZMACompressor(preset=self.PRESET)
            self._compress_str(prelude_str)
            
    def _compress_str(self, s : str) -> None:
        '''
        Internal helper function to compress a string
        '''
        strb : bytes = s.encode('ascii', errors='ignore')
        self.c_buf.append(self.comp.compress(strb))
        self.in_bytes += len(strb)
    
    def _finalize(self) -> float:
        '''
        Finalizes an LZMA compression cycle and returns the percentage compression ratio
        
        post: _ >= 0
        '''
        self.c_buf.append(self.comp.flush())
        compressed_size : int = len(b''.join(self.c_buf))
        if self.in_bytes == 0:
            return 0.0
        score = compressed_size / self.in_bytes
        self.in_bytes = 0
        self.c_buf = []
        return score
    
    def get_compression_ratio(self, s : str) -> Tuple[float, float]:
        '''
        Returns a tuple of floats with the compression ratio of the prelude (0 if no prelude) and passed string
        '''
        self._compress_str(s)
        return (self.prelude_ratio, self._finalize())

    def score_text(self, sample : str) -> Optional[Score]:
        '''
        Returns a tuple of a string (AI or Human) and a float confidence (higher is more confident) that the sample was generated 
        by either an AI or human. Returns None if it cannot make a determination
        '''
        if self.prelude_ratio == 0.0:
            return None
        (prelude_score, sample_score) = self.get_compression_ratio(sample)
        print(str((self.prelude_ratio, sample_score)))
        delta = prelude_score - sample_score
        determination = 'AI'
        if delta < 0:
            determination = 'Human'

        return (determination, abs(delta * 100))
        
def run_on_file(filename : str) -> Optional[Score]:
    '''Given a filename (and an optional number of decimal places to round to) returns the score for the contents of that file'''
    with open(filename, 'r') as fp:
        if ENGINE == CompressionEngine.LZMA:
            l = LzmaLlmDetector(prelude_file=PRELUDE_FILE)
        elif ENGINE == CompressionEngine.ZLIB:
            l = ZlibLlmDetector(prelude_file=PRELUDE_FILE)
        txt = fp.read()
        #print('Calculating score for input of length ' + str(len(txt)))
        return l.score_text(txt)

def _score_chunk(c : str, prelude_file : Optional[str] = None, prelude_ratio : Optional[float] = None) -> Score:
        if prelude_file != None:
            if ENGINE == CompressionEngine.LZMA:
                l = LzmaLlmDetector(prelude_file=prelude_file)
            if ENGINE == CompressionEngine.ZLIB:
                l = ZlibLlmDetector(prelude_file=prelude_file)
        else:
            if ENGINE == CompressionEngine.LZMA:
                l = LzmaLlmDetector(prelude_str=PRELUDE_STR, prelude_ratio=prelude_ratio)
            if ENGINE == CompressionEngine.ZLIB:
                l = ZlibLlmDetector(prelude_str=PRELUDE_STR, prelude_ratio=prelude_ratio)
        return l.score_text(c)

def run_on_file_chunked(filename : str, chunk_size : int = 1500, prelude_ratio : Optional[float] = None) -> Optional[Score]:
    '''
    Given a filename (and an optional chunk size and number of decimal places to round to) returns the score for the contents of that file.
    This function chunks the file into at most chunk_size parts to score separately, then returns an average. This prevents a very large input
    being skewed because its compression ratio starts to overwhelm the prelude file.
    '''
    with open(filename, 'r') as fp:
        contents = fp.read()
    return run_on_text_chunked(contents, chunk_size, prelude_ratio=prelude_ratio)

def run_on_text_chunked(s : str, chunk_size : int = 1500, prelude_file : Optional[str] = None, prelude_ratio : Optional[float] = None) -> Optional[Score]:
    '''
    Given a string (and an optional chunk size and number of decimal places to round to) returns the score for the passed string.
    This function chunks the input into at most chunk_size parts to score separately, then returns an average. This prevents a very large input
    being skewed because its compression ratio starts to overwhelm the prelude file.
    '''
    contents = clean_text(s)

    start = 0
    end = 0
    chunks = []
    while start + chunk_size < len(contents) and end != -1:
        end = contents.rfind(' ', start, start + chunk_size + 1)
        chunks.append(contents[start:end])
        start = end + 1
    chunks.append(contents[start:])
    scores = []
    if len(chunks) > 2:
        with Pool(cpu_count()) as pool:
            for r in pool.starmap(_score_chunk, zip(chunks, itertools.repeat(prelude_file), itertools.repeat(prelude_ratio))):
                scores.append(r)
    else:
        for c in chunks:
            scores.append(_score_chunk(c, prelude_file=prelude_file, prelude_ratio=prelude_ratio))
    ssum : float = 0.0
    for i, s in enumerate(scores):
        if s[0] == 'AI':
            ssum -= s[1] * (len(chunks[i]) / len(contents))
        else:
            ssum += s[1] * (len(chunks[i]) / len(contents))
    sa : float = ssum# / len(scores)
    if sa < 0:
        return ('AI', abs(sa))
    else:
        return ('Human', abs(sa))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", choices=['zlib', 'lzma'], help='Which compression engine to use: lzma or zlib', default='lzma', required=False)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-s", help='Read from stdin until EOF is reached instead of from a file', required=False, action='store_true')
    group.add_argument("sample_files", nargs='*', help='Text file(s) containing the sample to classify', default="")
    args = parser.parse_args()
    if args.e:
        if args.e == 'lzma':
            ENGINE = CompressionEngine.LZMA
        elif args.e == 'zlib':
            ENGINE = CompressionEngine.ZLIB
    if args.s:
        print(str(run_on_text_chunked(''.join(list(sys.stdin)))))
    elif len(args.sample_files) == 0:
        print("Please call with either a list of text files to analyze, or the -s flag to classify stdin.\nCall with the -h flag for additional help.")
    else:
        for f in args.sample_files:
            print(f)
            if os.path.isfile(f):
                print(str(run_on_file_chunked(f)))
