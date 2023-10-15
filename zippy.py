#!/usr/bin/env python3

# Code to attempt to detect AI-generated text [relatively] quickly via compression ratios
# (C) 2023 Thinkst Applied Research, PTY
# Author: Jacob Torrey <jacob@thinkst.com>

import lzma, argparse, os, itertools
from zlib import compressobj, Z_FINISH
from brotli import compress as brotli_compress, MODE_TEXT
from numpy import array_split
import re, sys, statistics
from abc import ABC, abstractmethod
from enum import Enum
from math import ceil
from typing import List, Optional, Tuple, TypeAlias
from multiprocessing import Pool, cpu_count

Score : TypeAlias = tuple[str, float]

class CompressionEngine(Enum):
    LZMA = 1
    ZLIB = 2
    BROTLI = 3

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
    s = re.sub(r'[^0-9A-Za-z,\.\(\) \n]', '', s)#.lower()

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

class BrotliLlmDetector(AIDetector):
    '''Class providing functionality to attempt to detect LLM/generative AI generated text using the brotli compression algorithm'''
    def __init__(self, prelude_file : Optional[str] = None, prelude_str : Optional[str] = None, prelude_ratio : Optional[float] = None, preset : int = 8):
        self.PRESET = preset
        self.WIN_SIZE = 24
        self.BLOCK_SIZE = 0
        self.prelude_ratio = 0.0
        if prelude_ratio != None:
            self.prelude_ratio = prelude_ratio
        
        if prelude_file != None:
            with open(prelude_file) as fp:
                self.prelude_str = clean_text(fp.read())
            self.prelude_ratio = self._compress(self.prelude_str)
            return
    
        if prelude_str != None:
            self.prelude_str = prelude_str
            self.prelude_ratio = self._compress(self.prelude_str)

    def _compress(self, s : str) -> float:
        orig_len = len(s.encode())
        c_len = len(brotli_compress(s.encode(), mode=MODE_TEXT, quality=self.PRESET, lgwin=self.WIN_SIZE, lgblock=self.BLOCK_SIZE))
        return c_len / orig_len
    
    def score_text(self, sample: str) -> Score | None:
        '''
        Returns a tuple of a string (AI or Human) and a float confidence (higher is more confident) that the sample was generated 
        by either an AI or human. Returns None if it cannot make a determination
        '''
        if self.prelude_ratio == 0.0:
            return None
        sample_score = self._compress(self.prelude_str + sample)
        #print('Brotli: ' + str((self.prelude_ratio, sample_score)))
        delta = self.prelude_ratio - sample_score
        determination = 'AI'
        if delta < 0:
            determination = 'Human'

        return (determination, abs(delta * 100))

class ZlibLlmDetector(AIDetector):
    '''Class providing functionality to attempt to detect LLM/generative AI generated text using the zlib compression algorithm'''
    def __init__(self, prelude_file : Optional[str] = None, prelude_str : Optional[str] = None, prelude_ratio : Optional[float] = None, preset : int = 9):
        self.PRESET = preset
        self.WBITS = -15
        self.prelude_ratio = 0.0
        if prelude_ratio != None:
            self.prelude_ratio = prelude_ratio
        
        if prelude_file != None:
            with open(prelude_file) as fp:
                self.prelude_str = clean_text(fp.read())
            lines = self.prelude_str.split('\n')
            self.prelude_chunks = array_split(lines, ceil(len(self.prelude_str) / 2**abs(self.WBITS)))
            self.prelude_ratio = statistics.mean(map(lambda x: self._compress('\n'.join(list(x))), self.prelude_chunks))   
            return
         
        if prelude_str != None:
            self.prelude_str = prelude_str
            lines = self.prelude_str.split('\n')
            self.prelude_chunks = array_split(lines, ceil(len(self.prelude_str) / 2**abs(self.WBITS)))
            self.prelude_ratio = statistics.mean(map(lambda x: self._compress('\n'.join(list(x))), self.prelude_chunks))

    def _compress(self, s : str) -> float:
        orig_len = len(s.encode())
        c = compressobj(level=self.PRESET, wbits=self.WBITS, memLevel=9)
        bytes = c.compress(s.encode())
        bytes += c.flush(Z_FINISH)
        c_len = len(bytes)
        return c_len / orig_len
    
    def score_text(self, sample: str) -> Score | None:
        '''
        Returns a tuple of a string (AI or Human) and a float confidence (higher is more confident) that the sample was generated 
        by either an AI or human. Returns None if it cannot make a determination
        '''
        if self.prelude_ratio == 0.0:
            return None
        sample_score = statistics.mean(map(lambda x: self._compress('\n'.join(x) + sample), self.prelude_chunks))
        #print('ZLIB: ' + str((self.prelude_ratio, sample_score)))
        delta = self.prelude_ratio - sample_score
        determination = 'AI'
        if delta < 0:
            determination = 'Human'

        return (determination, abs(delta * 100))


class LzmaLlmDetector(AIDetector):
    '''Class providing functionality to attempt to detect LLM/generative AI generated text using the LZMA compression algorithm'''
    def __init__(self, prelude_file : Optional[str] = None, prelude_str : Optional[str] = None, prelude_ratio : Optional[float] = None, preset : int = 4) -> None:
        '''Initializes a compression with the passed prelude file, and optionally the number of digits to round to compare prelude vs. sample compression'''
        self.PRESET : int = preset
        self.c_buf : List[bytes] = []
        self.in_bytes : int = 0
        self.prelude_ratio : float = 0.0
        if prelude_ratio != None:
            self.prelude_ratio = prelude_ratio

        if prelude_file != None:
            # Read it once to get the default compression ratio for the prelude
            with open(prelude_file, 'r') as fp:
                self.prelude_str = fp.read()
            self.prelude_ratio = self._compress(self.prelude_str)
            return
            #print(prelude_file + ' ratio: ' + str(self.prelude_ratio))

        if prelude_str != None:
            if self.prelude_ratio == 0.0:
                self.prelude_ratio = self._compress(prelude_str)

    def _compress(self, s : str) -> float:
        orig_len = len(s.encode())
        c = lzma.LZMACompressor(preset=self.PRESET)
        bytes = c.compress(s.encode())
        bytes += c.flush()
        c_len = len(bytes)
        return c_len / orig_len

    def score_text(self, sample : str) -> Optional[Score]:
        '''
        Returns a tuple of a string (AI or Human) and a float confidence (higher is more confident) that the sample was generated 
        by either an AI or human. Returns None if it cannot make a determination
        '''
        if self.prelude_ratio == 0.0:
            return None
        #print('LZMA: ' + str((self.prelude_ratio, sample_score)))
        delta = self.prelude_ratio - self._compress(self.prelude_str + sample)
        determination = 'AI'
        if delta < 0:
            determination = 'Human'

        return (determination, abs(delta * 100))
        
class Zippy:
    '''
    Class to wrap the functionality of Zippy
    '''
    def __init__(self, engine : CompressionEngine = CompressionEngine.LZMA, preset : Optional[int] = None, prelude_file : str = PRELUDE_FILE) -> None:
        self.ENGINE = engine
        self.PRESET = preset
        self.PRELUDE_FILE = prelude_file
        if engine == CompressionEngine.LZMA:
            if self.PRESET:
                self.detector = LzmaLlmDetector(prelude_file=self.PRELUDE_FILE, preset=self.PRESET)
            else:
                self.detector = LzmaLlmDetector(prelude_file=self.PRELUDE_FILE)
        elif engine == CompressionEngine.BROTLI:
            if self.PRESET:
                self.detector = BrotliLlmDetector(prelude_file=self.PRELUDE_FILE, preset=self.PRESET)
            else:
                self.detector = BrotliLlmDetector(prelude_file=self.PRELUDE_FILE)
        elif engine == CompressionEngine.ZLIB:
            if self.PRESET:
                self.detector = ZlibLlmDetector(prelude_file=self.PRELUDE_FILE, preset=self.PRESET)
            else:
                self.detector = ZlibLlmDetector(prelude_file=self.PRELUDE_FILE)

    def run_on_file(self, filename : str) -> Optional[Score]:
        '''Given a filename (and an optional number of decimal places to round to) returns the score for the contents of that file'''
        with open(filename, 'r') as fp:
            txt = fp.read()
            #print('Calculating score for input of length ' + str(len(txt)))
        return self.detector.score_text(txt)

    def _score_chunk(self, c : str, prelude_file : Optional[str] = None, prelude_ratio : Optional[float] = None) -> Score:
        if prelude_file is None and prelude_ratio != None:
            self.detector.prelude_str = PRELUDE_STR
            self.detector.prelude_ratio = prelude_ratio

        return self.detector.score_text(c)

    def run_on_file_chunked(self, filename : str, chunk_size : int = 1500, prelude_ratio : Optional[float] = None) -> Optional[Score]:
        '''
        Given a filename (and an optional chunk size and number of decimal places to round to) returns the score for the contents of that file.
        This function chunks the file into at most chunk_size parts to score separately, then returns an average. This prevents a very large input
        being skewed because its compression ratio starts to overwhelm the prelude file.
        '''
        with open(filename, 'r') as fp:
            contents = fp.read()
        return self.run_on_text_chunked(contents, chunk_size, prelude_ratio=prelude_ratio)

    def run_on_text_chunked(self, s : str, chunk_size : int = 1500, prelude_file : Optional[str] = None, prelude_ratio : Optional[float] = None) -> Optional[Score]:
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
                for r in pool.starmap(self._score_chunk, zip(chunks, itertools.repeat(prelude_file), itertools.repeat(prelude_ratio))):
                    scores.append(r)
        else:
            for c in chunks:
                scores.append(self._score_chunk(c, prelude_file=prelude_file, prelude_ratio=prelude_ratio))
        ssum : float = 0.0
        for i, s in enumerate(scores):
            if s[0] == 'AI':
                ssum -= s[1] * (len(chunks[i]) / len(contents))
            else:
                ssum += s[1] * (len(chunks[i]) / len(contents))
        sa : float = ssum
        if sa < 0:
            return ('AI', abs(sa))
        else:
            return ('Human', abs(sa))

class EnsembledZippy:
    '''
    Class to wrap the functionality of Zippy into an ensemble
    '''
    def __init__(self) -> None:
        self.ENGINES = [CompressionEngine.LZMA, CompressionEngine.BROTLI, CompressionEngine.ZLIB]
        self.WEIGHTS = [.33, .33, .33]
        self.component_classifiers : list[AIDetector] = []
        for i, e in enumerate(self.ENGINES):
            self.component_classifiers.append(Zippy(e))
        
    def _combine_scores(self, scores : list[Score]) -> Score:
        ssum : float = 0.0
        for i, s in enumerate(scores):
            if s[0] == 'AI':
                ssum -= s[1] * self.WEIGHTS[i]
            else:
                ssum += s[1] * self.WEIGHTS[i]
        sa : float = ssum
        if sa < 0:
            return ('AI', abs(sa))
        else:
            return ('Human', abs(sa))

    def run_on_file(self, filename : str) -> Optional[Score]:
        '''Given a filename (and an optional number of decimal places to round to) returns the score for the contents of that file'''
        with open(filename, 'r') as fp:
            txt = fp.read()
        scores = []
        for c in self.component_classifiers:
            scores.append(c.score_text(txt))
        return self._combine_scores(scores)

    def _score_chunk(self, c : str, prelude_file : Optional[str] = None, prelude_ratio : Optional[float] = None) -> Score:
        scores = []
        for c in self.component_classifiers:
            scores.append(c.score_text(c))
        return self._combine_scores(scores)

    def run_on_file_chunked(self, filename : str, chunk_size : int = 1500, prelude_ratio : Optional[float] = None) -> Optional[Score]:
        '''
        Given a filename (and an optional chunk size and number of decimal places to round to) returns the score for the contents of that file.
        This function chunks the file into at most chunk_size parts to score separately, then returns an average. This prevents a very large input
        being skewed because its compression ratio starts to overwhelm the prelude file.
        '''
        with open(filename, 'r') as fp:
            contents = fp.read()
        return self.run_on_text_chunked(contents, chunk_size)

    def run_on_text_chunked(self, s : str, chunk_size : int = 1500, prelude_file : Optional[str] = None, prelude_ratio : Optional[float] = None) -> Optional[Score]:
        '''
        Given a string (and an optional chunk size and number of decimal places to round to) returns the score for the passed string.
        This function chunks the input into at most chunk_size parts to score separately, then returns an average. This prevents a very large input
        being skewed because its compression ratio starts to overwhelm the prelude file.
        '''
        scores = []
        for c in self.component_classifiers:
            scores.append(c.run_on_text_chunked(s, chunk_size=chunk_size))
        return self._combine_scores(scores)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", required=False, help="Preset to use with compressor, higher values are slower but provide better compression")
    parser.add_argument("-e", choices=['zlib', 'lzma', 'brotli', 'ensemble'], help='Which compression engine to use: lzma, zlib, brotli, or an ensemble of all engines', default='lzma', required=False)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-s", help='Read from stdin until EOF is reached instead of from a file', required=False, action='store_true')
    group.add_argument("sample_files", nargs='*', help='Text file(s) containing the sample to classify', default="")
    args = parser.parse_args()
    engine = 'lzma'
    if args.e:
        if args.e == 'lzma':
            engine = CompressionEngine.LZMA
        elif args.e == 'zlib':
            engine = CompressionEngine.ZLIB
        elif args.e == 'brotli':
            engine = CompressionEngine.BROTLI
        elif args.e == 'ensemble':
            engine = None
    if args.s:
        if engine:
            if args.p:
                z = Zippy(engine, preset=int(args.p))
            else:
                z = Zippy(engine)
        else:
            z = EnsembledZippy()
        print(str(z.run_on_text_chunked(''.join(list(sys.stdin)))))
    elif len(args.sample_files) == 0:
        print("Please call with either a list of text files to analyze, or the -s flag to classify stdin.\nCall with the -h flag for additional help.")
    else:
        if engine:
            if args.p:
                z = Zippy(engine, preset=int(args.p))
            else:
                z = Zippy(engine)
        else:
            z = EnsembledZippy()
        for f in args.sample_files:
            print(f)
            if os.path.isfile(f):
                print(str(z.run_on_file_chunked(f)))
