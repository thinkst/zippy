#!/usr/bin/env python3

# Code to attempt to detect AT-generated text [relatively] quickly via compression ratios
# (C) 2023 Thinkst Applied Research, PTY
# Author: Jacob Torrey <jacob@thinkst.com>

import lzma, argparse, os
import re
from typing import List, Optional, Tuple

# The prelude file is a text file containing only AI-generated text, it is used to 'seed' the LZMA dictionary
PRELUDE_FILE : str = 'ai-generated.txt'
with open(PRELUDE_FILE, 'r') as fp:
    PRELUDE_STR = fp.read()

class LzmaLlmDetector:
    '''Class providing functionality to attempt to detect LLM/generative AI generated text using the LZMA compression algorithm'''
    def __init__(self, prelude_file : Optional[str] = None, fuzziness_digits : int = 3, prelude_str : Optional[str] = None) -> None:
        '''Initializes a compression with the passed prelude file, and optionally the number of digits to round to compare prelude vs. sample compression'''
        self.PRESET : int = 0
        self.comp = lzma.LZMACompressor(preset=self.PRESET)
        self.c_buf : List[bytes] = []
        self.in_bytes : int = 0
        self.prelude_ratio : float = 0.0
        self.FUZZINESS_THRESHOLD = fuzziness_digits
        self.SHORT_SAMPLE_THRESHOLD : int = 350 # What sample length is considered "short"

        if prelude_file != None:
            # Read it once to get the default compression ratio for the prelude
            with open(prelude_file, 'r') as fp:
                self._compress_str(fp.read())
            self.prelude_ratio = self._finalize()
            # Redo this to prime the compressor
            self.comp = lzma.LZMACompressor(preset=self.PRESET)
            with open(prelude_file, 'r') as fp:
                self._compress_str(fp.read())

        if prelude_str != None:
            self._compress_str(prelude_str)
            self.prelude_ratio = self._finalize()
            self.comp = lzma.LZMACompressor(preset=self.PRESET)
            self._compress_str(prelude_str)
    
    def _compress_str(self, s : str) -> None:
        '''
        Internal helper function to compress a string
        '''
        strb : bytes = s.encode('utf-8')
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
        return compressed_size / self.in_bytes
    
    def get_compression_ratio(self, s : str) -> Tuple[float, float]:
        '''
        Returns a tuple of floats with the compression ratio of the prelude (0 if no prelude) and passed string
        '''
        self._compress_str(s)
        return (self.prelude_ratio, self._finalize())

    def score_text(self, sample : str) -> Optional[Tuple[str, float]]:
        '''
        Returns a tuple of a string (AI or Human) and a float confidence (higher is more confident) that the sample was generated 
        by either an AI or human. Returns None if it cannot make a determination
        '''
        if self.prelude_ratio == 0.0:
            return None
        (prelude_score, sample_score) = self.get_compression_ratio(sample)
        #print(str((prelude_score, sample_score)))
        delta = prelude_score - sample_score
        determination = 'AI'
        if delta < 0:
            determination = 'Human'

        # If the sample doesn't 'move the needle', it's very close
        if round(delta, self.FUZZINESS_THRESHOLD) == 0 and len(sample) >= self.SHORT_SAMPLE_THRESHOLD:
            #print('Sample len to default to AI: ' + str(len(sample)))
            determination = 'AI'
        if round(delta, self.FUZZINESS_THRESHOLD) == 0 and len(sample) < self.SHORT_SAMPLE_THRESHOLD:
            #print('Sample len to default to Human: ' + str(len(sample)))
            determination = 'Human'
        #if abs(delta * 100) < .1 and determination == 'AI':
        #    print("Very low-confidence determination of: " + determination)
        return (determination, abs(delta * 100))
        
def run_on_file(filename : str, fuzziness : int = 3) -> Optional[Tuple[str, float]]:
    '''Given a filename (and an optional number of decimal places to round to) returns the score for the contents of that file'''
    with open(filename, 'r') as fp:
        l = LzmaLlmDetector(PRELUDE_FILE, fuzziness)
        txt = fp.read()
        #print('Calculating score for input of length ' + str(len(txt)))
        return l.score_text(txt)

def run_on_file_chunked(filename : str, chunk_size : int = 1024, fuzziness : int = 3) -> Optional[Tuple[str, float]]:
    '''
    Given a filename (and an optional chunk size and number of decimal places to round to) returns the score for the contents of that file.
    This function chunks the file into at most chunk_size parts to score separately, then returns an average. This prevents a very large input
    being skewed because its compression ratio starts to overwhelm the prelude file.
    '''
    with open(filename, 'r') as fp:
        contents = fp.read()
    
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
        l = LzmaLlmDetector(fuzziness_digits=fuzziness, prelude_str=PRELUDE_STR)
        scores.append(l.score_text(c))
    ssum : float = 0.0
    for s in scores:
        if s[0] == 'AI':
            ssum -= s[1]
        else:
            ssum += s[1]
    sa : float = ssum / len(scores)
    if sa < 0:
        return ('AI', abs(sa))
    else:
        return ('Human', abs(sa))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("sample_files", nargs='+', help='Text file(s) containing the sample to classify')
    args = parser.parse_args()

    for f in args.sample_files:
        print(f)
        if os.path.isfile(f):
            print(str(run_on_file_chunked(f)))