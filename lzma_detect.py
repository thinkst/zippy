#!/usr/bin/env python3

# Code to attempt to detect AT-generated text [relatively] quickly via compression ratios
# (C) 2023 Thinkst Applied Research, PTY
# Author: Jacob Torrey <jacob@thinkst.com>

import lzma
from typing import List, Optional, Tuple

PRELUDE_FILE : str = 'ai-generated.txt'

class LzmaLlmDetector:
    '''Class providing functionality to attempt to detect LLM/generative AI generated text using the LZMA compression algorithm'''
    def __init__(self, prelude_file : Optional[str] = None, fuzziness_digits : int = 3) -> None:
        '''Initializes a compression with the passed prelude file, and optionally the number of digits to round to compare prelude vs. sample compression'''
        self.comp = lzma.LZMACompressor()
        self.c_buf : List[bytes] = []
        self.in_bytes : int = 0
        self.prelude_ratio : float = 0.0
        self.FUZZINESS_THRESHOLD = fuzziness_digits

        if prelude_file != None:
            # Read it once to get the default compression ratio for the prelude
            with open(prelude_file, 'r') as fp:
                self._compress_str(fp.read())
            self.prelude_ratio = self._finalize()
            # Redo this to prime the compressor
            self.comp = lzma.LZMACompressor()
            with open(prelude_file, 'r') as fp:
                self._compress_str(fp.read())
    
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
        print(str((prelude_score, sample_score)))
        delta = prelude_score - sample_score
        determination = 'AI'
        if delta < 0 or round(delta, self.FUZZINESS_THRESHOLD) == 0:
            determination = 'Human'
        return (determination, abs(delta * 100))
        
