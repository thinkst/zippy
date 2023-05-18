#/usr/bin/env python3

# Tool to calculate the burstiness of a text
# (C) 2023 Thinkst Applied Research, PTY
# Author: Jacob Torrey

import argparse, os
from numpy import std, var
from typing import List, Tuple

def calc_burstiness(s : str) -> Tuple[Tuple[float, float]]:
    '''
    Given a string returns the standard deviation and variance of sentence length in terms of both chars and words
    '''
    lens : List[Tuple[int, int]] = []
    sentences = s.split('.')
    for sentence in sentences:
        chars = len(sentence)
        if chars < 1:
            continue
        words = len(sentence.split(' '))
        lens.append((chars, words))
    cd = (std([x[0] for x in lens]), var([x[0] for x in lens]))
    wd = (std([x[1] for x in lens]), var([x[1] for x in lens]))
    return (cd, wd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("sample_files", nargs='+', help='Text file(s) containing the sample to analyze')
    args = parser.parse_args()
    ws = 0
    wv = 0
    for f in args.sample_files:
        print(f)
        if os.path.isfile(f):
            with open(f, 'r') as fp:
                text = fp.read()
            b = calc_burstiness(text)
            ws += b[1][0]
            wv += b[1][1]
            print(str(b))
    print("Average std: " + str(ws/len(args.sample_files)) + " var: " + str(wv/len(args.sample_files)))