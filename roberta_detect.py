#!/usr/bin/env python3

# HuggingFace API test harness

import re
from typing import Optional, Tuple

from roberta_local import classify_text

def run_on_file_chunked(filename : str, chunk_size : int = 800, fuzziness : int = 3) -> Optional[Tuple[str, float]]:
	'''
	Given a filename (and an optional chunk size) returns the score for the contents of that file.
	This function chunks the file into at most chunk_size parts to score separately, then returns an average. This prevents a very large input
	overwhelming the model.
	'''
	with open(filename, 'r') as fp:
		contents = fp.read()
	return run_on_text_chunked(contents, chunk_size, fuzziness)

def run_on_text_chunked(contents : str, chunk_size : int = 800, fuzziness : int = 3) -> Optional[Tuple[str, float]]:
	'''
	Given a text (and an optional chunk size) returns the score for the contents of that string.
	This function chunks the string into at most chunk_size parts to score separately, then returns an average. This prevents a very large input
	overwhelming the model.
	'''

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
		if end == -1:
			end = contents.rfind('\n', start, start + chunk_size + 1)
		if end == -1:
			print("Unable to chunk naturally!")
			end = start + chunk_size + 1
		chunks.append(contents[start:end])
		start = end + 1
	chunks.append(contents[start:])
	scores = classify_text(chunks)
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

