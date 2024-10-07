# ZipPy: Fast method to classify text as AI or human-generated

This is a research repo for fast AI detection using compression.
While there are a number of existing LLM detection systems, they all use a large model trained on either an LLM or
its training data to calculate the probability of each word given the preceding, then calculate a score where
the more high-probability tokens are more likely to be AI-originated. Techniques and tools in this repo are looking for
faster approximation to be embeddable and more scalable.

### Additional resources

Below are some other places to learn about ZipPy:
* [Blog post about ZipPy](https://blog.thinkst.com/2023/06/meet-zippy-a-fast-ai-llm-text-detector.html)
* [Hack.LU talk video](https://www.youtube.com/watch?v=CIdVix6k5Jw)

## Compression-based detector (`zippy.py` and `nlzmadetect`)

ZipPy uses either the LZMA, Brotli, or zlib compression ratios as a way to indirectly measure the perplexity of a text.
Compression ratios have been used in the past to [detect anomalies in network data](https://ieeexplore.ieee.org/abstract/document/5199270)
for intrusion detection, so if perplexity is roughly a measure of anomalous tokens, it may be possible to use compression to detect low-perplexity text.
LZMA and zlib create a dictionary of seen tokens and then use though in place of future tokens. The dictionary size, token length, etc.
are all dynamic (though influenced by the 'preset' of 0-9--with 0 being the fastest but worse compression than 9). The basic idea
is to 'seed' a compression stream with a corpus of AI-generated text (`ai-generated.txt`) and then measure the compression ratio of 
just the seed data with that of the sample appended. Samples that follow more closely in word choice, structure, etc. will achieve a higher 
compression ratio due to the prevalence of similar tokens in the dictionary, novel words, structures, etc. will appear anomalous to the seeded
dictionary, resulting in a worse compression ratio.

### Current evaluation

Some of the leading LLM detection tools are: 
~~[OpenAI's model detector (v2)](https://openai.com/blog/new-ai-classifier-for-indicating-ai-written-text)~~, [Content at Scale](https://contentatscale.ai/ai-content-detector/), [GPTZero](https://gptzero.me/), [CrossPlag's AI detector](https://crossplag.com/ai-content-detector/), and [Roberta](https://huggingface.co/roberta-base-openai-detector). 
Here are each of them compared with both the LZMA and zlib detector across the test datasets:

![ROC curve of detection tools](https://github.com/thinkst/zippy/blob/main/ai_detect_roc.png?raw=true)

### Installation

You can install zippy one of two ways:

#### Using python/pip

Via pip:
```shell
pip3 install thinkst-zippy
```

Or from source:
```shell
python3 setup.py build && python3 setup.py sdist && pip3 install dist/*.tar.gz
```

Now you can `import zippy` in other scripts.

#### Using pkgx

```shell
pkgx install zippy # or run it directly `pkgx zippy -h`
```

### Usage

ZipPy will read files passed as command-line arguments or will read from stdin to allow for piping of text to it. 

Once you've [installed](#Installation) zippy it will add a new script (`zippy`) that you can use directly:

```shell
$ zippy -h
usage: zippy [-h] [-p P] [-e {zlib,lzma,brotli,ensemble}] [-s | sample_files ...]

positional arguments:
  sample_files          Text file(s) containing the sample to classify

options:
  -h, --help            show this help message and exit
  -p P                  Preset to use with compressor, higher values are slower but provide better compression
  -e {zlib,lzma,brotli,ensemble}
                        Which compression engine to use: lzma, zlib, brotli, or an ensemble of all engines
  -s                    Read from stdin until EOF is reached instead of from a file
$ zippy samples/human-generated/about_me.txt 
samples/human-generated/about_me.txt
('Human', 0.06013429262166636)
```

If you want to use the ZipPy technology in your browser, check out the [Chrome extension](https://chrome.google.com/webstore/detail/ai-noise-cancelling-headp/okghlbkbacncfnfcielbncabioedklcn) or the [Firefox extension](https://addons.mozilla.org/firefox/addon/ai-noise-cancelling-headphones/) that runs ZipPy in-browser to flag potentially AI-generated content.

### Interpreting the results

At its core, the output from ZipPy is purely a statistical comparison of the similarity between the LLM-generate corpus (or corpi)
and the provided sample to test. Samples that are closer (i.e., more tokens match the known-LLM corpus) will score with higher confidence
as AI-generated; samples that are less compressible to an LLM-trained compression dictionary are flagged as human-generated. There are
a few caveats to the output that are worth noting:

* The comparison is based on the similarity of the text, a different type of sample, e.g., in a different language, or with many fictional
names, will be less similar to the English-languge corpus. Either a new LLM-generated corpus is needed, or a different (larger) toolchain
that can handle multiple language types is needed. Using ZipPy as-built willl provide poor responses to non-English human language samples,
computer language samples, and English samples that are not clear prose (or poetry).

* The confidence score is a raw delta between the compression ratios for the prelude file (LLM-generated corpus), and the compression ratio with
the sample included. Higher values indicate more similarity for AI-classified inputs, and more dissimilarity for those classified as human, but
the scores are not a percentage or otherwise a point on a discrete range. A score of 0 means there is no indication either way, it is [possible in
testing](https://github.com/thinkst/zippy/blob/main/test_zippy_detect.py#L17) to ignore results that are "too close", in the browser extensions
these values are adjusted slightly before being used to calculate the transparency to err on the side of not hiding text.
