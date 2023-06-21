# ZipPy: Fast method to classify text as AI or human-generated

This is a research repo for fast AI detection using compression.
While there are a number of existing LLM detection systems, they all use a large model trained on either an LLM or
its training data to calculate the probability of each word given the preceeding, then calculating a score where
the more high-probability tokens are more likely to be AI-originated. Techniques and tools in this repo are looking for
faster approximation to be embeddable and more scalable.

## LZMA compression detector (`zippy.py` and `nlzmadetect`)

ZipPy uses the LZMA compression ratios as a way to indirectly measure the perplexity of a text.
Compression ratios have been used in the past to [detect anomalies in network data](http://owncloud.unsri.ac.id/journal/security/ontheuse_compression_Network_anomaly_detec.pdf)
for intrusion detection, so if perplexity is roughly a measure of anomalous tokens, it may be possible to use compression to detect low-perplexity text.
LZMA creates a dictionary of seen tokens, and then uses though in place of future tokens. The dictionary size, token length, etc.
are all dynamic (though influenced by the 'preset' of 0-9--with 0 being the fastest but worse compression than 9). The basic idea
is to 'seed' an LZMA compression stream with a corpus of AI-generated text (`ai-generated.txt`) and then measure the compression ratio of 
just the seed data with that of the sample appended. Samples that follow more closely in word choice, structure, etc. will acheive a higher 
compression ratio due to the prevalence of similar tokens in the dictionary, novel words, structures, etc. will appear anomalous to the seeded
dictionary, resulting in a worse compression ratio.

### Current evaluation

The leading LLM detection tools are [OpenAI's model detector (v2)](https://openai.com/blog/new-ai-classifier-for-indicating-ai-written-text), [GPTZero](https://gptzero.me/), and [Roberta](https://huggingface.co/roberta-base-openai-detector). Here are each of them compared with the LZMA detector across all the test datasets:

![ROC curve of detection tools](https://github.com/thinkst/zippy/blob/main/ai_detect_roc.png?raw=true)
