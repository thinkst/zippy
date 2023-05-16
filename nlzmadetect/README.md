# Nim package to classify test as LLM-generated

This is a nim version of the LZMA detector written in Python. 

## Instructions
Build with `nimble debug` or `nimble release` to generate a CLI program (either with more debugging information or a faster release build).

Run `./nlzmadetect` with a filename to check (or multiple)

Test against the samples repository with `nimble test`

To build the web version, invoke `nimble buildjs` and then serve this directory with a web server. Browse to `aidetect.html` to perform in-browser
text classification.
