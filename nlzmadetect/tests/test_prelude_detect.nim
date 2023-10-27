import unittest
import nlzmadetect
import std/async

proc awaitScore(s : string) : (string, float64) =
    return read run_on_text_chunked(s)

suite "Verify the prelude file (i.e., training data) is correctly classified as an AI-generated file":
    setup:
        let (d, _) = awaitScore(readFile "../zippy/ai-generated.txt")

    test "Detect prelude":
        check(d == "AI")
