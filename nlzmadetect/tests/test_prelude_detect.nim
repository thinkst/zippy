import unittest
import nlzmadetect

suite "Verify the prelude file (i.e., training data) is correctly classified as an AI-generated file":
    setup:
        let (d, _) = run_on_file_chunked("../ai-generated.txt")

    test "Detect prelude":
        check(d == "AI")