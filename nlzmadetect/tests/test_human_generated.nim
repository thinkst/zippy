import unittest
import nlzmadetect
import std/os
import strutils

suite "Test the LZMA detector against human-generated files":
    var files : seq[string] = @[]
    var passed = 0
    var failed = 0
    for f in walkDir("../samples/human-generated"):
        files.add(f.path)

    test "Classify human-generated samples":
        for f in files:
            let (d, s) = run_on_text_chunked(readFile f)
            if d == "Human":
                passed += 1
            else:
                echo f & " was classified as " & d & " with a score of: " & $s.formatFloat(ffDecimal, 8)
                failed += 1

    echo $passed & " / " & $(passed + failed) & " tests passed (" & $failed & " failed)"