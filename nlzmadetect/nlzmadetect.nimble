# Package

version       = "0.1.0"
author        = "Jacob Torrey"
description   = "Program to determine if text is LLM-generated or not using LZMA compression"
license       = "MIT"
srcDir        = "src"
bin           = @["nlzmadetect"]
installExt    = @["nim"]


# Dependencies

requires "nim >= 1.6.4, https://github.com/ranok/nim-lzma >= 0.1.2"
