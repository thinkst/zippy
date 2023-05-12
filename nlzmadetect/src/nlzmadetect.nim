import std/[re, math]
import lzma
import encodings
import strutils
when isMainModule:
  import std/[parseopt, os]

const PRELUDE_FILE = "../ai-generated.txt"
const COMPRESSION_PRESET = 2.int32
const SHORT_SAMPLE_THRESHOLD = 350
var PRELUDE_STR = readFile(PRELUDE_FILE).convert("us-ascii", "UTF-8").replace(re"[^\x00-\x7F]")

proc compress_str(s : string, preset = COMPRESSION_PRESET): float64

var PRELUDE_RATIO = compress_str("")

proc compress_str(s : string, preset = COMPRESSION_PRESET): float64 =
  let
    in_len = PRELUDE_STR.len + s.len
    combined = convert(PRELUDE_STR & s, "us-ascii", "UTF-8").replace(re"[^\x00-\x7F]")
    out_len = compress(combined.cstring, preset, 0.int32).len
  return out_len.toFloat / in_len.toFloat

proc score_string*(s : string, fuzziness : int): (string, float64) =
  let 
    sample_ratio = compress_str(s)
    delta = PRELUDE_RATIO - sample_ratio
  var determination = "AI"
  if delta < 0:
    determination = "Human"
  
  if 0.0 == round(delta, fuzziness) and s.len >= SHORT_SAMPLE_THRESHOLD:
    determination = "AI"
  if 0.0 == round(delta, fuzziness) and s.len < SHORT_SAMPLE_THRESHOLD:
    determination = "Human"

  return (determination, abs(delta) * 100.0)

proc run_on_file_chunked*(filename : string, chunk_size : int = 1024, fuzziness : int = 3): (string, float64) =
  var inf = readFile(filename)

  inf = replace(inf, re" +", " ")
  inf = replace(inf, re"\t")
  inf = replace(inf, re"\n+", "\n")
  inf = replace(inf, re"\n ", "\n")
  inf = replace(inf, re" \n", "\n")

  var
    start = 0
    send = 0
    chunks : seq[string] = @[]
  while start + chunk_size < inf.len and send != -1:
      send = inf.rfind(' ', start, start + chunk_size)
      chunks.add(inf[start..send])
      start = send + 1
  chunks.add(inf[start..inf.len-1])

  var scores : seq[(string, float64)] = @[]
  for c in chunks:
    scores.add(score_string(c, fuzziness))
  var ssum : float64 = 0.0
  for s in scores:
    if s[0] == "AI":
        ssum -= s[1]
    else:
        ssum += s[1]
  var sa : float64 = ssum / len(scores).toFloat
  if sa < 0:
      return ("AI", abs(sa))
  else:
      return ("Human", abs(sa))

when isMainModule:
  proc display_help() =
    echo "Call with one or more files to classify"

when isMainModule:
  var 
    filenames : seq[string] = @[]
    parser = initOptParser()
  while true:
    parser.next()
    case parser.kind
    of cmdEnd: break
    of cmdShortOption, cmdLongOption:
      if parser.key == "help" or parser.key == "h":
        display_help()
        quit 0
    of cmdArgument:
      filenames.add(parser.key)
  if filenames.len == 0:
    display_help()
    quit 0
  for fn in filenames:
    if fileExists(fn):
      echo fn
      let (d, s) = run_on_file_chunked(fn)
      echo "(" & d & ", " & $s.formatFloat(ffDecimal, 8) & ")"
