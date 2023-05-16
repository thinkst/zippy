when defined(c):
  import std/[re, threadpool, encodings]
  import lzma
when defined(js):
  import std/[jsffi, jsre]
  import dom
import std/math
import strutils
when isMainModule and defined(c):
  import std/[parseopt, os]

const COMPRESSION_PRESET = 1.int32
const SHORT_SAMPLE_THRESHOLD = 350


const PRELUDE_FILE = "../../ai-generated.txt"
const PRELUDE_STR = staticRead(PRELUDE_FILE)
proc compress_str(s : string, preset = COMPRESSION_PRESET): float64
var PRELUDE_RATIO = compress_str("")

when defined(js):
  var console {.importc, nodecl.}: JsObject
  proc compress(str : cstring, mode : int) : seq[byte] {.importjs: "LZMA.compress(#, #)".}
  console.log("Initialized with a prelude compression ratio of: " & $PRELUDE_RATIO)

# Target independent wrapper for LZMA compression
proc ti_compress(input : cstring, preset: int32, check: int32): seq[byte] = 
  when defined(c):
    return compress(input, preset, check)
  when defined(js):
    return compress(input, preset)

proc compress_str(s : string, preset = COMPRESSION_PRESET): float64 =
  let
    in_len = PRELUDE_STR.len + s.len
  var combined : string = PRELUDE_STR & s
  when defined(c):
    combined = convert(PRELUDE_STR & s, "us-ascii", "UTF-8").replace(re"[^\x00-\x7F]")
  when defined(js):
    let nonascii = newRegExp(r"[^\x00-\x7F]")
    combined = $combined.cstring.replace(nonascii, "")
  let out_len = ti_compress(combined.cstring, preset, 0.int32).len
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

when defined(c):
  proc score_chunk(chunk : string, fuzziness : int): float64 =
    var (d, s) = score_string(chunk, fuzziness)
    if d == "AI":
      return -1.0 * s
    return s

proc run_on_text_chunked*(text : string, chunk_size : int = 1024, fuzziness : int = 3): (string, float64) =
  var inf : string = text
  when defined(c):
    inf = replace(inf, re" +", " ")
    inf = replace(inf, re"\t")
    inf = replace(inf, re"\n+", "\n")
    inf = replace(inf, re"\n ", "\n")
    inf = replace(inf, re" \n", "\n")
  when defined(js):
    inf = $inf.cstring.replace(newRegExp(r" +"), " ")
    inf = $inf.cstring.replace(newRegExp(r"\t"), "")
    inf = $inf.cstring.replace(newRegExp(r"\n+"), "\n")
    inf = $inf.cstring.replace(newRegExp(r"\n "), "\n")
    inf = $inf.cstring.replace(newRegExp(r" \n"), "\n")

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

  when defined(c):
    var flows : seq[FlowVar[float64]] = @[]
    for c in chunks:
      flows.add(spawn score_chunk(c, fuzziness))

    for f in flows:
      let score = ^f
      var d : string = "Human"
      if score < 0.0:
        d = "AI"
        scores.add((d, score * -1.0))
      else:
        scores.add((d, score))
  when defined(js):
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

when isMainModule and defined(c):
  proc display_help() =
    echo "Call with one or more files to classify"

when defined(c) and isMainModule:
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
      let (d, s) = run_on_text_chunked(readFile(fn))
      echo "(" & d & ", " & $s.formatFloat(ffDecimal, 8) & ")"

when defined(js) and isMainModule:
  document.getElementById("preset_value").value = ($COMPRESSION_PRESET).cstring

  proc do_detect() {.exportc.} =
    let
      text : string = $document.getElementById("text_input").value
    var (d, s) = run_on_text_chunked(text)
    var color = "rgba(255, 0, 0, " & $(s.round(3) * 10.0) & ")"
    if d == "Human":
      color = "rgba(0, 255, 0, " & $(s.round(3) * 10.0) & ")"
    document.getElementById("output_span").textContent = d.cstring & ", confidence score of: " & ($s.round(6)).cstring
    document.getElementById("output_span").style.backgroundColor = color.cstring
