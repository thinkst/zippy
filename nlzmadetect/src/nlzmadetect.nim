when defined(c):
  import std/[re, encodings]
  import lzma
when defined(js):
  import std/[jsffi, jsre]
import std/[math, async]
import strutils
when isMainModule and defined(c):
  import std/[parseopt, os]

when defined(js) and not defined(extension):
  import dom
  var local_prelude : string = ""
when defined(extension):
  var local_prelude {.exportc.} : cstring = ""

type
  Engine = enum
    LZMAEngine, ZLibEngine

var COMPRESSION_PRESET {.exportc.} = 2.int32
const SHORT_SAMPLE_THRESHOLD = 350

const ACTIVE_ENGINE = Engine.LZMAEngine

const PRELUDE_FILE = "../../zippy/ai-generated.txt"
const PRELUDE_STR {.exportc.} = staticRead(PRELUDE_FILE)
proc compress_str(s : string, preset = COMPRESSION_PRESET): Future[float64] {.async.}
var PRELUDE_RATIO = 0.0

when defined(js):
  var console {.importc, nodecl.}: JsObject
  when not defined(extension):
    proc getLocalStorageItem(key : cstring) : JsObject {.importjs: "localStorage.getItem(#)".}
    proc setLocalStorageItem(key : cstring, val : cstring) {.importjs: "localStorage.setItem(#, #)".}
  proc lzma_compress(str : cstring, mode : int) : seq[byte] {.importjs: "LZMA.compress(#, #)".}
  proc zlib_compress(str : cstring) : Future[seq[byte]] {.importjs: "ZLIB_compress(#)", async.}

proc init() {.async.} =
  PRELUDE_RATIO = await compress_str("")
  when defined(js):
    console.log("Initialized " & $ACTIVE_ENGINE & " with a prelude compression ratio of: " & $PRELUDE_RATIO)

discard init()
# Target independent wrapper for LZMA compression
proc ti_compress(input : cstring, preset: int32, check: int32): Future[seq[byte]] {.async.} = 
  when defined(c):
    return compress(input, preset, check)
  when not defined(c):
    when ACTIVE_ENGINE == Engine.ZLibEngine:
      return await zlib_compress(input)
    when ACTIVE_ENGINE == Engine.LZMAEngine:
      return lzma_compress(input, preset)

proc compress_str(s : string, preset = COMPRESSION_PRESET): Future[float64] {.async.} =
  when defined(c):
    let in_len = PRELUDE_STR.len + s.len
    var combined : string = PRELUDE_STR & s
    combined = convert(PRELUDE_STR & s, "us-ascii", "UTF-8").replace(re"[^\x00-\x7F]")
  when defined(js):
    let in_len = PRELUDE_STR.len + local_prelude.len + s.len
    when defined(extension):
      var combined : string = PRELUDE_STR & $(local_prelude) & s
    when not defined(extension):
      var combined : string = PRELUDE_STR & local_prelude & s
    let nonascii = newRegExp(r"[^\x00-\x7F]")
    combined = $combined.cstring.replace(nonascii, "")
  let out_len = (await ti_compress(combined.cstring, preset, 0.int32)).len
  return out_len.toFloat / in_len.toFloat

proc score_string*(s : string, fuzziness : int): Future[(string, float64)] {.async.} =
  let 
    sample_ratio = await compress_str(s)
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
  proc score_chunk(chunk : string, fuzziness : int): Future[float64] {.async.} =
    var (d, s) = await score_string(chunk, fuzziness)
    if d == "AI":
      return -1.0 * s
    return s

proc run_on_text_chunked*(text : string, chunk_size : int = 1024, fuzziness : int = 3): Future[(string, float64)] {.async.} =
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

  for c in chunks:
    scores.add(await score_string(c, fuzziness))

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
      proc print_results(score : Future[(string, float64)]) =
        let (d, s) = score.read()
        echo "(" & d & ", " & $s.formatFloat(ffDecimal, 8) & ")"
      echo fn
      run_on_text_chunked(readFile(fn)).addCallback(print_results)

when defined(js) and isMainModule:
  when not defined(extension):
    if not document.getElementById("preset_value").isNil:
      document.getElementById("preset_value").value = ($COMPRESSION_PRESET).cstring
    if getLocalStorageItem("local_prelude") != jsNull:
      local_prelude = $(getLocalStorageItem("local_prelude").to(cstring))
      discard init()

    proc add_to_lp() {.exportc.} =
      let new_text = document.getElementById("text_input").value
      local_prelude = local_prelude & "\n" & ($new_text)
      if getLocalStorageItem("local_prelude") != jsNull:
        let existing = getLocalStorageItem("local_prelude").to(cstring)
        setLocalStorageItem("local_prelude", existing & "\n" & new_text)
      else:
        setLocalStorageItem("local_prelude", new_text)
      discard init()

    proc do_detect() {.exportc, async.} =
      let
        text : string = $document.getElementById("text_input").value
      var (d, s) = await run_on_text_chunked(text)
      var color = "rgba(255, 0, 0, " & $(s.round(3) * 10.0) & ")"
      if d == "Human":
        color = "rgba(0, 255, 0, " & $(s.round(3) * 10.0) & ")"
      document.getElementById("output_span").textContent = d.cstring & ", confidence score of: " & ($s.round(6)).cstring
      document.getElementById("output_span").style.backgroundColor = color.cstring
  
  when defined(extension):
    proc add_to_lp(text : cstring) {.exportc.} =
      local_prelude = local_prelude & "\n".cstring & text
      discard init()

    proc detect_string(s : cstring) : Future[float] {.exportc,async.} =
      # Returns the opacity for the element containing the passed string (higher for human-generated)
      var (d, s) = await run_on_text_chunked($s)
      if d == "Human":
        return 1.0
      var opacity = 1.0 - s.round(3) * 10
      if opacity < 0.0:
        opacity = 0.0
      return opacity
