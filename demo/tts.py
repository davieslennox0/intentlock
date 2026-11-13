"""Speak every line of narration.py with Kokoro, measure each one, and build:
   timeline.js / timeline.json   scene and line timings that drive the video (visuals follow the voice)
   narration.wav                 the voice track, each line placed at its start time

Needs `pip install kokoro-onnx` and the model files from
https://github.com/thewh1teagle/kokoro-onnx/releases (model-files-v1.0):
  KOKORO_DIR=/path/with/kokoro-v1.0.int8.onnx+voices-v1.0.bin python tts.py
"""
import json, os, pathlib, wave
import numpy as np
from kokoro_onnx import Kokoro

from narration import GAP, LEAD, SCENES, TAIL

HERE = pathlib.Path(__file__).resolve().parent
MODELS = pathlib.Path(os.environ.get("KOKORO_DIR", "tts-models"))
MODEL = os.environ.get("KOKORO_MODEL", "kokoro-v1.0.int8.onnx")
VOICE = os.environ.get("KOKORO_VOICE", "af_heart")
SPEED = float(os.environ.get("KOKORO_SPEED", "1.08"))

kokoro = Kokoro(str(MODELS / MODEL), str(MODELS / "voices-v1.0.bin"))


def speak(text):
    audio, sr = kokoro.create(text, voice=VOICE, speed=SPEED, lang="en-us")
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    return pcm, sr, len(pcm) / sr


t, timeline, clips, SR = 0.0, [], [], None
for sc in SCENES:
    a, cur, lines = t, t + LEAD, []
    for ln in sc["lines"]:
        wav, SR, dur = speak(ln["say"])
        lines.append({"a": round(cur, 3), "b": round(cur + dur, 3), "dur": round(dur, 3), "show": ln["show"], "ev": ln.get("ev")})
        clips.append((cur, wav))
        cur += dur + GAP
    b = max(cur - GAP + TAIL, a + sc["min"])
    timeline.append({"id": sc["id"], "a": round(a, 3), "b": round(b, 3), "lines": lines})
    t = b

total = round(t, 3)
track = np.zeros(int(total * SR) + SR, dtype=np.int16)
for start, wav in clips:
    i = int(start * SR)
    track[i:i + len(wav)] = wav
with wave.open(str(HERE / "narration.wav"), "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(track.tobytes())

(HERE / "timeline.json").write_text(json.dumps({"total": total, "scenes": timeline}, indent=1))
(HERE / "timeline.js").write_text("window.TL = " + json.dumps({"total": total, "scenes": timeline}) + ";\n")
print(f"voice {VOICE} speed {SPEED}  total {total}s")
for sc in timeline:
    print(sc["id"], f"{sc['a']:6.1f} -> {sc['b']:6.1f}", " | ".join(f"{l['dur']:.1f}s" for l in sc["lines"]))
