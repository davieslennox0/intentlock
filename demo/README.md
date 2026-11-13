# Intentlock demo video

`intentlock-demo.mp4` (about 2 minutes 25, 1920x1080, with a voice-over) walks through the project:
the incident behind the threat model, one poisoned invoice given to the base agent and to the
hardened agent with the guard, a held-out attack that the guard blocks, the headline numbers, and
the caveats.

**It is a replay of recorded episodes from the final run; no live model.** Every tool output, agent
action, guard verdict and final message shown is quoted from the recorded traces of run
`20261112T1326-final-d033d35` (scenario `sd-00370` on the test split, base and hardened + guard;
scenario `sd-00461` on held-out formats, hardened + guard). Addresses are shortened for the screen
and long outputs are trimmed with an ellipsis. The figures are the ones in the main README;
interpretability figures come from run `20261112T2003-interp-c369bcb`. All tools are mocks: nothing
is signed, sent or executed.

## How it is built

- `narration.py` is the script. `tts.py` speaks each line with
  [Kokoro](https://github.com/thewh1teagle/kokoro-onnx) (voice `af_heart`), measures how long each
  line takes, and writes `timeline.js`, `timeline.json` and `narration.wav`.
- `demo.html` draws every frame as a function of time (`setT(t)`), driven by those timings, so
  captions and events follow the voice.
- `render.py` steps headless Chromium through the frames and streams them into ffmpeg.

## Rebuild it

```
cd demo
pip install kokoro-onnx websockets        # plus chromium and ffmpeg on PATH
# model files: kokoro-v1.0.int8.onnx and voices-v1.0.bin from the kokoro-onnx releases page
KOKORO_DIR=/path/to/models python tts.py
python render.py preview 38 91 108        # single frames at those seconds, in frames/
python render.py video 24                 # writes intentlock-demo.mp4
```

Fonts in `fonts/` are Geist, Geist Mono and Instrument Serif (SIL Open Font License).
