"""Render demo.html to frames with headless Chromium, then (optionally) stitch with ffmpeg.
  python render.py preview 2 8 21        -> frames/preview_*.png at those seconds
  python render.py video [fps]           -> streams frames into ffmpeg, writes intentlock-demo.mp4
"""
import asyncio, base64, json, os, pathlib, subprocess, sys, tempfile, urllib.request
import websockets

HERE = pathlib.Path(__file__).resolve().parent
FRAMES = HERE / "frames"; FRAMES.mkdir(exist_ok=True)
PORT = 9352
TOTAL = json.loads((HERE / 'timeline.json').read_text())['total']
proc = subprocess.Popen(["chromium", "--headless=new", "--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage",
    "--allow-file-access-from-files", f"--remote-debugging-port={PORT}", f"--user-data-dir={tempfile.mkdtemp()}",
    "--hide-scrollbars", "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


async def main(mode, args):
    for _ in range(60):
        try:
            tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json", timeout=2)); break
        except Exception:
            await asyncio.sleep(0.5)
    url = next(t["webSocketDebuggerUrl"] for t in tabs if t["type"] == "page")
    async with websockets.connect(url, max_size=100_000_000) as ws:
        n = 0

        async def send(m, **p):
            nonlocal n; n += 1; i = n
            await ws.send(json.dumps({"id": i, "method": m, "params": p}))
            while True:
                r = json.loads(await ws.recv())
                if r.get("id") == i:
                    return r.get("result", r)

        async def ev(e):
            return (await send("Runtime.evaluate", expression=e, returnByValue=True, awaitPromise=True)).get("result", {}).get("value")
        await send("Page.enable"); await send("Runtime.enable")
        await send("Emulation.setDeviceMetricsOverride", width=1280, height=720, deviceScaleFactor=1.5, mobile=False)
        await send("Page.navigate", url=f"file://{HERE}/demo.html")
        for _ in range(80):
            if await ev("window.ready === true"):
                break
            await asyncio.sleep(0.25)
        else:
            raise SystemExit("page never became ready")
        if mode == "preview":
            for t in map(float, args):
                await ev(f"setT({t})"); await asyncio.sleep(0.15)
                s = await send("Page.captureScreenshot", format="png")
                (FRAMES / f"preview_{t:05.1f}.png").write_bytes(base64.b64decode(s["data"]))
                print("preview", t)
        else:
            fps = int(args[0]) if args else 24
            total = int(TOTAL * fps)
            audio = HERE / "narration.wav"
            # frames go straight into the encoder (no files on disk): near-lossless JPEG in, quality-first H.264 out
            cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-f", "image2pipe", "-c:v", "mjpeg", "-i", "-"]
            if audio.exists():
                cmd += ["-i", str(audio)]
            cmd += ["-vf", "scale=in_range=pc:out_range=tv,format=yuv420p", "-c:v", "libx264", "-crf", "16", "-preset", "veryfast",
                    "-tune", "animation", "-color_range", "tv", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"]
            if audio.exists():
                cmd += ["-c:a", "aac", "-b:a", "192k", "-af", "volume=-3dB", "-shortest"]
            cmd += ["-movflags", "+faststart", str(HERE / "intentlock-demo.mp4")]
            enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            for k in range(total):
                await ev(f"setT({k / fps})")
                shot = await send("Page.captureScreenshot", format="jpeg", quality=96)
                enc.stdin.write(base64.b64decode(shot["data"]))
                if k % 100 == 0:
                    print(f"frame {k}/{total}", flush=True)
            enc.stdin.close()
            if enc.wait() != 0:
                raise SystemExit("ffmpeg failed")
            print("video done")

try:
    asyncio.run(main(sys.argv[1], sys.argv[2:]))
finally:
    proc.terminate()
