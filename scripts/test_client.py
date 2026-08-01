import asyncio, io, json, time, glob
import numpy as np, cv2
from aiohttp import ClientSession, WSMsgType
from PIL import Image

BASE = "http://localhost:8123"
WS = "ws://localhost:8123/stream"
REPO = "/root/gtk-projects/world-model/worldfm"

async def main():
    async with ClientSession() as s:
        print("health:", await (await s.get(f"{BASE}/health")).json())
        sc = await (await s.get(f"{BASE}/scene")).json()
        K, c2w0, isz = sc["K"], sc["c2w0"], sc["image_size"]
        print(f"scene: name={sc['name']} step={sc['step']} image_size={isz} render_size={sc['render_size']}")

        # --- GET /frame at c2w0 ---
        q = f"c2w={json.dumps(c2w0)}&K={json.dumps(K)}"
        r = await s.get(f"{BASE}/frame?{q}")
        jpeg = await r.read()
        img = Image.open(io.BytesIO(jpeg))
        arr = np.asarray(img.convert("RGB"))
        img.save("outputs/mario/frame_c2w0.jpg")
        print(f"/frame: status={r.status} ctype={r.content_type} size={img.size} std={arr.std():.1f} mean={arr.mean():.1f}")
        assert r.status == 200 and img.size == (isz, isz) and arr.std() > 5, "frame looks invalid"

        # structural compare to the cached step-1 output for pose 0 (stochastic -> not pixel-equal)
        cached = sorted(glob.glob(f"{REPO}/outputs/mario/step1_output_*.png"))
        if cached:
            ref = np.asarray(Image.open(cached[0]).convert("RGB").resize((isz, isz)))
            mad = np.abs(arr.astype(int) - ref.astype(int)).mean()
            # color-histogram correlation as a scene-similarity proxy
            def hist(a):
                h = cv2.calcHist([a], [0,1,2], None, [8,8,8], [0,256]*3).flatten()
                return h / (h.sum()+1e-9)
            corr = cv2.compareHist(hist(arr), hist(ref), cv2.HISTCMP_CORREL)
            print(f"vs cached {cached[0].split('/')[-1]}: mean|diff|={mad:.1f}  histCorrel={corr:.3f}  "
                  f"(histCorrel>~0.7 => same scene/coloring)")
        else:
            print("no cached step1 output to compare")

        print("metrics:", json.dumps(await (await s.get(f"{BASE}/metrics")).json()))

        # --- WS /stream ---
        print("--- WS /stream (5s) ---")
        async with s.ws_connect(WS) as ws:
            hello = stats = None; nframe = 0; t0 = time.time()
            async def sender():
                for k in range(20):
                    await ws.send_json({"c2w": c2w0, "K": K})
                    await asyncio.sleep(0.25)
            snd = asyncio.create_task(sender())
            first = None
            while time.time() - t0 < 5.0:
                try:
                    msg = await asyncio.wait_for(ws.receive(), timeout=5.0)
                except asyncio.TimeoutError:
                    break
                if msg.type == WSMsgType.TEXT:
                    d = json.loads(msg.data)
                    if d.get("type") == "hello": hello = d
                    elif d.get("type") == "stats": stats = d
                elif msg.type == WSMsgType.BINARY:
                    nframe += 1
                    if first is None:
                        first = Image.open(io.BytesIO(msg.data)).size
            snd.cancel()
            dt = time.time() - t0
            print(f"hello.name={hello['scene']['name'] if hello else None}  first_frame_size={first}  "
                  f"frames={nframe}  ~{nframe/dt:.2f} FPS  last_stats={stats}")
        print("=== TEST DONE ===")

asyncio.run(main())
