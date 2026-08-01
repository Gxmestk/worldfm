#!/usr/bin/env python3
"""WorldFM interactive live-loop webserver (aiohttp + WebSocket).

Wraps the per-frame WorldFM API from ``run_pipeline`` to serve arbitrary
camera poses in real time (~3.9 FPS at step-1) over HTTP/WebSocket, reusing
the cached offline anchors in ``outputs/<name>/intermediates``. The shipped
demo is a batch renderer (fixed trajectory -> MP4); this turns it into an
interactive fly-through.

Design (verified against source — see docs/live-loop.md):
  * The per-frame path is already pose-general: ``step3_render_one`` accepts
    any K(3x3) + c2w(4x4 OpenCV camera-to-world) and renders cond1 from the
    cached PLY; cond2 is *selected* by nearest-view match from the 42-view DB.
  * One dedicated inference thread is the ONLY caller of torch. The WorldFM
    path has no locks/streams and uses ``reduce-overhead`` CUDA graphs, which
    are unsafe under concurrent replay — so all GPU work is serialized here.
  * Streaming uses latest-pose-wins + frame broadcast => smooth fly-through
    with bounded RAM (no frame pile-up). ``GET /frame`` is a one-shot job that
    takes priority over the stream each worker iteration.
  * step=1 only (config override). No FLUX/MoGe loaded (reuse-only), so the
    process stays well inside the 16 GB RAM cgroup / 16 GB VRAM budget.

Run via ``scripts/serve.sh`` (sets LD_LIBRARY_PATH + uv + self-guard), e.g.::

    uv run python live_server.py --meta demo/meta.json --host 0.0.0.0 --port 8000

Endpoints:  GET /  GET /health  GET /scene  GET /frame  WS /stream  GET /metrics
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import hashlib
import json
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from aiohttp import WSMsgType, web
from omegaconf import OmegaConf

# Reuse the pipeline's verified step functions + helpers. run_pipeline's
# top-level imports of moge_pano/panogen are try/except-guarded, so importing
# it is safe without setup_external_repos() — and reuse-only needs neither
# MoGe nor HunyuanWorld, so we deliberately do NOT call setup_external_repos
# (avokes loading models we don't need; keeps us in the RAM budget).
import run_pipeline as rp

WORLDFM_ROOT = rp.WORLDFM_ROOT
VIEWER_PATH = WORLDFM_ROOT / "live" / "viewer.html"


class LiveServer:
    """Holds resident pipeline state and serializes all inference on one thread."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.cfg: OmegaConf | None = None

        # scene / resident state (populated by preload())
        self.meta: dict = {}
        self.name: str = ""
        self.K: np.ndarray | None = None
        self.c2w0: np.ndarray | None = None
        self.pp_result = None
        self.renderer = None
        self.cond_db = None
        self.rcfg = None
        self.S: int = 512
        self.svc = None
        self.wcfg = None

        # server / concurrency state
        self.loop = None                 # set in on_startup (the run_app loop)
        self.ready = False
        self._stop = threading.Event()
        self._wake = threading.Event()   # nudges the worker out of idle wait
        self._preload_done = threading.Event()
        self._preload_error: Exception | None = None
        self._pose_lock = threading.Lock()
        self._stream_pose: tuple | None = None          # (K, c2w, seed)
        self._stream_version = 0
        self._last_stream_version = -1
        self._oneshot: queue.Queue = queue.Queue()      # (K, c2w, seed, future)
        self._subs: set = set()                          # asyncio.Queue per WS client (loop-owned)
        self._frame_times: collections.deque = collections.deque(maxlen=30)
        self._infer_thread = threading.Thread(target=self._infer_loop, daemon=True)

    # ------------------------------------------------------------------ config
    def _build_cfg(self) -> OmegaConf:
        # default.yaml model_path is the interpolation weights/worldfm_${step}-step.pth,
        # so overriding step alone selects the 1-step checkpoint.
        override = OmegaConf.create({"worldfm": {"step": int(self.args.step)}})
        return OmegaConf.merge(rp.DEFAULT_CFG, override)

    # ----------------------------------------------------------------- preload
    def preload(self) -> None:
        cfg = self._build_cfg()
        self.cfg = cfg

        if cfg.pipeline.gpu_index >= 0 and torch.cuda.is_available():
            torch.cuda.set_device(int(cfg.pipeline.gpu_index))

        meta_path = Path(self.args.meta).resolve()
        self.meta = rp._load_meta(str(meta_path))
        self.name = self.meta["name"]
        self.K = np.asarray(self.meta["K"], dtype=np.float64)
        c2w_list = [np.asarray(c, dtype=np.float64) for c in self.meta["c2w"]]
        self.c2w0 = c2w_list[0]

        base_output = Path(str(cfg.pipeline.output_dir))
        if not base_output.is_absolute():
            base_output = (WORLDFM_ROOT / base_output).resolve()
        output_dir = base_output / self.name
        cache_dir = rp._intermediates_dir(output_dir)

        print(f"[live] scene={self.name} cache={cache_dir} step={int(cfg.worldfm.step)}", flush=True)
        print("[live] loading cached intermediates (ply + 42 conditions)...", flush=True)
        self.pp_result = rp._load_postprocess_result(cache_dir)

        print("[live] building point renderer + condition DB (once)...", flush=True)
        self.renderer, self.cond_db, self.rcfg, self.S = rp.step3_init(self.pp_result, cfg=cfg)

        print("[live] loading WorldFM service (checkpoint + VAE + torch.compile)...", flush=True)
        self.svc, self.wcfg = rp.step4_init(cfg=cfg)

        print("[live] VAE-encoding 42 cond2 candidates (once)...", flush=True)
        self.svc.set_cond2_candidates_from_arrays(self.pp_result.condition_images)

    # --------------------------------------------------------------- inference
    def _render(self, K: np.ndarray, c2w: np.ndarray, seed: int | None):
        """step3_render_one -> step4_infer_one. Runs ONLY on the worker thread."""
        if seed is not None:
            torch.manual_seed(int(seed) & 0x7FFFFFFF)
        t0 = time.perf_counter()
        render_u8, _cond_nearest, idx, hits, samples = rp.step3_render_one(
            self.renderer, self.cond_db, self.pp_result, K, c2w,
            rcfg=self.rcfg, render_size=self.S,
        )
        frame = rp.step4_infer_one(
            self.svc, render_u8, None, wcfg=self.wcfg, cond2_index=idx, profile=False,
        )
        ms = (time.perf_counter() - t0) * 1000.0
        return frame, int(idx), int(hits), int(samples), ms

    @staticmethod
    def _encode_jpeg(frame: np.ndarray, quality: int) -> bytes:
        # frame is (H,W,3) uint8 RGB; cv2 wants BGR.
        ok, buf = cv2.imencode(
            ".jpg", frame[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        )
        if not ok:
            raise RuntimeError("JPEG encode failed")
        return buf.tobytes()

    def _pose_seed(self, K: np.ndarray, c2w: np.ndarray) -> int:
        h = hashlib.blake2b(digest_size=4)
        h.update(np.round(c2w, 4).tobytes())
        h.update(np.round(K, 4).tobytes())
        return int.from_bytes(h.digest(), "big")

    def _fps(self) -> float:
        if len(self._frame_times) < 2:
            return 0.0
        dt = self._frame_times[-1] - self._frame_times[0]
        return (len(self._frame_times) - 1) / dt if dt > 0 else 0.0

    # --------------------------------------------------------------- worker IO
    def _fulfill(self, fut, result, loop) -> None:
        loop.call_soon_threadsafe(self._safe_set, fut, result)

    @staticmethod
    def _safe_set(fut, result) -> None:
        if fut.done():
            return
        if isinstance(result, BaseException):
            fut.set_exception(result)
        else:
            fut.set_result(result)

    def _dispatch_frame(self, jpeg: bytes, stats: dict) -> None:
        """Runs on the event loop: fan-out one frame to every WS subscriber."""
        for q in list(self._subs):
            try:
                q.put_nowait((jpeg, stats))
            except asyncio.QueueFull:  # noqa: F821 - bound backpressure, drop oldest-ish
                pass

    def _handle_oneshot(self, job) -> None:
        K, c2w, seed, fut = job
        try:
            frame, idx, _h, _s, ms = self._render(K, c2w, seed)
            jpeg = self._encode_jpeg(frame, self.args.jpeg_quality)
            self._fulfill(fut, (jpeg, {"idx": idx, "ms": round(ms, 1)}), self.loop)
        except BaseException as exc:  # fulfill with exception so /frame 500s cleanly
            self._fulfill(fut, exc, self.loop)

    def _handle_stream(self, K: np.ndarray, c2w: np.ndarray, seed: int | None) -> None:
        try:
            frame, idx, _h, _s, ms = self._render(K, c2w, seed)
            jpeg = self._encode_jpeg(frame, self.args.jpeg_quality)
            self._frame_times.append(time.perf_counter())
            stats = {"idx": idx, "ms": round(ms, 1), "fps": round(self._fps(), 1)}
            self.loop.call_soon_threadsafe(self._dispatch_frame, jpeg, stats)
        except Exception as exc:  # never let one bad pose kill the worker
            print(f"[live] stream render error: {exc!r}", flush=True)

    def _infer_loop(self) -> None:
        """Worker thread: ALL torch runs here (preload + every frame).

        torch.compile(reduce-overhead) graphs are thread-affine — the setup
        (step4_init) and every call, including the first lazy capture, must
        happen on the SAME thread, or the compiled VAE/model forward asserts.
        So preload() runs here, not on the main/event-loop thread; every
        /frame and /stream job is also dispatched here.
        """
        try:
            self.preload()
        except BaseException as exc:  # surface to main before serving
            self._preload_error = exc
            self._preload_done.set()
            return
        self._preload_done.set()

        while not self._stop.is_set():
            # 1) drain one-shot jobs (GET /frame) — priority
            drained = False
            while True:
                try:
                    job = self._oneshot.get_nowait()
                except queue.Empty:
                    break
                self._handle_oneshot(job)
                drained = True
            if drained:
                continue
            # 2) streaming: render the newest pose if it changed
            with self._pose_lock:
                pose = self._stream_pose
                ver = self._stream_version
            if pose is not None and ver != self._last_stream_version:
                self._last_stream_version = ver
                self._handle_stream(*pose)
                continue
            # 3) idle until a pose/job arrives (or periodic timeout safety net)
            self._wake.wait(timeout=0.5)
            self._wake.clear()

    def submit_oneshot(self, K, c2w, seed, fut) -> None:
        self._oneshot.put((np.asarray(K, dtype=np.float64),
                           np.asarray(c2w, dtype=np.float64), seed, fut))
        self._wake.set()

    def set_stream_pose(self, K: np.ndarray, c2w: np.ndarray) -> None:
        seed = self._pose_seed(K, c2w) if self.args.seed else None
        with self._pose_lock:
            self._stream_pose = (K, c2w, seed)
            self._stream_version += 1
        self._wake.set()

    # ----------------------------------------------------------- warmup / life
    async def warmup(self, app: web.Application) -> None:
        self.loop = asyncio.get_running_loop()  # noqa: F841 - set before any job fulfills
        n = max(1, int(self.args.warmup))
        print(f"[live] warming up ({n} frames, captures torch.compile graphs)...", flush=True)
        # tiny orbit variants around c2w0 so the graph sees representative input
        poses = [self.c2w0]
        for _ in range(max(0, n - 1)):
            jitter = np.eye(4, dtype=np.float64)
            jitter[:3, 3] = np.random.default_rng(0).uniform(-0.05, 0.05, size=3) * (len(poses))
            poses.append(self.c2w0 @ jitter)
        for p in poses[:n]:
            fut = self.loop.create_future()
            self.submit_oneshot(self.K, p, None, fut)
            try:
                await fut  # first call traces+captures the compiled graphs (~10s @ step1)
            except Exception as exc:
                print(f"[live] warmup frame failed: {exc!r}", flush=True)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        self.ready = True
        print(f"[live] READY — viewer at http://{self.args.host}:{self.args.port}/ "
              f"(WS /stream, GET /frame)", flush=True)

    async def cleanup(self, app: web.Application) -> None:
        self._stop.set()
        self._wake.set()
        if self._infer_thread.is_alive():
            self._infer_thread.join(timeout=5.0)
        for obj in (self.svc, self.renderer, self.cond_db):
            del obj
        self.svc = self.renderer = self.cond_db = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[live] shut down.", flush=True)

    # -------------------------------------------------------------- endpoints
    def _scene_info(self) -> dict:
        return {
            "name": self.name,
            "step": int(self.cfg.worldfm.step),
            "image_size": int(self.cfg.worldfm.image_size),
            "render_size": int(self.S),
            "K": self.K.tolist(),
            "c2w0": self.c2w0.tolist(),
        }

    async def index(self, request: web.Request) -> web.FileResponse:
        if not VIEWER_PATH.exists():
            raise web.HTTPNotFound(text="viewer.html missing — expected at live/viewer.html")
        return web.FileResponse(str(VIEWER_PATH))

    async def health(self, request: web.Request) -> web.Response:
        return web.json_response({"status": "ready" if self.ready else "loading",
                                  "step": int(self.cfg.worldfm.step), "name": self.name,
                                  "subscribers": len(self._subs)})

    async def scene(self, request: web.Request) -> web.Response:
        return web.json_response(self._scene_info())

    async def metrics(self, request: web.Request) -> web.Response:
        vram_alloc = vram_peak = 0.0
        if torch.cuda.is_available():
            vram_alloc = torch.cuda.memory_allocated() / (1024 ** 2)
            vram_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
        mem_avail = self._mem_available_mb()
        return web.json_response({
            "ready": self.ready,
            "vram_alloc_mb": round(vram_alloc, 1),
            "vram_peak_mb": round(vram_peak, 1),
            "mem_available_mb": mem_avail,
            "fps": round(self._fps(), 1),
            "subscribers": len(self._subs),
            "stream_version": self._stream_version,
        })

    @staticmethod
    def _mem_available_mb() -> int:
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
        except Exception:
            pass
        return -1

    @staticmethod
    def _parse_pose(c2w_raw, K_raw, default_c2w, default_K):
        c2w = np.asarray(json.loads(c2w_raw), dtype=np.float64) if c2w_raw is not None else default_c2w
        K = np.asarray(json.loads(K_raw), dtype=np.float64) if K_raw is not None else default_K
        LiveServer._validate_pose(K, c2w)
        return K, c2w

    @staticmethod
    def _validate_pose(K: np.ndarray, c2w: np.ndarray) -> None:
        if K.shape != (3, 3):
            raise ValueError(f"K must be (3,3), got {K.shape}")
        if c2w.shape != (4, 4):
            raise ValueError(f"c2w must be (4,4), got {c2w.shape}")
        if not (np.all(np.isfinite(K)) and np.all(np.isfinite(c2w))):
            raise ValueError("K/c2w must be finite")

    async def frame(self, request: web.Request) -> web.Response:
        """One-shot render: ?c2w=[[..]]&K=[[..]] (URL-encoded JSON). Defaults to c2w0/K."""
        try:
            K, c2w = self._parse_pose(
                request.query.get("c2w"), request.query.get("K"), self.c2w0, self.K)
        except (ValueError, json.JSONDecodeError) as exc:
            raise web.HTTPBadRequest(text=f"bad pose: {exc}")
        seed = self._pose_seed(K, c2w) if self.args.seed else None
        fut = self.loop.create_future()
        self.submit_oneshot(K, c2w, seed, fut)
        try:
            jpeg, _stats = await fut
        except Exception as exc:
            raise web.HTTPInternalServerError(text=f"render failed: {exc}")
        return web.Response(body=jpeg, content_type="image/jpeg",
                            headers={"Cache-Control": "no-store"})

    async def stream(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(max_msg_size=0, heartbeat=30.0)
        await ws.prepare(request)
        q: asyncio.Queue = asyncio.Queue(maxsize=4)
        self._subs.add(q)
        await ws.send_json({"type": "hello", "scene": self._scene_info(), "ready": self.ready})

        async def reader():
            try:
                async for msg in ws:
                    if msg.type == WSMsgType.TEXT:
                        try:
                            data = json.loads(msg.data)
                            K = np.asarray(data.get("K", self.K.tolist()), dtype=np.float64)
                            c2w = np.asarray(data["c2w"], dtype=np.float64)
                            self._validate_pose(K, c2w)
                        except (ValueError, KeyError, TypeError):
                            await ws.send_json({"type": "error", "msg": "need {c2w:4x4, K:3x3}"})
                            continue
                        self.set_stream_pose(K, c2w)
                    elif msg.type == WSMsgType.ERROR:
                        break
            except (ConnectionResetError, asyncio.CancelledError):
                pass

        async def writer():
            try:
                while True:
                    jpeg, stats = await q.get()
                    await ws.send_bytes(jpeg)
                    await ws.send_json({"type": "stats", "ready": self.ready, **stats})
            except (ConnectionResetError, asyncio.CancelledError):
                pass

        # Seed with the home pose so a frame streams immediately on connect.
        self.set_stream_pose(self.K, self.c2w0)
        rt = asyncio.create_task(reader())
        wt = asyncio.create_task(writer())
        try:
            await rt  # exits when the client disconnects
        finally:
            wt.cancel()
            self._subs.discard(q)
            try:
                await wt
            except (asyncio.CancelledError, Exception):
                pass
            await ws.close()
        return ws

    # ------------------------------------------------------------------- build
    def build_app(self) -> web.Application:
        app = web.Application(client_max_size=1 << 20)
        app.router.add_get("/", self.index)
        app.router.add_get("/health", self.health)
        app.router.add_get("/scene", self.scene)
        app.router.add_get("/metrics", self.metrics)
        app.router.add_get("/frame", self.frame)
        app.router.add_get("/stream", self.stream)
        app.on_startup.append(self.warmup)
        app.on_cleanup.append(self.cleanup)
        return app


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WorldFM interactive live-loop webserver")
    p.add_argument("--meta", default="demo/meta.json", help="meta.json (name/image/K/c2w)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--step", type=int, default=1, choices=[1, 2],
                   help="WorldFM DMD step (1=~3.9FPS fastest, 2=~2.6FPS quality)")
    p.add_argument("--warmup", type=int, default=3, help="torch.compile warmup frames at boot")
    p.add_argument("--seed", dest="seed", action="store_true", default=True,
                   help="Seed per pose for a stable preview (default; no flicker on hold)")
    p.add_argument("--no_seed", dest="seed", action="store_false",
                   help="Disable per-pose seeding (fresh noise each frame)")
    p.add_argument("--jpeg_quality", type=int, default=85)
    return p.parse_args()


def main() -> int:
    server = LiveServer(parse_args())
    server._infer_thread.start()
    server._preload_done.wait()           # preload runs ON the worker thread
    if server._preload_error is not None:
        print(f"[live] preload failed: {server._preload_error!r}", flush=True)
        raise server._preload_error
    app = server.build_app()
    web.run_app(app, host=server.args.host, port=server.args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
