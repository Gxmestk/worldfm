"""Generate viewable previews of the WorldFM intermediates into outputs/mario/gallery.

Renders: input photo, FLUX panorama, depth (false-color), the 3D point cloud (cond1
snapshots from the demo's own camera poses, so scale is correct by construction),
a montage of the 42 cond2 reference views, and copies the step1/step2 videos. The
point cloud is rendered with the repo's own TorchPointCloudRenderer (no FLUX/MoGe
loaded) using meta.json poses — i.e. exactly what the model sees as cond1.
"""
import glob, json, math, os, shutil, sys
from pathlib import Path
import numpy as np, cv2, torch

REPO = Path("/root/gtk-projects/world-model/worldfm")
sys.path.insert(0, str(REPO))
from modules.point_renderer import TorchPointCloudRenderer  # noqa: E402

OUT = REPO / "outputs/mario"; INTER = OUT / "intermediates"; DEMO = REPO / "demo"
G = OUT / "gallery"; G.mkdir(parents=True, exist_ok=True)   # -> outputs/mario/gallery (in-repo)

# ---- depth (false-color + grayscale) from the depth that actually built the PLY ----
npz = np.load(INTER / "postprocess_arrays.npz")
depth = npz["depth"].astype(np.float32)
valid = depth > 0
lo, hi = float(depth[valid].min()), float(depth[valid].max())
n = np.zeros_like(depth)
n[valid] = np.clip((depth[valid] - lo) / (hi - lo + 1e-9), 0, 1)
turbo = cv2.applyColorMap((n * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
turbo[~valid] = 0
cv2.imwrite(str(G / "depth_turbo.png"), turbo)
cv2.imwrite(str(G / "depth_gray.png"), (n * 255).astype(np.uint8))
print("depth range", lo, hi, "shape", depth.shape)

# ---- point cloud: render cond1-style snapshots from the demo's own camera poses ----
xyz = npz["ply_xyz"].astype(np.float32)
rgb = npz["ply_rgb"].astype(np.float32) / 255.0
meta = json.load(open(DEMO / "meta.json"))
K = np.asarray(meta["K"], dtype=np.float64)
c2ws = [np.asarray(c, dtype=np.float64) for c in meta["c2w"]]
dev = "cuda" if torch.cuda.is_available() else "cpu"
renderer = TorchPointCloudRenderer(points_xyz=xyz, points_rgb=rgb, width=512, height=512,
                                   device=dev, mode="fast")
for i, idx in enumerate([0, 6, 12, 18, 24, 29]):
    out = renderer.render_torch(K_3x3=K, c2w_4x4=c2ws[idx], c2w_is_camera_to_world=True)
    img = out.rgb_u8.cpu().numpy()[:, :, ::-1].copy()   # RGB -> BGR for cv2
    cv2.putText(img, f"traj pose #{idx}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imwrite(str(G / f"ply_cond1_{i}.png"), img)
print("ply snapshots done; points:", xyz.shape[0])

# ---- 42 condition views montage (cond2 candidates) ----
files = sorted(glob.glob(str(INTER / "conditions" / "*.png")))
thumbs = [cv2.resize(cv2.imread(f), (180, 180)) for f in files]
cols = 7
rows = math.ceil(len(thumbs) / cols)
mont = np.zeros((rows * 180, cols * 180, 3), dtype=np.uint8)
for k, t in enumerate(thumbs):
    r, c = divmod(k, cols)
    mont[r * 180:(r + 1) * 180, c * 180:(c + 1) * 180] = t
cv2.imwrite(str(G / "conditions_montage.png"), mont)
print("conditions:", len(files))

# ---- copy the directly-viewable originals ----
for src, dst in [
    (DEMO / "mario.png", "input_mario.png"),
    (OUT / "panorama.png", "panorama.png"),
    (INTER / "panorama.png", "panorama_fullres.png"),
]:
    if src.exists():
        shutil.copy(src, G / dst)
for f in ["step1_web.mp4", "step2_web.mp4"]:
    if (OUT / f).exists():
        shutil.copy(OUT / f, G / f)
for tag in ["step1", "step2"]:
    src = OUT / f"{tag}_output_0000.png"
    if src.exists():
        shutil.copy(src, G / f"{tag}_frame0.png")

print("gallery files:", sorted(os.listdir(G)))
