# scripts/build_dataset_from_app.py
import os, json, random, sys, re, subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT  = Path(__file__).resolve().parents[1]
APP   = ROOT / "app"
RAW   = ROOT / "data" / "raw_overlap"
MINI  = ROOT / "data" / "mini"
CLIPS = MINI / "clips"

RAW.mkdir(parents=True, exist_ok=True)
CLIPS.mkdir(parents=True, exist_ok=True)

def sanitize(stem: str) -> str:
    s = stem.strip()
    if s.startswith('-'): s = "v"+s[1:]
    return re.sub(r"[^0-9A-Za-z_\-]+", "_", s) or "clip"

def make_one(cls: str, vf: Path, window=3.0, stride=0.5) -> bool:
    out_dir = RAW / cls / vf.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "src.data.make_dataset",
        "--video", str(vf.resolve()),
        "--out",   str(out_dir.resolve()),
        "--prefix", sanitize(vf.stem),
        "--window_sec", str(window),
        "--stride_sec", str(stride),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("[FAIL]", vf, "\n", (r.stderr or r.stdout)[:600])
        return False
    return True

def build(window=3.0, stride=0.5, workers=None):
    videos = []
    for clsdir in sorted(APP.iterdir()):
        if not clsdir.is_dir(): continue
        for vf in clsdir.glob("*.mp4"):
            videos.append((clsdir.name, vf))
    print(f"[INFO] classes={len({c for c,_ in videos})} videos={len(videos)}")

    ok=0
    with ThreadPoolExecutor(max_workers=workers or os.cpu_count()) as ex:
        futs=[ex.submit(make_one, c, v, window, stride) for c,v in videos]
        for f in as_completed(futs):
            ok += int(f.result())
    print(f"[INFO] extracted_ok={ok}, failed={len(videos)-ok}")

    # merge real *.npy into unified index
    index=[]; c=0
    for clsdir in sorted(RAW.iterdir()):
        if not clsdir.is_dir(): continue
        cls=clsdir.name
        for vdir in sorted(clsdir.iterdir()):
            if not vdir.is_dir(): continue
            for npy in sorted(vdir.glob("*.npy")):
                dst = CLIPS / f"clip_{c:08d}.npy"
                try:
                    dst.write_bytes(npy.read_bytes())
                except FileNotFoundError:
                    continue
                index.append({"path": str(dst.relative_to(MINI)).replace("\\","/"),
                              "label": cls, "video_id": vdir.name})
                c+=1

    random.seed(42); random.shuffle(index)
    MINI.mkdir(parents=True, exist_ok=True)
    (MINI/"index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"[OK] windows={len(index)} -> data/mini/index.json")

if __name__ == "__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--window", type=float, default=3.0)
    ap.add_argument("--stride", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=None)
    a=ap.parse_args()
    build(a.window, a.stride, a.workers)
