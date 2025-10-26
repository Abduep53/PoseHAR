import os, json, re
from pathlib import Path
import subprocess, sys

APP = Path("app")
OUT_ROOTS = {
    "normal": Path("data/normal_overlap"),
    "run":    Path("data/run_overlap"),
    "fall":   Path("data/fall_overlap"),
}

def label_of(name: str):
    if name.startswith("normal_"): return "normal"
    if name.startswith("run_"):    return "run"
    if name.startswith("fall_"):   return "fall"
    return None

def run(cmd):
    print(">>", " ".join(cmd)); subprocess.check_call(cmd)

def main():
    for p in OUT_ROOTS.values(): p.mkdir(parents=True, exist_ok=True)

    # 1) per-video extraction into its own subfolder, with unique prefix
    for vf in sorted(APP.glob("*.mp4")):
        lbl = label_of(vf.name)
        if not lbl: 
            print("[SKIP]", vf.name); 
            continue
        stem = vf.stem
        out_dir = OUT_ROOTS[lbl] / stem
        prefix = stem
        run([sys.executable, "-m", "src.data.make_dataset",
             "--video", str(vf), "--out", str(out_dir),
             "--prefix", prefix, "--window_sec", "2.0", "--stride_sec", "1.0"])

    # 2) merge to data/mini
    mini = Path("data/mini"); clips = mini / "clips"
    mini.mkdir(parents=True, exist_ok=True); clips.mkdir(parents=True, exist_ok=True)
    index = []; c=0
    for lbl, root in OUT_ROOTS.items():
        for sub in sorted(root.glob("*")):
            if not sub.is_dir(): continue
            idx = sub / "index.json"
            if not idx.exists(): continue
            items = json.load(open(idx))
            for it in items:
                # Extract just the filename from the path
                filename = Path(it["path"]).name
                src = sub / filename
                new_name = f"clip_{c:07d}.npy"
                dst = clips / new_name
                dst.parent.mkdir(parents=True, exist_ok=True)
                # copy bytes
                dst.write_bytes(src.read_bytes())
                index.append({"path": f"clips/{new_name}", "label": lbl})
                c += 1

    # 3) stratified shuffle
    import random
    random.seed(42)
    random.shuffle(index)
    json.dump(index, open(mini/"index.json","w"), indent=2)
    print(f"[OK] merged {len(index)} windows -> data/mini")

if __name__=="__main__":
    main()
