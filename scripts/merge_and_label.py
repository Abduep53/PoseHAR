import os, json, shutil, random
from pathlib import Path

SOURCES = [
    ("data/normal_overlap", "normal"),
    ("data/run_overlap",    "run"),
    ("data/fall_overlap",   "fall"),
]
DST = Path("data/mini"); CLIPS = DST / "clips"

def main():
    DST.mkdir(parents=True, exist_ok=True); CLIPS.mkdir(parents=True, exist_ok=True)
    index = []; c = 0
    for src, label in SOURCES:
        idxp = Path(src) / "index.json"
        if not idxp.exists(): 
            print("[WARN] missing", idxp); 
            continue
        items = json.load(open(idxp))
        for it in items:
            npy_src = Path(src) / it["path"]
            if not npy_src.exists(): continue
            new = f"clip_{c:07d}.npy"
            shutil.copyfile(npy_src, CLIPS / new)
            index.append({"path": f"clips/{new}", "label": label}); c += 1
    random.seed(42); random.shuffle(index)
    json.dump(index, open(DST/"index.json","w"), indent=2)
    print(f"[OK] merged {len(index)} windows into data/mini (normal/run/fall)")

if __name__ == "__main__":
    main()