# scripts/calibrate_threshold.py
import json, argparse, numpy as np
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--index", default="data/mini/index.json")
    ap.add_argument("--target_unknown_rate", type=float, default=0.03)
    ap.add_argument("--out_dir", default=None)
    a=ap.parse_args()

    w=Path(a.weights); out=Path(a.out_dir) if a.out_dir else w.parent
    out.mkdir(parents=True, exist_ok=True)

    # fallback: если нет реальных maxprob, ставим безопасные значения
    calib={"temperature":1.2,"threshold":0.6,"metric":"maxprob","note":"fallback"}
    mval=out/"metrics_val.json"
    if mval.exists():
        try:
            mv=json.loads(mval.read_text(encoding="utf-8"))
            arr=np.array(mv.get("val_maxprob",[]),dtype=float)
            if arr.size>10:
                tau=float(np.quantile(arr, a.target_unknown_rate))
                calib={"temperature":1.2,"threshold":max(0.4,min(0.9,tau)),"metric":"maxprob"}
        except: pass

    (out/"calibration.json").write_text(json.dumps(calib,indent=2),encoding="utf-8")
    print("[OK] calibration ->", out/"calibration.json", calib)

if __name__=="__main__":
    main()
