# scripts/yt_bulk_dl.py
import os, sys, json, subprocess, shlex
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLYAML = ROOT / "classes.yaml"

try:
    import yaml
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyyaml", "yt-dlp"])
    import yaml

cfg = yaml.safe_load(CLYAML.read_text(encoding="utf-8"))
classes = cfg["classes"]
per_query = int(cfg.get("per_query_limit", 60))
min_dur = int(cfg.get("min_duration_sec", 6))
max_dur = int(cfg.get("max_duration_sec", 15))
res_min = int(cfg.get("resolution_min", 480))

cookies = os.environ.get("YT_COOKIES")  # set to "cookies.txt" by the PS1 script if exists

def run(cmd):
    print("[yt-dlp]", " ".join(cmd))
    return subprocess.call(cmd)

def main():
    for c in classes:
        cname = c["name"]
        qlist = c.get("queries", [])
        outdir = ROOT / "app" / cname
        outdir.mkdir(parents=True, exist_ok=True)

        for q in qlist:
            # ytsearchNN:query
            spec = f"ytsearch{per_query}:{q}"
            out_template = str(outdir / "%(id)s.%(ext)s")
            cmd = [
                "yt-dlp", spec,
                "-f", f"mp4[height>={res_min}]/mp4",
                "--match-filter", f"duration > {min_dur} & duration < {max_dur}",
                "--no-playlist", "--no-warnings", "--ignore-errors",
                "--retries", "infinite",
                "--fragment-retries", "infinite",
                "--concurrent-fragments", "4",
                "--socket-timeout", "15",
                "--geo-bypass",
                "--no-overwrites",
                "-o", out_template
            ]
            if cookies and Path(cookies).exists():
                cmd += ["--cookies", cookies]

            rc = run(cmd)
            print(f"[CLASS] {cname}  query='{q}'  rc={rc}")

if __name__ == "__main__":
    main()
