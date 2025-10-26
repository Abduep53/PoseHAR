import json, os
from pathlib import Path
SRC = Path("data/mini/index.json")
SEEN_OUT = Path("data/seen/index.json"); UNSEEN_OUT = Path("data/unseen/index.json")
SEEN_OUT.parent.mkdir(parents=True, exist_ok=True); UNSEEN_OUT.parent.mkdir(parents=True, exist_ok=True)
items = json.load(open(SRC))
seen = [it for it in items if it["label"] in ("normal","run")]
unseen = [it for it in items if it["label"]=="fall"]
json.dump(seen, open(SEEN_OUT,"w"), indent=2)
json.dump(unseen, open(UNSEEN_OUT,"w"), indent=2)
print(f"[OK] seen={len(seen)} unseen={len(unseen)}")
