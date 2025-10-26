. .\.venv\Scripts\Activate.ps1
param([string]$w="runs\n30\best.ckpt")
python - $w << 'PY'
import sys, json, torch
from src.utils_classes import load_class_names
w = sys.argv[1] if len(sys.argv)>1 else r"runs\n30\best.ckpt"
names = load_class_names(w)
print("N classes:", len(names))
print("First 15:", names[:15])
ckpt = torch.load(w, map_location="cpu")
sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
# Try to infer out dim from last layer weights if present
shape = None
for k,v in sd.items():
    if v.ndim==2 and v.shape[0]==len(names): shape=v.shape
print("Head shape:", shape)
PY

