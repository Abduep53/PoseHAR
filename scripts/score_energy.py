import os, json, numpy as np, torch
from pathlib import Path
from src.models.temporal_tcn import TCN_Tiny

def load_clip(npy_path):
    import numpy as np, torch
    x = np.load(npy_path).astype(np.float32)
    ref = x[:, :1, :]; x = x - ref
    scale = np.maximum(1e-3, np.linalg.norm(x[:,11,:]-x[:,12,:], axis=-1, keepdims=True)).mean()
    x = x/scale
    return torch.from_numpy(x).permute(2,1,0).unsqueeze(0)

def energy(logits, T=1.0):
    import torch
    return float((-T * torch.logsumexp(logits / T, dim=1)).item())

def collect(index_json, label_binary, model):
    root = Path("data/mini")  # Use the original mini dataset as source
    items = json.load(open(index_json))
    scores, labels = [], []
    for it in items:
        npy = root / it["path"]
        x = load_clip(npy); 
        with torch.no_grad(): logits = model(x)
        scores.append(energy(logits)); labels.append(label_binary)
    return scores, labels

def main():
    model = TCN_Tiny(joints=33, classes=3)
    model.load_state_dict(torch.load("runs/seen_only/best.ckpt", map_location="cpu")); model.eval()
    s1,l1 = collect("data/seen/index.json",   0, model)
    s2,l2 = collect("data/unseen/index.json", 1, model)
    d = {"scores": s1+s2, "labels": l1+l2}
    os.makedirs("runs/open_set", exist_ok=True)
    json.dump(d, open("runs/open_set/energy_scores.json","w"), indent=2)
    print("[OK] saved runs/open_set/energy_scores.json", "known=",len(s1),"unknown=",len(s2))
if __name__=="__main__": main()
