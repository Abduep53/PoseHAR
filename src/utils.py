
import os, random, numpy as np, torch
def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
def ensure_dir(p): os.makedirs(p, exist_ok=True)
