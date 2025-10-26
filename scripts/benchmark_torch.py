import time, torch, numpy as np, json, yaml
from src.models.temporal_tcn import TCN_Tiny

def get_num_classes():
    """Get number of classes from classes.yaml"""
    try:
        cfg = yaml.safe_load(open("classes.yaml"))
        return len(cfg["classes"])
    except:
        return 3  # fallback

def main():
    num_classes = get_num_classes()
    print(f"[INFO] Benchmarking Torch model with {num_classes} classes")
    
    m = TCN_Tiny(joints=33, classes=num_classes)
    m.load_state_dict(torch.load("runs/n30/best.ckpt", map_location="cpu"))
    m.eval()
    
    dummy = torch.randn(1,2,33,60)
    # warmup
    for _ in range(10): _=m(dummy)
    N=100; t0=time.time()
    for _ in range(N): _=m(dummy)
    dt=(time.time()-t0)/N
    print(f"[OK] Torch latency per window: {dt*1000:.2f} ms")
    json.dump({"torch_latency_ms":dt*1000}, open("runs/n30/latency_torch.json","w"), indent=2)

if __name__=="__main__": main()
