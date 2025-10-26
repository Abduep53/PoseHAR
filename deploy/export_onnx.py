
import argparse, torch, yaml
from src.models.temporal_tcn import TCN_Tiny

def get_num_classes():
    """Get number of classes from classes.yaml"""
    try:
        cfg = yaml.safe_load(open("classes.yaml"))
        return len(cfg["classes"])
    except:
        return 3  # fallback

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--weights', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--joints', type=int, default=33); ap.add_argument('--frames', type=int, default=60)
    a=ap.parse_args()
    
    # Get number of classes dynamically
    num_classes = get_num_classes()
    print(f"[INFO] Exporting ONNX model with {num_classes} classes")
    
    m=TCN_Tiny(joints=a.joints,classes=num_classes); m.load_state_dict(torch.load(a.weights,map_location='cpu')); m.eval()
    dummy=torch.randn(1,2,a.joints,a.frames)
    torch.onnx.export(m,dummy,a.out,input_names=['poses'],output_names=['logits'],opset_version=18)
    print(f"[OK] Exported to {a.out}")

if __name__=='__main__': main()
