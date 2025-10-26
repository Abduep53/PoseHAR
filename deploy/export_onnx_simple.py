import argparse, torch, sys
from src.models.temporal_tcn import TCN_Tiny

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--joints', type=int, default=33)
    ap.add_argument('--frames', type=int, default=60)
    a=ap.parse_args()
    
    m=TCN_Tiny(joints=a.joints,classes=4)
    m.load_state_dict(torch.load(a.weights,map_location='cpu'))
    m.eval()
    
    dummy=torch.randn(1,2,a.joints,a.frames)
    
    # Suppress verbose output
    import warnings
    warnings.filterwarnings('ignore')
    
    try:
        torch.onnx.export(m,dummy,a.out,input_names=['poses'],output_names=['logits'],opset_version=18,verbose=False)
        print(f"Exported to {a.out}")
    except Exception as e:
        print(f"Export failed: {e}")
        sys.exit(1)

if __name__=='__main__': 
    main()
