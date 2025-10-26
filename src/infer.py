import argparse, torch, numpy as np
from collections import deque
from src.models.temporal_tcn import TCN_Tiny
from src.data.make_dataset import extract_poses_from_video
from src.utils_classes import load_class_names

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--weights',required=True)
    ap.add_argument('--video',required=True)
    ap.add_argument('--joints',type=int,default=33)
    ap.add_argument('--threshold',type=float,default=0.6)
    ap.add_argument('--alpha',type=float,default=0.6)
    ap.add_argument('--window_size',type=int,default=7)
    a=ap.parse_args()
    
    # Load class names dynamically
    class_names = load_class_names(a.weights)
    num_classes = len(class_names)
    print(f"[INFO] Loaded {num_classes} classes: {class_names}")
    
    # Build model with dynamic classes
    m=TCN_Tiny(joints=a.joints,classes=num_classes)
    ckpt = torch.load(a.weights,map_location='cpu')
    m.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
    m.eval()
    
    windows=extract_poses_from_video(a.video,3.0,0.5)  # Use new window/stride
    smoothed_probs = None
    label_history = deque(maxlen=a.window_size)
    
    for i,w in enumerate(windows):
        x=w.astype(np.float32); ref=x[:,:1,:]; x=x-ref
        scale=np.maximum(1e-3,np.linalg.norm(x[:,11,:]-x[:,12,:],axis=-1,keepdims=True)).mean()
        x=x/scale
        
        # Ensure consistent window size (90 frames for 3s window)
        T_target = 90
        if x.shape[0] < T_target:
            # Pad with last frame
            pad_frames = T_target - x.shape[0]
            x = np.concatenate([x, np.tile(x[-1:], (pad_frames, 1, 1))], axis=0)
        elif x.shape[0] > T_target:
            # Truncate to target length
            x = x[:T_target]
        
        x=torch.from_numpy(x).permute(2,1,0).unsqueeze(0)
        with torch.no_grad(): 
            logits = m(x)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            
            # EMA smoothing
            if smoothed_probs is None:
                smoothed_probs = probs
            else:
                smoothed_probs = a.alpha * smoothed_probs + (1 - a.alpha) * probs
            
            # Get prediction
            pred_idx = np.argmax(smoothed_probs)
            max_prob = smoothed_probs[pred_idx]
            
            # Apply threshold for Unknown
            if max_prob < a.threshold:
                pred_label = "Unknown"
            else:
                pred_label = class_names[pred_idx]
            
            label_history.append(pred_label)
            
            # Majority vote over last N windows
            if len(label_history) > 0:
                from collections import Counter
                vote_counts = Counter(label_history)
                # Exclude Unknown from majority vote
                known_votes = {k: v for k, v in vote_counts.items() if k != "Unknown"}
                if known_votes:
                    majority_label = max(known_votes, key=known_votes.get)
                else:
                    majority_label = "Unknown"
            else:
                majority_label = pred_label
            
            print(f"window {i}: {pred_label} (prob={max_prob:.3f}) | majority: {majority_label}")
if __name__=='__main__': main()
