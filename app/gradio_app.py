import sys, os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import gradio as gr
import torch, numpy as np
from collections import deque, Counter

from src.models.temporal_tcn import TCN_Tiny
from src.data.make_dataset import extract_poses_from_video
from src.utils_classes import load_class_names

DEFAULT_WEIGHTS = "runs/n30_bal/best.ckpt"

state = {"model": None, "classes": ["normal","run","fall"], "smoothed_probs": None, "label_history": deque(maxlen=7)}

def load(weights_path):
    """Load weights and dynamically detect classes."""
    try:
        classes = load_class_names(weights_path)
        num_classes = len(classes)
        m = TCN_Tiny(joints=33, classes=num_classes)
        ckpt = torch.load(weights_path, map_location="cpu")
        m.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
        m.eval()
        state["model"] = m
        state["classes"] = classes
        state["smoothed_probs"] = None
        state["label_history"].clear()
        class_preview = ', '.join(classes[:10]) + ('...' if num_classes > 10 else '')
        return f"Loaded: {weights_path} | classes={num_classes}: {class_preview}", [], "—", gr.update(visible=True)
    except Exception as e:
        state["model"] = None
        return f"ERROR: {repr(e)}", [], "—", gr.update(visible=False)

def _prep_windows(vfile, window_sec=3.0, stride_sec=0.5, target_frames=90):
    wins = extract_poses_from_video(vfile, window_sec, stride_sec, target_frames=target_frames)
    out = []
    for w in wins:
        x = w.astype(np.float32)               # [T,33,2]
        ref = x[:, :1, :]
        x = x - ref
        scale = np.maximum(1e-3, np.linalg.norm(x[:,11,:]-x[:,12,:], axis=-1, keepdims=True)).mean()
        x = x / scale
        xt = torch.from_numpy(x).permute(2,1,0).unsqueeze(0)  # [1,2,33,90]
        out.append(xt)
    return out

def _energy(logits, T=1.0):
    # negative energy (чем больше по модулю, тем более уверенная известная)
    return float((-T * torch.logsumexp(logits / T, dim=1)).item())

def predict(vfile, unknown_thresh=0.6, temp=1.0, alpha=0.6):
    """
    Returns per-window predictions with EMA smoothing and majority vote.
    """
    if state["model"] is None:
        return [{"error": "Model not loaded"}], "—", "—"
    try:
        model = state["model"]
        classes = state.get("classes", ["normal","run","fall"])
        batches = _prep_windows(vfile, 3.0, 0.5, 90)
        results = []
        
        # Reset state for new video
        state["smoothed_probs"] = None
        state["label_history"].clear()
        
        with torch.no_grad():
            for i, xt in enumerate(batches):
                logits = model(xt)
                probs = torch.softmax(logits/temp, dim=1)[0].cpu().numpy()
                
                # EMA smoothing
                if state["smoothed_probs"] is None:
                    state["smoothed_probs"] = probs
                else:
                    state["smoothed_probs"] = alpha * state["smoothed_probs"] + (1 - alpha) * probs
                
                # Get prediction
                pred_idx = int(np.argmax(state["smoothed_probs"]))
                max_prob = state["smoothed_probs"][pred_idx]
                
                # Apply threshold for Unknown
                if max_prob < unknown_thresh:
                    pred_lbl = "Unknown"
                else:
                    pred_lbl = classes[pred_idx]
                
                state["label_history"].append(pred_lbl)
                
                # Majority vote over last 7 windows
                if len(state["label_history"]) > 0:
                    vote_counts = Counter(state["label_history"])
                    known_votes = {k: v for k, v in vote_counts.items() if k != "Unknown"}
                    if known_votes:
                        majority_label = max(known_votes, key=known_votes.get)
                    else:
                        majority_label = "Unknown"
                else:
                    majority_label = pred_lbl
                
                results.append({
                    **{lbl: float(state["smoothed_probs"][i]) for i,lbl in enumerate(classes)},
                    "max_prob": float(max_prob),
                    "pred": pred_lbl,
                    "majority": majority_label,
                })
        
        # Final majority vote
        if len(state["label_history"]) > 0:
            vote_counts = Counter(state["label_history"])
            known_votes = {k: v for k, v in vote_counts.items() if k != "Unknown"}
            final = "Unknown" if not known_votes else max(known_votes, key=known_votes.get)
        else:
            final = "Unknown"
            
        return results, final, f"Processed {len(batches)} windows"
    except Exception as e:
        return [{"error": repr(e)}], "—", "—"

with gr.Blocks() as demo:
    gr.Markdown("# PRISM — Robust 30-Class Pose-Only Recognition")
    with gr.Row():
        w = gr.Textbox(label="Weights", value=DEFAULT_WEIGHTS)
        b = gr.Button("Load", variant="primary")
    status = gr.Textbox(label="Status", interactive=False)

    v = gr.Video(label="Upload clip")

    with gr.Row():
        thr = gr.Slider(label="Unknown threshold", value=0.6, minimum=0.1, maximum=1.0, step=0.05)
        temp = gr.Slider(label="Temperature", value=1.0, minimum=0.5, maximum=5.0, step=0.1)
        alpha = gr.Slider(label="EMA smoothing (α)", value=0.6, minimum=0.1, maximum=0.9, step=0.1)

    run_btn = gr.Button("Predict", variant="primary")
    final = gr.Textbox(label="Final (majority vote)", interactive=False)
    info = gr.Textbox(label="Info", interactive=False, visible=False)
    out = gr.JSON(label="Per-window predictions (smoothed + majority)")

    # Load: статус + очистка результатов
    b.click(load, inputs=w, outputs=[status, out, final, info])
    # Predict
    run_btn.click(predict, inputs=[v, thr, temp, alpha], outputs=[out, final, info])

if __name__ == '__main__':
    demo.launch()
