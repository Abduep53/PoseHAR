import os, json, numpy as np, torch, argparse, yaml
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from src.models.temporal_tcn import TCN_Tiny
from src.data.datasets import split_by_group, load_items

def get_classes_from_yaml():
    """Get class names from classes.yaml"""
    try:
        cfg = yaml.safe_load(open("classes.yaml"))
        return [c["name"] for c in cfg["classes"]]
    except:
        return ['normal','run','fall']  # fallback

def load_clip(npy_path):
    x = np.load(npy_path).astype(np.float32)
    ref = x[:, :1, :]
    x = x - ref
    scale = np.maximum(1e-3, np.linalg.norm(x[:,11,:]-x[:,12,:], axis=-1, keepdims=True)).mean()
    x = x/scale
    x = torch.from_numpy(x).permute(2,1,0).unsqueeze(0)
    return x

def plot_confusion_matrix(cm, labels, title, save_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_split(model, items, labels, split_name, out_dir):
    """Evaluate model on a specific split"""
    print(f"[EVAL] Evaluating {split_name} split ({len(items)} samples)...")
    
    y_true, y_pred = [], []
    root = Path("data/mini")
    
    for it in items:
        x = load_clip(root / it['path'])
        with torch.no_grad():
            logits = model(x)
            p = int(torch.argmax(logits, dim=1))
        y_true.append(labels.index(it['label']))
        y_pred.append(p)
    
    # Calculate metrics
    rep = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate macro metrics
    macro_precision = rep['macro avg']['precision']
    macro_recall = rep['macro avg']['recall']
    macro_f1 = rep['macro avg']['f1-score']
    accuracy = rep['accuracy']
    
    print(f"[{split_name.upper()}] Accuracy: {accuracy:.4f}, Macro F1: {macro_f1:.4f}")
    
    # Save metrics
    metrics = {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class": {label: rep[label] for label in labels},
        "confusion_matrix": cm.tolist()
    }
    
    json.dump(metrics, open(out_dir / f"metrics_{split_name}.json", "w"), indent=2)
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, labels, f"Confusion Matrix - {split_name.title()}", 
                         out_dir / f"confusion_{split_name}.png")
    
    return metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--index', default='data/mini/index.json')
    ap.add_argument('--out_dir', default='runs/n30')
    ap.add_argument('--classes_yaml', default='classes.yaml')
    args = ap.parse_args()

    # Get class names
    labels = get_classes_from_yaml()
    num_classes = len(labels)
    print(f"[INFO] Evaluating {num_classes} classes: {labels}")

    # Load model
    model = TCN_Tiny(joints=33, classes=num_classes)
    model.load_state_dict(torch.load(args.weights, map_location='cpu'))
    model.eval()

    # Load data and create splits
    items = load_items("data/mini")
    train_items, val_items, test_items = split_by_group(items, seed=42, ratios=(0.7,0.15,0.15))
    
    print(f"[INFO] Data splits - Train: {len(train_items)}, Val: {len(val_items)}, Test: {len(test_items)}")

    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate on validation and test sets
    val_metrics = evaluate_split(model, val_items, labels, "val", out_dir)
    test_metrics = evaluate_split(model, test_items, labels, "test", out_dir)

    # Print summary
    print("\n=== EVALUATION SUMMARY ===")
    print(f"Validation - Accuracy: {val_metrics['accuracy']:.4f}, Macro F1: {val_metrics['macro_f1']:.4f}")
    print(f"Test - Accuracy: {test_metrics['accuracy']:.4f}, Macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"Results saved to: {out_dir}")
    print(f"Confusion matrices: {out_dir}/confusion_val.png, {out_dir}/confusion_test.png")

if __name__ == "__main__":
    main()
