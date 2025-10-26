import json, numpy as np, matplotlib.pyplot as plt, argparse
from sklearn.metrics import roc_curve, auc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scores_json', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    
    data = json.load(open(args.scores_json))
    scores = np.array(data['scores'])
    labels = np.array(data['labels'])
    
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Open-Set ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(args.out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] ROC curve saved to {args.out} (AUC = {roc_auc:.3f})")

if __name__ == "__main__":
    main()