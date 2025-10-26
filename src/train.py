import os, argparse, torch, torch.nn as nn, json, yaml, numpy as np
from torch.optim import Adam
from src.config import TrainConfig
from src.utils import set_seed, ensure_dir
from src.data.datasets import make_loaders
from src.models.temporal_tcn import TCN_Tiny
from sklearn.metrics import f1_score
def augment_pose(x, joint_dropout=0.0, jitter=0.0, tempo_warp=0.0):
    """Apply pose augmentations"""
    if joint_dropout > 0 and torch.rand(1) < joint_dropout:
        # Random joint dropout
        mask = torch.rand(x.shape[1], 1, 1) > joint_dropout
        x = x * mask.to(x.device)
    
    if jitter > 0:
        # Add Gaussian jitter
        noise = torch.randn_like(x) * jitter
        x = x + noise
    
    # Temporarily disable tempo warping due to shape issues
    # if tempo_warp > 0 and torch.rand(1) < 0.5:
    #     # Temporal warping - x is [B, C, T] where B=batch, C=2*33=66, T=90
    #     T = x.shape[-1]
    #     warp_factor = 1.0 + torch.randn(1) * tempo_warp
    #     warp_factor = torch.clamp(warp_factor, 0.8, 1.2)
    #     new_T = int(T * warp_factor)
    #     if new_T != T:
    #         # Reshape to 3D: [B*C, T] for interpolation
    #         B, C, T_orig = x.shape
    #         x_flat = x.view(B * C, T_orig)
    #         x_flat = torch.nn.functional.interpolate(x_flat.unsqueeze(1), size=new_T, mode='linear', align_corners=False)
    #         x_flat = x_flat.squeeze(1)
    #         x = x_flat.view(B, C, new_T)
    #         if new_T > T:
    #             x = x[:, :, :T]
    #         else:
    #             pad = torch.zeros(x.shape[0], x.shape[1], T - new_T, device=x.device)
    #             x = torch.cat([x, pad], dim=-1)
    
    return x

def train_epoch(m,ld,d,crit,opt,aug_joint_dropout=0.0,aug_jitter=0.0,aug_tempo=0.0):
    m.train(); tot=0; corr=0; loss=0
    for x,y in ld:
        x,y=x.to(d),y.to(d)
        # Apply augmentations
        x = augment_pose(x, aug_joint_dropout, aug_jitter, aug_tempo)
        logits=m(x); l=crit(logits,y)
        opt.zero_grad(); l.backward(); opt.step()
        loss+=l.item()*x.size(0); corr+=(logits.argmax(1)==y).sum().item(); tot+=x.size(0)
    return loss/max(tot,1), corr/max(tot,1)

def eval_epoch(m,ld,d,crit):
    m.eval(); tot=0; corr=0; loss=0; all_preds=[]; all_labels=[]
    with torch.no_grad():
        for x,y in ld:
            x,y=x.to(d),y.to(d); logits=m(x); l=crit(logits,y)
            loss+=l.item()*x.size(0); corr+=(logits.argmax(1)==y).sum().item(); tot+=x.size(0)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    f1 = f1_score(all_labels, all_preds, average='macro')
    return loss/max(tot,1), corr/max(tot,1), f1

def get_num_classes():
    """Get number of classes from classes.yaml"""
    try:
        cfg = yaml.safe_load(open("classes.yaml"))
        return len(cfg["classes"])
    except:
        return 3  # fallback

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data_root',default='data/mini'); ap.add_argument('--epochs',type=int,default=25)
    ap.add_argument('--batch_size',type=int,default=64); ap.add_argument('--lr',type=float,default=1e-3)
    ap.add_argument('--model',default='tcn_tiny'); ap.add_argument('--save',default='runs/n30')
    ap.add_argument('--aug_joint_dropout',type=float,default=0.07)
    ap.add_argument('--aug_jitter',type=float,default=2.0)
    ap.add_argument('--aug_tempo',type=float,default=0.12)
    ap.add_argument('--use_class_weights',type=int,default=1)
    ap.add_argument('--early_stop',type=int,default=1)
    a=ap.parse_args(); 
    
    cfg=TrainConfig(data_root=a.data_root,epochs=a.epochs,batch_size=a.batch_size,lr=a.lr,model=a.model,save_dir=a.save)
    set_seed(cfg.seed); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get train/val/test loaders with class frequencies
    tr,va,te,class_names,class_freqs=make_loaders(cfg.data_root,cfg.batch_size)
    print(f"[INFO] Classes ({len(class_names)}): {class_names}")
    print(f"[INFO] Class frequencies: {class_freqs}")
    num_classes = len(class_names)
    
    # Compute class weights for balanced training
    if a.use_class_weights:
        weights = []
        for cls in class_names:
            freq = class_freqs.get(cls, 1)
            weight = 1.0 / np.sqrt(freq + 1e-6)
            weights.append(weight)
        class_weights = torch.tensor(weights, dtype=torch.float32).to(dev)
        print(f"[INFO] Class weights: {class_weights}")
    else:
        class_weights = None
    
    m=TCN_Tiny(joints=33,classes=num_classes).to(dev)
    crit=nn.CrossEntropyLoss(weight=class_weights); opt=Adam(m.parameters(),lr=cfg.lr); ensure_dir(cfg.save_dir); best=0
    
    # Training metrics storage
    metrics = {"train": [], "val": [], "test": []}
    best_f1 = 0
    patience = 10
    patience_counter = 0
    
    for ep in range(cfg.epochs):
        tl,ta=train_epoch(m,tr,dev,crit,opt,a.aug_joint_dropout,a.aug_jitter,a.aug_tempo)
        vl,va_,vf1=eval_epoch(m,va,dev,crit)
        print(f"[EP {ep+1}/{cfg.epochs}] train_acc={ta:.4f} train_loss={tl:.4f} | val_acc={va_:.4f} val_loss={vl:.4f} val_f1={vf1:.4f}")
        
        # Store metrics
        metrics["train"].append({"epoch": ep+1, "acc": ta, "loss": tl})
        metrics["val"].append({"epoch": ep+1, "acc": va_, "loss": vl, "f1": vf1})
        
        # Early stopping on validation F1
        if a.early_stop:
            if vf1 > best_f1:
                best_f1 = vf1
                patience_counter = 0
                # Save checkpoint with metadata
                ckpt = {
                    "state_dict": m.state_dict(),
                    "meta": {
                        "class_names": class_names,
                        "num_classes": len(class_names),
                        "model": "tcn_tiny"
                    }
                }
                torch.save(ckpt, os.path.join(cfg.save_dir,'best.ckpt'))
                print(f"[BEST] New best validation F1: {vf1:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"[EARLY STOP] No improvement for {patience} epochs. Stopping at epoch {ep+1}")
                    break
        else:
            # Original accuracy-based saving
            if va_ > best:
                best = va_
                ckpt = {
                    "state_dict": m.state_dict(),
                    "meta": {
                        "class_names": class_names,
                        "num_classes": len(class_names),
                        "model": "tcn_tiny"
                    }
                }
                torch.save(ckpt, os.path.join(cfg.save_dir,'best.ckpt'))
                print(f"[BEST] New best validation accuracy: {va_:.4f}")
    
    # Final test evaluation
    print("[FINAL] Evaluating on test set...")
    test_loss, test_acc, test_f1 = eval_epoch(m, te, dev, crit)
    print(f"[TEST] test_acc={test_acc:.4f} test_loss={test_loss:.4f} test_f1={test_f1:.4f}")
    
    # Save metrics
    metrics["test"] = [{"epoch": "final", "acc": test_acc, "loss": test_loss, "f1": test_f1}]
    json.dump(metrics, open(os.path.join(cfg.save_dir, "metrics_val.json"), "w"), indent=2)
    json.dump({"test_acc": test_acc, "test_loss": test_loss, "test_f1": test_f1}, open(os.path.join(cfg.save_dir, "metrics_test.json"), "w"), indent=2)
    
    # Save labels.json for inference
    with open(os.path.join(cfg.save_dir, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)
    
    print(f"[DONE] Training completed. Best val F1: {best_f1:.4f}, Test F1: {test_f1:.4f}")

if __name__=='__main__': main()
