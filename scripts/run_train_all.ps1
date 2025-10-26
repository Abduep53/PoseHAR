# scripts/run_train_all.ps1
$ErrorActionPreference = 'Stop'

# Check if virtual environment exists, if not create it
if (!(Test-Path .venv)) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
}

# Activate virtual environment
. .\.venv\Scripts\Activate.ps1

Write-Host "`n[0] Environment"
python -c "import torch,sys;print('python',sys.version);print('cuda',torch.cuda.is_available(), torch.version.cuda if hasattr(torch.version,'cuda') else None)"
$gpu = python -c "import torch; print('CUDA_OK' if torch.cuda.is_available() else 'CPU_ONLY')"
if ($gpu.Trim() -ne "CUDA_OK") { Write-Host "WARNING: CUDA is not available; training will run on CPU." -ForegroundColor Yellow }

# Install requirements if needed
Write-Host "`n[0a] Installing requirements..."
pip install -r requirements.txt

# ---------- 1) Pose extraction + dataset build ----------
Write-Host "`n[1] Build dataset from app/ (pose extraction + merge)"
python scripts\build_dataset_from_app.py | Tee-Object -FilePath logs_extract.txt

if (!(Test-Path data\mini\index.json)) { Write-Error "Dataset index not created (data\mini\index.json missing)"; exit 1 }

Write-Host "`n[1a] Sanity: window count & label distribution"
Write-Host "Windows:" (Get-Content data\mini\index.json).Length
python -c "import json,collections; idx=json.load(open(r'data/mini/index.json','r',encoding='utf-8')); cnt=collections.Counter([it['label'] for it in idx]); print('unique labels:', len(cnt)); [print(f'{k:18s} {v}') for k,v in cnt.most_common(40)]"

# ---------- 2) Train on GPU with safe fallback batch size ----------
Write-Host "`n[2] Train (try bs=64->48->32)"
$trained = $false
foreach ($bs in 64,48,32) {
  Write-Host "Trying batch_size=$bs"
  try {
    python -m src.train --data_root data\mini --epochs 25 --batch_size $bs --model tcn_tiny --save runs\n30 | Tee-Object -FilePath logs_train.txt
    if (Test-Path runs\n30\best.ckpt) { $trained = $true; break }
  } catch { Write-Host "Train failed with bs=$bs, trying next..." -ForegroundColor Yellow }
}
if (-not $trained) { Write-Error "Training did not produce runs\n30\best.ckpt"; exit 1 }

# ---------- 3) Evaluation ----------
Write-Host "`n[3] Evaluate (val & test) + confusion matrices"
python scripts\eval_metrics.py --weights runs\n30\best.ckpt --index data\mini\index.json --out_dir runs\n30 --classes_yaml classes.yaml

if (!(Test-Path runs\n30\metrics_test.json)) { Write-Error "metrics_test.json missing"; exit 1 }

Write-Host "`n[3a] Quick metrics"
python -c "import json,os; m=json.load(open(r'runs/n30/metrics_test.json','r',encoding='utf-8')); pcs=m.get('per_class_f1',{}); print('macroF1:', round(m.get('macro_f1',-1),4)); print('weakest 5:', sorted(pcs.items(), key=lambda x:x[1])[:5]); print('confusion:', 'runs/n30/confusion_val.png', 'runs/n30/confusion_test.png')"

# ---------- 4) Export ONNX + latency ----------
Write-Host "`n[4] Export ONNX + benchmarks"
python deploy\export_onnx.py --weights runs\n30\best.ckpt --out deploy\prism_tcn_n30.onnx
python scripts\benchmark_torch.py
python scripts\benchmark_onnx.py

# ---------- 5) Launch Gradio demo ----------
Write-Host "`n[5] Launching Gradio demo (use weights path: runs/n30/best.ckpt)"
python app\gradio_app.py
