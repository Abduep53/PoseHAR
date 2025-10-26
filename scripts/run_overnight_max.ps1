# scripts/run_overnight_max.ps1  (PowerShell-совместимый; без python heredoc)
$ErrorActionPreference = "Stop"

# 0) venv
. .\.venv\Scripts\Activate.ps1

# 0.1) логи
New-Item -ItemType Directory -Path logs -Force | Out-Null
$logFile = "logs\overnight_{0}.log" -f (Get-Date -Format yyyyMMdd_HHmmss)
Start-Transcript -Path $logFile -Append

Write-Host "`n[0] Env"
python -c "import torch,sys,platform; print('Python:',sys.version.split()[0]); print('Torch:',torch.__version__); print('CUDA:',torch.cuda.is_available()); print('GPU:', (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')); print('OS:',platform.system(),platform.release())"

# 1) Download in 4 waves (resume-safe)
$useCookies = Test-Path ".\cookies.txt"
for ($i=1; $i -le 4; $i++) {
  Write-Host "`n[1] Download wave $i/4 ..."
  if ($useCookies) { $env:YT_COOKIES="cookies.txt" } else { Remove-Item Env:YT_COOKIES -ErrorAction SilentlyContinue }
  python scripts\yt_bulk_dl.py | Tee-Object -FilePath ("logs\download_wave$($i).txt")
  Start-Sleep -Seconds 5
}

# 2) Pose extraction + dataset build (3.0s/0.5s)
Write-Host "`n[2] Build dataset (pose extraction + merge)"
python scripts\build_dataset_from_app.py --window 3.0 --stride 0.5 | Tee-Object -FilePath logs\extract.txt
if (!(Test-Path "data\mini\index.json")) { Write-Error "No data\mini\index.json"; Stop-Transcript; exit 1 }

Write-Host "`n[2a] Sanity"
python -c "import json,collections; idx=json.load(open(r'data/mini/index.json','r',encoding='utf-8')); cnt=collections.Counter([x['label'] for x in idx]); print('Windows:',len(idx)); print('unique labels:',len(cnt)); print('top10:',cnt.most_common(10))"

# 3) Train (balanced, AMP, early stop) -> runs\n30_max
Write-Host "`n[3] Train"
$trained=$false
foreach ($bs in 64,48,32) {
  Write-Host "batch_size=$bs"
  try {
    python -m src.train --data_root data\mini --epochs 30 --batch_size $bs --model tcn_tiny --save runs\n30_max --use_class_weights 1 --aug_joint_dropout 0.07 --aug_jitter 2.0 --aug_tempo 0.12 --early_stop 1 --amp 1 | Tee-Object -FilePath logs\train.txt
    if (Test-Path "runs\n30_max\best.ckpt") { $trained=$true; break }
  } catch { Write-Host "retry next bs..." -ForegroundColor Yellow }
}
if (-not $trained) { Write-Error "Training failed (no best.ckpt)"; Stop-Transcript; exit 1 }

# 4) Evaluate
Write-Host "`n[4] Eval"
python scripts\eval_metrics.py --weights runs\n30_max\best.ckpt --index data\mini\index.json --out_dir runs\n30_max --classes_yaml classes.yaml
if (!(Test-Path "runs\n30_max\metrics_test.json")) { Write-Error "No metrics_test.json"; Stop-Transcript; exit 1 }

Write-Host "`n[4a] Quick metrics"
python -c "import json; m=json.load(open(r'runs/n30_max/metrics_test.json','r',encoding='utf-8')); pcs=m.get('per_class_f1',{}); print('macroF1:', round(m.get('macro_f1',-1),4)); print('weakest 8:', sorted(pcs.items(), key=lambda x:x[1])[:8])"

# 5) Calibration (temperature & threshold)
Write-Host "`n[5] Calibration"
python scripts\calibrate_threshold.py --weights runs\n30_max\best.ckpt --index data\mini\index.json --target_unknown_rate 0.03

# 6) Export ONNX + bench
Write-Host "`n[6] Export + Bench"
python deploy\export_onnx.py --weights runs\n30_max\best.ckpt --out deploy\prism_tcn_n30.onnx
python scripts\benchmark_torch.py
python scripts\benchmark_onnx.py

# 7) Summary
Write-Host "`n=== OVERNIGHT SUMMARY ==="
Write-Host "Index:        data\mini\index.json"
Write-Host "Weights:      runs\n30_max\best.ckpt"
Write-Host "Metrics:      runs\n30_max\metrics_val.json , runs\n30_max\metrics_test.json"
Write-Host "Confusions:   runs\n30_max\confusion_val.png , runs\n30_max\confusion_test.png"
Write-Host "Calibration:  runs\n30_max\calibration.json"
Write-Host "ONNX:         deploy\prism_tcn_n30.onnx"
Write-Host "Logs:         logs\*.txt + transcript ($logFile)"
Stop-Transcript
