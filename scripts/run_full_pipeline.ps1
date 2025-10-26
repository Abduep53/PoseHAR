# Full Pipeline Runner for 30-Class PRISM Training
# This script runs the complete end-to-end pipeline

Write-Host "=== PRISM 30-Class Training Pipeline ===" -ForegroundColor Green
Write-Host "Starting full pipeline execution..." -ForegroundColor Yellow

# Step 1: Activate virtual environment
Write-Host "`n[1/8] Activating virtual environment..." -ForegroundColor Cyan
. .\.venv\Scripts\Activate.ps1
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to activate virtual environment" -ForegroundColor Red
    exit 1
}
Write-Host "✅ Virtual environment activated" -ForegroundColor Green

# Step 2: YouTube bulk download
Write-Host "`n[2/8] Starting YouTube bulk download..." -ForegroundColor Cyan
python scripts\yt_bulk_dl.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ YouTube download failed, retrying once..." -ForegroundColor Red
    python scripts\yt_bulk_dl.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ YouTube download failed after retry" -ForegroundColor Red
        exit 1
    }
}
Write-Host "✅ YouTube download completed" -ForegroundColor Green

# Step 3: Pose extraction
Write-Host "`n[3/8] Starting parallel pose extraction..." -ForegroundColor Cyan
python scripts\build_dataset_from_app.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Pose extraction failed, retrying once..." -ForegroundColor Red
    python scripts\build_dataset_from_app.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Pose extraction failed after retry" -ForegroundColor Red
        exit 1
    }
}
Write-Host "✅ Pose extraction completed" -ForegroundColor Green

# Step 4: Training
Write-Host "`n[4/8] Starting N-class model training..." -ForegroundColor Cyan
python -m src.train --data_root data\mini --epochs 25 --batch_size 64 --model tcn_tiny --save runs\n30
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Training failed, retrying once..." -ForegroundColor Red
    python -m src.train --data_root data\mini --epochs 25 --batch_size 64 --model tcn_tiny --save runs\n30
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Training failed after retry" -ForegroundColor Red
        exit 1
    }
}
Write-Host "✅ Training completed" -ForegroundColor Green

# Step 5: Evaluation
Write-Host "`n[5/8] Starting model evaluation..." -ForegroundColor Cyan
python scripts\eval_metrics.py --weights runs\n30\best.ckpt --index data\mini\index.json --out_dir runs\n30 --classes_yaml classes.yaml
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Evaluation failed, retrying once..." -ForegroundColor Red
    python scripts\eval_metrics.py --weights runs\n30\best.ckpt --index data\mini\index.json --out_dir runs\n30 --classes_yaml classes.yaml
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Evaluation failed after retry" -ForegroundColor Red
        exit 1
    }
}
Write-Host "✅ Evaluation completed" -ForegroundColor Green

# Step 6: ONNX export
Write-Host "`n[6/8] Starting ONNX export..." -ForegroundColor Cyan
python deploy\export_onnx.py --weights runs\n30\best.ckpt --out deploy\prism_tcn_n30.onnx
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ ONNX export failed, retrying once..." -ForegroundColor Red
    python deploy\export_onnx.py --weights runs\n30\best.ckpt --out deploy\prism_tcn_n30.onnx
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ ONNX export failed after retry" -ForegroundColor Red
        exit 1
    }
}
Write-Host "✅ ONNX export completed" -ForegroundColor Green

# Step 7: Latency benchmarks
Write-Host "`n[7/8] Running latency benchmarks..." -ForegroundColor Cyan
python scripts\benchmark_torch.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Torch benchmark failed" -ForegroundColor Red
}

python scripts\benchmark_onnx.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ ONNX benchmark failed" -ForegroundColor Red
}
Write-Host "✅ Latency benchmarks completed" -ForegroundColor Green

# Step 8: Final summary
Write-Host "`n[8/8] Generating final summary..." -ForegroundColor Cyan
Write-Host "`n=== 30-CLASS TRAINING SUMMARY ===" -ForegroundColor Green
Write-Host "Classes defined:" -ForegroundColor Yellow
Get-Content classes.yaml | Select-Object -First 5
Write-Host "..." -ForegroundColor Gray

$indexFile = "data\mini\index.json"
if (Test-Path $indexFile) {
    $totalWindows = (Get-Content $indexFile | ConvertFrom-Json).Count
    Write-Host "Total windows: $totalWindows" -ForegroundColor Yellow
} else {
    Write-Host "Total windows: Unable to count" -ForegroundColor Red
}

Write-Host "`nResults:" -ForegroundColor Yellow
Write-Host "• Val metrics: runs\n30\metrics_val.json" -ForegroundColor White
Write-Host "• Test metrics: runs\n30\metrics_test.json" -ForegroundColor White
Write-Host "• Confusion matrices: runs\n30\confusion_val.png, runs\n30\confusion_test.png" -ForegroundColor White
Write-Host "• ONNX model: deploy\prism_tcn_n30.onnx" -ForegroundColor White
Write-Host "• Latency results: runs\n30\latency_torch.json, runs\n30\latency_onnx.json" -ForegroundColor White

Write-Host "`n🎉 Pipeline completed successfully!" -ForegroundColor Green
Write-Host "All components are ready for deployment." -ForegroundColor Green
