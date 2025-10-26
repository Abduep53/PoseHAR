. .\.venv\Scripts\Activate.ps1
$bs=64
foreach ($b in 64,48,32) {
  try {
    python -m src.train --data_root data\mini --epochs 25 --batch_size $b --model tcn_tiny --save runs\n30_bal --aug_joint_dropout 0.07 --aug_jitter 2.0 --aug_tempo 0.12 --use_class_weights 1 --early_stop 1
    if (Test-Path runs\n30_bal\best.ckpt) { $bs=$b; break }
  } catch {}
}
python scripts\eval_metrics.py --weights runs\n30_bal\best.ckpt --index data\mini\index.json --out_dir runs\n30_bal --classes_yaml classes.yaml
python deploy\export_onnx.py --weights runs\n30_bal\best.ckpt --out deploy\prism_tcn_n30.onnx
