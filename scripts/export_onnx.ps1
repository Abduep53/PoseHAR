
. .\.venv\Scripts\Activate.ps1
python deploy/export_onnx.py --weights runs/baseline/best.ckpt --out deploy/prism_tcn.onnx
