
. .\.venv\Scripts\Activate.ps1
python -m src.data.make_dataset --video app/demo_clip.mp4 --out data/mini --window_sec 2.0
python -m src.train --data_root data/mini --epochs 5 --model tcn_tiny --save runs/baseline
