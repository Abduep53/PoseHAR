. .\.venv\Scripts\Activate.ps1
python scripts\build_dataset_from_app.py --window 3.0 --stride 0.5
Write-Host "Windows:" (Get-Content data\mini\index.json).Length
python -c "import json,collections; idx=json.load(open(r'data/mini/index.json','r',encoding='utf-8')); cnt=collections.Counter([it['label'] for it in idx]); print('unique labels:', len(cnt)); print(cnt.most_common(10))"
