
# Create venv using python or py fallback
if (!(Test-Path .\.venv)) {
  try { python -m venv .venv } catch { }
  if (!(Test-Path .\.venv)) { try { py -3.10 -m venv .venv } catch { } }
}
$venvPy = Join-Path (Resolve-Path .\.venv).Path "Scripts/python.exe"
if (!(Test-Path $venvPy)) { throw "Virtualenv python not found at $venvPy" }
& $venvPy -m pip install --upgrade pip
& $venvPy -m pip install -r requirements.txt
