from pathlib import Path
import json, yaml, torch

def load_class_names(weights_path: str | None) -> list[str]:
    """
    Priority:
    1) weights.meta.class_names (PyTorch checkpoint)
    2) runs/<exp>/labels.json next to weights
    3) classes.yaml (top-level; key 'classes' or list of dicts with 'name')
    4) data/mini/index.json (derive sorted unique labels)
    """
    # 1) try checkpoint
    if weights_path:
        p = Path(weights_path)
        if p.suffix == ".ckpt" and p.exists():
            try:
                ckpt = torch.load(p, map_location="cpu")
                meta = ckpt.get("meta", {}) if isinstance(ckpt, dict) else {}
                names = meta.get("class_names")
                if isinstance(names, (list, tuple)) and len(names) > 0:
                    return list(names)
            except Exception:
                pass
        # 2) labels.json near weights (runs/<exp>/labels.json)
        if p.parent.joinpath("labels.json").exists():
            try:
                return json.loads(p.parent.joinpath("labels.json").read_text(encoding="utf-8"))
            except Exception:
                pass

    # 3) classes.yaml at repo root
    cy = Path("classes.yaml")
    if cy.exists():
        try:
            cfg = yaml.safe_load(cy.read_text(encoding="utf-8"))
            # support formats: {'classes':[{'name':...}, ...]} or {'classes':[...]} or plain list
            if isinstance(cfg, dict) and "classes" in cfg:
                items = cfg["classes"]
            else:
                items = cfg
            out = []
            for it in items:
                if isinstance(it, dict):
                    name = it.get("name") or it.get("label") or it.get("class")
                else:
                    name = str(it)
                if name: out.append(str(name))
            if out:
                return out
        except Exception:
            pass

    # 4) derive from data/mini/index.json
    idx = Path("data/mini/index.json")
    if idx.exists():
        try:
            index = json.loads(idx.read_text(encoding="utf-8"))
            labs = sorted({it["label"] for it in index})
            if labs: return labs
        except Exception:
            pass

    # fallback to classic 3
    return ["normal","run","fall"]

