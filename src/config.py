
from dataclasses import dataclass
@dataclass
class TrainConfig:
    data_root: str = "data/mini"
    epochs: int = 5
    batch_size: int = 32
    lr: float = 1e-3
    model: str = "tcn_tiny"
    save_dir: str = "runs/baseline"
    window_size: int = 60  # ~2s @30fps
    joints: int = 33
    seed: int = 42
