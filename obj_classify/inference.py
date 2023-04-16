from pathlib import Path
import torch
from torch import nn, optim, jit
from torch.nn import functional as F
from pytorch3d import io, vis


class Inference:
    def __init__(self) -> None:
        pass

    def classify(self, file: str | Path):
        path = Path(file)
        pass
