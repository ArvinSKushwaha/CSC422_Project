from dataclasses import dataclass
from beartype import beartype
import numpy as np
from numpy import typing as npt
from pytorch3d.io.utils import PathManager
from pytorch3d.structures import Meshes
import torch
from enum import Enum, auto
from pytorch3d import io
from pathlib import Path


path_manager = PathManager()
path_manager.set_logging(False)


@beartype
class Set(Enum):
    TEST = auto()
    TRAIN = auto()


@beartype
@dataclass(slots=True)
class Object:
    category: str
    path: str
    set: Set


@beartype
@dataclass
class DataIterator:
    paths: npt.NDArray[np.object_]
    labels: torch.Tensor
    index: int
    batch_size: int
    len: int
    device: torch.device

    def __len__(self) -> int:
        return self.len

    def __iter__(self) -> "DataIterator":
        return self

    def __getitem__(self, idx: int) -> tuple[Meshes, torch.Tensor]:
        paths = self.paths[self.batch_size * idx : self.batch_size * (idx + 1)]
        labels = self.labels[self.batch_size * idx : self.batch_size * (idx + 1)].to(
            self.device
        )

        meshes = io.load_objs_as_meshes(
            paths.tolist(), load_textures=False, path_manager=path_manager
        ).to(self.device)
        return (meshes, labels)

    def __next__(self) -> tuple[Meshes, torch.Tensor]:
        if self.index == self.len:
            raise StopIteration

        self.index += 1
        ret = self[self.index - 1]
        return ret


@beartype
@dataclass
class DataSeries:
    paths: npt.NDArray[np.object_]
    labels: torch.Tensor
    batch_size: int = 64
    device: torch.device = torch.device("cpu")

    def scramble(self):
        indices = np.random.permutation(len(self.paths))

        self.paths = self.paths[indices]
        self.labels = self.labels[indices]

    def set_batch_size(self, batch_size: int):
        self.batch_size = batch_size

    def __len__(self) -> int:
        return len(self.paths) // self.batch_size

    def __getitem__(self, idx: int) -> tuple[Meshes, torch.Tensor]:
        paths = self.paths[self.batch_size * idx : self.batch_size * (idx + 1)]
        labels = self.labels[self.batch_size * idx : self.batch_size * (idx + 1)].to(
            self.device
        )

        meshes = io.load_objs_as_meshes(
            paths.tolist(), load_textures=False, path_manager=path_manager
        ).to(self.device)
        return (meshes, labels)

    def set_device(self, device: torch.device):
        self.device = device
        self.labels.to(self.device)

    def __iter__(self) -> DataIterator:
        return DataIterator(
            paths=self.paths,
            labels=self.labels,
            index=0,
            batch_size=self.batch_size,
            len=len(self.paths) // self.batch_size,
            device=self.device,
        )


@beartype
@dataclass(init=False, slots=True)
class Dataset:
    train_objs: list[Object]
    test_objs: list[Object]
    categories: set[str]

    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        self.categories = set()
        self.train_objs = []
        self.test_objs = []

        for cat in path.iterdir():
            if not cat.is_dir():
                continue

            category = cat.stem
            self.categories.add(category)

            train_path = cat.joinpath("train")
            for fp in train_path.iterdir():
                if not ".obj" == fp.suffix:
                    continue
                self.train_objs.append(
                    Object(category, str(fp.resolve(True)), Set.TRAIN)
                )

            test_path = cat.joinpath("test")
            for fp in test_path.iterdir():
                if not ".obj" in fp.suffixes:
                    continue
                self.test_objs.append(Object(category, str(fp.resolve(True)), Set.TEST))

    def to_data(self) -> tuple[dict[int, str], DataSeries, DataSeries]:
        inv_mapping = {i: category for i, category in enumerate(self.categories)}
        mapping = {category: i for i, category in inv_mapping.items()}

        train_samples = len(self.train_objs)
        test_samples = len(self.test_objs)

        train_paths: npt.NDArray[np.object_] = np.empty(train_samples, dtype=np.object_)
        train_y = torch.empty(train_samples, dtype=torch.int64)

        for i, obj in enumerate(self.train_objs):
            train_paths[i] = obj.path
            train_y[i] = mapping[obj.category]

        test_paths: npt.NDArray[np.object_] = np.empty(test_samples, dtype=np.object_)
        test_y = torch.empty(test_samples, dtype=torch.int64)

        for i, obj in enumerate(self.test_objs):
            test_paths[i] = obj.path
            test_y[i] = mapping[obj.category]

        return (
            inv_mapping,
            DataSeries(train_paths, torch.Tensor(train_y)),
            DataSeries(test_paths, torch.Tensor(test_y)),
        )
