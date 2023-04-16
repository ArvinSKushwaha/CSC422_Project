from pathlib import Path
from typing import Any
from beartype import beartype
from pytorch3d.structures import Meshes
from torch import optim
from tqdm import trange, tqdm
# from re import fullmatch
import torch
from torch.nn import functional as F
from .train import Dataset
from ..model import PointClassifier


@beartype
class Trainer:
    def __init__(self) -> None:
        location_of_ds = input("Where is the dataset? ")
        dataset = Dataset(location_of_ds)

        self.model_dir = Path("./model_history/")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        #
        # oldest: int = -1
        # for i in self.model_dir.iterdir():
        #     match = fullmatch(r"(\d{4}).pt", i.name)
        #     if match is None:
        #         continue
        #
        #     oldest = max(oldest, int(match.group()))
        #
        # if oldest != -1:
        #     self.save_number = oldest

        self.device = torch.device("cuda")

        self.mapping, self.train_series, self.test_series = dataset.to_data()

        self.train_series.set_device(self.device)
        self.test_series.set_device(self.device)
        self.classifier = PointClassifier().to(self.device)

        self.optimizer = optim.Adam(self.classifier.parameters(), lr=1e-3)
        self.metrics = {}

    def train(self, epochs: int = 30):
        for epoch in trange(epochs):
            self.on_epoch_start(epoch)

            for step, (meshes, labels) in enumerate(tqdm(self.train_series)):
                self.on_step_start(meshes, labels, step, epoch)
                probs, enriched = self.forward(meshes, epoch)

                loss = F.cross_entropy(probs, labels, label_smoothing=0.1)

                self.backward(loss, enriched)
                self.on_step_end(meshes, labels, probs, enriched, loss, step, epoch)

            self.on_epoch_end(epoch)

    def on_epoch_start(self, epoch: int):
        self.train_series.scramble()
        self.metrics = {
            "epoch_average": (int(0), int(0)),
        }

    def on_epoch_end(self, epoch: int):
        save_path = self.model_dir.joinpath(f"{epoch:04d}.pt")
        torch.save(self.classifier, save_path)

        correct, total = self.metrics["epoch_average"]
        tqdm.write(f"Epoch has average accuracy of: {correct/total}")

        tqdm.write("Validating")

        correct, total = 0, 0
        loss = 0.0
        for meshes, labels in tqdm(self.test_series):
            probs: torch.Tensor = self.classifier(self.classifier.prepare_data(meshes))
            loss += F.cross_entropy(probs, labels, reduction="sum").item()
            correct += int(torch.sum(probs.argmax(dim=1) == labels))
            total += labels.numel()

        tqdm.write(f"Validation Accuracy: {correct/total}, Loss: {loss/total}")

    def forward(self, meshes: Meshes, epoch: int) -> tuple[torch.Tensor, Any]:
        sampled_points = self.classifier.prepare_data(meshes)
        probs: torch.Tensor = self.classifier(sampled_points)

        return probs, sampled_points

    def backward(self, loss: torch.Tensor, enriched: Any):
        loss.backward()

    def on_step_start(
        self, meshes: Meshes, labels: torch.Tensor, step: int, epoch: int
    ):
        tqdm.write(f"Starting training step: {step} in epoch: {epoch}")
        self.optimizer.zero_grad()

    def on_step_end(
        self,
        meshes: Meshes,
        labels: torch.Tensor,
        probs: torch.Tensor,
        enriched: Any,
        loss: torch.Tensor,
        step: int,
        epoch: int,
    ):
        self.optimizer.step()

        correct = int(torch.sum(probs.argmax(dim=1) == labels).item())
        total = labels.numel()

        tqdm.write(f"Minibatch Accuracy: {correct/total}, Loss: {loss.item()}")

        epoch_correct, epoch_total = self.metrics["epoch_average"]
        epoch_correct += correct
        epoch_total += total

        self.metrics["epoch_average"] = (epoch_correct, epoch_total)
        tqdm.write(f"Epoch has average accuracy of: {epoch_correct/epoch_total}")
