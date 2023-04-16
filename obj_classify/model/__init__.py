from beartype import beartype
from pytorch3d.structures import Meshes
import numpy as np
import torch
from torch import nn
from pytorch3d.ops import sample_points_from_meshes, ball_query


@beartype
class PointClassifier(nn.Module):
    def __init__(self, samples: int = 1000) -> None:
        super().__init__()
        self.samples = samples

        self.cluster_mlp = nn.Sequential(
            nn.Linear(40, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

        # self.dropout = nn.Dropout1d(inplace=True)

        self.classifier = nn.Sequential(
            nn.Linear(samples, samples // 2),
            nn.ReLU(),
            nn.Linear(samples // 2, samples // 4),
            nn.ReLU(),
            nn.Linear(samples // 4, samples // 8),
            nn.ReLU(),
            nn.Linear(samples // 8, samples // 16),
            nn.ReLU(),
            nn.Linear(samples // 16, 10),
            nn.Softmax(dim=1),
        )

    # Input: (batch_size, samples, 3)
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        query = ball_query(points, points, K=10, return_nn=True)
        dists: torch.Tensor = query[0]  # (batch_size, samples, 10)
        neighbors: torch.Tensor = query[2]  # (batch_size, samples, 10, 3)

        neighbor_dist = torch.cat([dists.unsqueeze(-1), neighbors], dim=-1).flatten(
            start_dim=2
        )  # (batch_size, samples, 40)
        pointwise_features: torch.Tensor = self.cluster_mlp(
            neighbor_dist
        )  # (batch_size, samples, 1)

        pointwise_features.squeeze_(-1)  # (batch_size, samples)

        # randomize ordering to preserve symmetrization
        indices = torch.from_numpy(np.random.permutation(self.samples)).to(
            points.device
        )
        pointwise_features = pointwise_features.index_select(1, indices)

        classification: torch.Tensor = self.classifier(
            pointwise_features
        )  # (batch_size, 10)
        return classification

    def prepare_data(self, meshes: Meshes) -> torch.Tensor:
        points = sample_points_from_meshes(meshes, num_samples=self.samples)
        assert isinstance(points, torch.Tensor)

        return points
