"""Independent CNN predictor for promoter log10 expression strength."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..data import DNA_ALPHABET, one_hot_encode_sequences
from ..metrics import regression_metrics


def _group_norm(channels: int) -> nn.GroupNorm:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


class MultiScalePromoterCNN(nn.Module):
    """Multi-scale 1D CNN that scores a one-hot promoter."""

    def __init__(
        self,
        n_bases: int,
        sequence_length: int,
        stem_channels: int,
        kernel_sizes: tuple[int, ...],
        merge_channels: int,
        hidden_size: int,
        position_hidden_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.stems = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(n_bases, stem_channels, kernel_size=kernel, padding=kernel // 2),
                    _group_norm(stem_channels),
                    nn.ReLU(inplace=True),
                )
                for kernel in kernel_sizes
            ]
        )
        merged = stem_channels * len(kernel_sizes)
        self.trunk = nn.Sequential(
            nn.Conv1d(merged, merge_channels, kernel_size=5, padding=2),
            _group_norm(merge_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(merge_channels, merge_channels, kernel_size=3, padding=1),
            _group_norm(merge_channels),
            nn.ReLU(inplace=True),
        )
        self.position_head = (
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(merge_channels * sequence_length, position_hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            if position_hidden_size > 0
            else None
        )
        head_input_size = merge_channels * 2 + max(0, position_hidden_size)
        self.head = nn.Sequential(
            nn.Linear(head_input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        stems = torch.cat([stem(encoded) for stem in self.stems], dim=1)
        hidden = self.trunk(stems)
        pooled = torch.cat((hidden.mean(dim=-1), hidden.amax(dim=-1)), dim=1)
        if self.position_head is not None:
            pooled = torch.cat((pooled, self.position_head(hidden)), dim=1)
        return self.head(pooled).squeeze(-1)


class CNNStrengthPredictor:
    """Trainable CNN strength model with the architecture stored in the checkpoint."""

    def __init__(
        self,
        sequence_length: int = 50,
        alphabet: str = DNA_ALPHABET,
        stem_channels: int = 32,
        kernel_sizes: tuple[int, ...] = (3, 5, 7),
        merge_channels: int = 64,
        hidden_size: int = 64,
        position_hidden_size: int = 64,
        dropout: float = 0.35,
        learning_rate: float = 5e-4,
        weight_decay: float = 1e-3,
        batch_size: int = 128,
        max_epochs: int = 100,
        patience: int = 20,
        min_delta: float = 0.002,
        loss_name: str = "huber",
        selection_metric: str = "pearson",
        calibration: str = "median_bias",
        correlation_weight: float = 0.5,
        mutation_rate: float = 0.25,
        device: str | None = None,
    ) -> None:
        if sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        self.sequence_length = int(sequence_length)
        self.alphabet = str(alphabet)
        self.stem_channels = int(stem_channels)
        self.kernel_sizes = tuple(int(value) for value in kernel_sizes)
        self.merge_channels = int(merge_channels)
        self.hidden_size = int(hidden_size)
        self.position_hidden_size = int(position_hidden_size)
        self.dropout = float(dropout)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.loss_name = str(loss_name).lower()
        self.selection_metric = str(selection_metric).lower()
        self.calibration = str(calibration).lower()
        if self.loss_name not in {"huber", "mse"}:
            raise ValueError("loss_name must be 'huber' or 'mse'")
        if self.selection_metric not in {"mae", "rmse", "pearson", "spearman"}:
            raise ValueError("selection_metric must be mae, rmse, pearson or spearman")
        if self.calibration not in {"none", "median_bias"}:
            raise ValueError("calibration must be 'none' or 'median_bias'")
        self.correlation_weight = float(correlation_weight)
        self.mutation_rate = float(mutation_rate)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.target_mean: float = 0.0
        self.target_std: float = 1.0
        self.prediction_offset: float = 0.0
        self.best_epoch: int | None = None
        self.evaluation_metrics: dict[str, Any] = {}
        self.history: dict[str, list[float]] = {"train_mse": [], "validation_mae": []}
        self.network = MultiScalePromoterCNN(
            n_bases=len(self.alphabet),
            sequence_length=self.sequence_length,
            stem_channels=self.stem_channels,
            kernel_sizes=self.kernel_sizes,
            merge_channels=self.merge_channels,
            hidden_size=self.hidden_size,
            position_hidden_size=self.position_hidden_size,
            dropout=self.dropout,
        ).to(self.device)

    @property
    def spec(self) -> dict[str, Any]:
        """Return the data contract and architecture needed to reload the model."""
        return {
            "model_name": "cnn_strength",
            "input_encoding": "one_hot_ncl",
            "label_transform": "log10",
            "sequence_length": self.sequence_length,
            "alphabet": self.alphabet,
            "stem_channels": self.stem_channels,
            "kernel_sizes": list(self.kernel_sizes),
            "merge_channels": self.merge_channels,
            "hidden_size": self.hidden_size,
            "position_hidden_size": self.position_hidden_size,
            "position_feature_source": "convolutional_trunk" if self.position_hidden_size > 0 else "none",
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "loss_name": self.loss_name,
            "selection_metric": self.selection_metric,
            "calibration": self.calibration,
            "correlation_weight": self.correlation_weight,
            "mutation_rate": self.mutation_rate,
            "feature_layout": "one_hot_channels_by_length",
            "feature_count": self.sequence_length * len(self.alphabet),
        }

    def _encode(self, sequences: np.ndarray) -> torch.Tensor:
        encoded = one_hot_encode_sequences(sequences, self.sequence_length, self.alphabet)
        return torch.from_numpy(encoded)

    def _dataloader(self, sequences: np.ndarray, targets: np.ndarray | None, shuffle: bool, seed: int) -> DataLoader:
        encoded = self._encode(sequences)
        if targets is None:
            dataset = TensorDataset(encoded)
        else:
            dataset = TensorDataset(encoded, torch.from_numpy(np.asarray(targets, dtype=np.float32)))
        batch_size = max(2, min(self.batch_size, len(dataset)))
        generator = torch.Generator()
        generator.manual_seed(seed)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False,
            generator=generator if shuffle else None,
        )

    def _mutate_onehot(self, encoded: torch.Tensor) -> torch.Tensor:
        if self.mutation_rate <= 0.0:
            return encoded
        batch, channels, length = encoded.shape
        mutated = encoded.clone()
        replace = torch.rand(batch, device=encoded.device) < self.mutation_rate
        if not bool(replace.any()):
            return mutated
        positions = torch.randint(0, length, (batch,), device=encoded.device)
        bases = torch.randint(0, channels, (batch,), device=encoded.device)
        indices = replace.nonzero(as_tuple=False).squeeze(-1)
        mutated[indices, :, positions[indices]] = 0.0
        mutated[indices, bases[indices], positions[indices]] = 1.0
        return mutated

    @staticmethod
    def _correlation_loss(predicted: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        predicted = predicted - predicted.mean()
        targets = targets - targets.mean()
        denominator = predicted.norm() * targets.norm()
        if float(denominator) < 1e-8:
            return predicted.new_zeros(())
        return 1.0 - (predicted * targets).sum() / denominator

    def fit(
        self,
        sequences: np.ndarray,
        targets: np.ndarray,
        validation_sequences: np.ndarray,
        validation_targets: np.ndarray,
        seed: int = 0,
    ) -> "CNNStrengthPredictor":
        torch.manual_seed(seed)
        self.prediction_offset = 0.0
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        y_train = np.asarray(targets, dtype=np.float32)
        self.target_mean = float(y_train.mean())
        self.target_std = float(y_train.std())
        if self.target_std < 1e-6:
            self.target_std = 1.0
        scaled_train = (y_train - self.target_mean) / self.target_std
        scaled_validation = (np.asarray(validation_targets, dtype=np.float32) - self.target_mean) / self.target_std

        train_loader = self._dataloader(sequences, scaled_train, shuffle=True, seed=seed)
        validation_loader = self._dataloader(validation_sequences, scaled_validation, shuffle=False, seed=seed)

        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs)
        regression_loss = nn.SmoothL1Loss(beta=0.5) if self.loss_name == "huber" else nn.MSELoss()
        best_score = -float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        stalled = 0
        self.history = {
            "train_loss": [],
            "validation_mae": [],
            "validation_rmse": [],
            "validation_pearson": [],
            "validation_spearman": [],
        }

        for epoch in range(1, self.max_epochs + 1):
            self.network.train()
            train_losses = []
            for batch_x, batch_y in train_loader:
                batch_x = self._mutate_onehot(batch_x.to(self.device))
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                predicted = self.network(batch_x)
                loss = regression_loss(predicted, batch_y)
                if self.correlation_weight > 0.0 and len(batch_y) > 2:
                    loss = loss + self.correlation_weight * self._correlation_loss(predicted, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 5.0)
                optimizer.step()
                train_losses.append(float(loss.detach().cpu()))
            scheduler.step()

            validation = self._validation_metrics(validation_loader)
            self.history["train_loss"].append(float(np.mean(train_losses)))
            self.history["validation_mae"].append(validation["mae"])
            self.history["validation_rmse"].append(validation["rmse"])
            self.history["validation_pearson"].append(validation["pearson"])
            self.history["validation_spearman"].append(validation["spearman"])

            selected_value = validation[self.selection_metric]
            score = -selected_value if self.selection_metric in {"mae", "rmse"} else selected_value
            if score > best_score + self.min_delta:
                best_score = score
                best_state = {key: value.detach().cpu().clone() for key, value in self.network.state_dict().items()}
                self.best_epoch = epoch
                stalled = 0
            else:
                stalled += 1
                if stalled >= self.patience:
                    break

        if best_state is None:
            raise RuntimeError("CNN training failed to produce a checkpoint")
        self.network.load_state_dict(best_state)
        self.network.to(self.device)
        self.network.eval()
        if self.calibration == "median_bias":
            validation_predictions = self.predict(validation_sequences)
            self.prediction_offset = float(
                np.median(np.asarray(validation_targets, dtype=np.float64) - validation_predictions)
            )
        return self

    def _validation_metrics(self, loader: DataLoader) -> dict[str, float]:
        self.network.eval()
        predicted_batches = []
        target_batches = []
        with torch.no_grad():
            for batch_x, batch_y in loader:
                predicted_batches.append(self.network(batch_x.to(self.device)).cpu().numpy())
                target_batches.append(batch_y.numpy())
        predicted = np.concatenate(predicted_batches) * self.target_std + self.target_mean
        targets = np.concatenate(target_batches) * self.target_std + self.target_mean
        return regression_metrics(targets, predicted)

    def _scaled_mae(self, loader: DataLoader) -> float:
        return self._validation_metrics(loader)["mae"]

    def predict(self, sequences: np.ndarray) -> np.ndarray:
        self.network.eval()
        loader = self._dataloader(sequences, None, shuffle=False, seed=0)
        outputs = []
        with torch.no_grad():
            for (batch_x,) in loader:
                predicted = self.network(batch_x.to(self.device)).cpu().numpy()
                outputs.append(predicted * self.target_std + self.target_mean)
        return np.concatenate(outputs).astype(np.float64) + self.prediction_offset

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "status": "independent_strength_evaluator",
                "model_name": "cnn_strength",
                "spec": self.spec,
                "state_dict": {key: value.detach().cpu() for key, value in self.network.state_dict().items()},
                "target_mean": self.target_mean,
                "target_std": self.target_std,
                "prediction_offset": self.prediction_offset,
                "best_epoch": self.best_epoch,
                "history": self.history,
                "evaluation_metrics": self.evaluation_metrics,
            },
            output,
        )

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "CNNStrengthPredictor":
        checkpoint_path = Path(path)
        if checkpoint_path.suffix == ".npz":
            raise ValueError(
                "Received a ridge .npz checkpoint; retrain the independent evaluator "
                "with scripts/train_independent_evaluator.py to produce model.pt"
            )
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        spec = dict(checkpoint["spec"])
        model = cls(
            sequence_length=int(spec["sequence_length"]),
            alphabet=str(spec["alphabet"]),
            stem_channels=int(spec["stem_channels"]),
            kernel_sizes=tuple(spec["kernel_sizes"]),
            merge_channels=int(spec["merge_channels"]),
            hidden_size=int(spec["hidden_size"]),
            position_hidden_size=int(spec.get("position_hidden_size", 0)),
            dropout=float(spec["dropout"]),
            learning_rate=float(spec.get("learning_rate", 5e-4)),
            weight_decay=float(spec.get("weight_decay", 1e-3)),
            batch_size=int(spec.get("batch_size", 128)),
            max_epochs=int(spec.get("max_epochs", 100)),
            patience=int(spec.get("patience", 20)),
            min_delta=float(spec.get("min_delta", 0.002)),
            loss_name=str(spec.get("loss_name", "huber")),
            selection_metric=str(spec.get("selection_metric", "pearson")),
            calibration=str(spec.get("calibration", "none")),
            correlation_weight=float(spec.get("correlation_weight", 0.5)),
            mutation_rate=float(spec.get("mutation_rate", 0.25)),
            device=device,
        )
        model.network.load_state_dict(checkpoint["state_dict"])
        model.network.to(model.device)
        model.network.eval()
        model.target_mean = float(checkpoint["target_mean"])
        model.target_std = float(checkpoint["target_std"])
        model.prediction_offset = float(checkpoint.get("prediction_offset", 0.0))
        model.best_epoch = None if checkpoint.get("best_epoch") is None else int(checkpoint["best_epoch"])
        model.history = checkpoint.get("history", {"train_mse": [], "validation_mae": []})
        model.evaluation_metrics = dict(checkpoint.get("evaluation_metrics", {}))
        return model
