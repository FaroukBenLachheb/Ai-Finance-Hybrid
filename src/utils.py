# src/utils.py
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

def create_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    window_size: int = 30
) -> tuple[np.ndarray, np.ndarray]:
    """
    Make rolling windows and use the label aligned with the **last** timestep
    in the window (common convention when label already encodes next-day move).
    Skip any window with NaN/Inf or NaN target.
    """
    values = df[feature_cols].values.astype(np.float32)
    labels = df[label_col].values

    X, y = [], []
    for i in range(len(df) - window_size + 1):
        window = values[i:i + window_size]           # t ... t+W-1
        target = labels[i + window_size - 1]         # label at t+W-1
        if target != target:                         # NaN
            continue
        if not np.isfinite(window).all():            # NaN/Inf in features
            continue
        X.append(window)
        y.append(target)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class TimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).long()
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
