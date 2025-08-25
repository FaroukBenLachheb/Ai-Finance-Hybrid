# src/preprocess.py
from __future__ import annotations
import os
from typing import Dict, List
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

PROC_DIR = "data/processed"
os.makedirs(PROC_DIR, exist_ok=True)

def fit_scaler_on_train(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    scaler_path: str = os.path.join(PROC_DIR, "feature_scaler.joblib"),
) -> StandardScaler:
    """
    Fit scaler on TRAIN rows that are finite (no NaN/Inf).
    """
    feats = train_df[feature_cols]
    mask = np.isfinite(feats.values).all(axis=1)
    clean = feats.loc[mask]
    scaler = StandardScaler().fit(clean.values)
    joblib.dump(scaler, scaler_path)
    print(f"[preprocess] Saved scaler to {scaler_path} (fit on {len(clean)}/{len(feats)} rows)")
    return scaler

def transform_splits(
    splits: Dict[str, pd.DataFrame],
    feature_cols: List[str],
    scaler: StandardScaler,
) -> Dict[str, pd.DataFrame]:
    """
    Transform and DROP any rows that remain non-finite in features.
    """
    out = {}
    for name, df in splits.items():
        df2 = df.copy()
        vals = df2[feature_cols].values
        scaled = scaler.transform(vals)
        df2[feature_cols] = scaled
        # drop non-finite rows in features
        mask = np.isfinite(df2[feature_cols].values).all(axis=1)
        dropped = len(df2) - mask.sum()
        if dropped > 0:
            print(f"[preprocess] Dropped {dropped} non-finite rows from split '{name}' after scaling")
        out[name] = df2.loc[mask]
    return out
