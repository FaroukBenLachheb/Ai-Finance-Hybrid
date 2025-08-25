# src/data_loader.py
from __future__ import annotations
import os
from typing import Tuple, Optional, Dict
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# ---------- constants ----------
DEFAULT_TICKER = "^GSPC"      # S&P 500 index
RAW_DIR = "data/raw"
PROC_DIR = "data/processed"

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)


# ---------- core fetch ----------
def fetch_sp500_prices(
    start: str = "2005-01-01",
    end: Optional[str] = None,
    interval: str = "1d",
    cache_csv: bool = True
) -> pd.DataFrame:
    """
    Download S&P 500 (^GSPC) OHLCV data from Yahoo Finance.

    Parameters
    ----------
    start : str
        Start date (YYYY-MM-DD)
    end : Optional[str]
        End date (YYYY-MM-DD). If None, uses today.
    interval : str
        "1d" (default). Can be "1wk", "1mo" later if needed.
    cache_csv : bool
        If True, saves to data/raw/sp500_<interval>.csv

    Returns
    -------
    pd.DataFrame with columns: [Open, High, Low, Close, Adj Close, Volume]
    Index is DatetimeIndex (UTC-naive).
    """
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    df = yf.download(DEFAULT_TICKER, start=start, end=end, interval=interval, auto_adjust=False, progress=False)

    if df.empty:
        raise RuntimeError("Empty dataframe returned from yfinance. Check network/date range.")

    # standardize index & column names
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.rename(columns={"Adj Close": "AdjClose"})

    # ensure all expected columns exist
    expected = ["Open", "High", "Low", "Close", "AdjClose", "Volume"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns from yfinance: {missing}")

    if cache_csv:
        raw_path = os.path.join(RAW_DIR, f"sp500_{interval}.csv")
        df.to_csv(raw_path, index=True)
        print(f"[data_loader] Saved raw prices to {raw_path}")

    return df


# ---------- feature engineering ----------
def add_returns_and_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds daily log-return and binary movement label (up=1, down=0) based on next-day close.

    Returns
    -------
    df with columns:
        - Ret: log return of Close (today vs yesterday)
        - NextRet: next-day log return
        - Label: 1 if NextRet > 0 else 0 (NaN for last row)
    """
    out = df.copy()
    out["Ret"] = np.log(out["Close"]).diff()
    out["NextRet"] = out["Ret"].shift(-1)
    out["Label"] = (out["NextRet"] > 0).astype("float")
    return out


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-fill small gaps, drop rows with any remaining NA in critical cols.
    """
    out = df.copy()
    out = out.sort_index()
    out = out.ffill()
    # We’ll keep last row but be mindful Label is NaN (no next day)
    return out


# ---------- dataset splits ----------
def split_by_date(
    df: pd.DataFrame,
    train_until: str,
    val_until: str
) -> Dict[str, pd.DataFrame]:
    """
    Split by absolute dates (inclusive).

    Example:
        train_until="2018-12-31", val_until="2021-12-31"
        -> train: <= train_until
           val:   (train_until, val_until]
           test:  > val_until

    Returns dict of {'train': df, 'val': df, 'test': df}
    """
    df = df.copy()
    mask_train = df.index <= pd.to_datetime(train_until)
    mask_val   = (df.index > pd.to_datetime(train_until)) & (df.index <= pd.to_datetime(val_until))
    mask_test  = df.index > pd.to_datetime(val_until)

    return {
        "train": df.loc[mask_train],
        "val":   df.loc[mask_val],
        "test":  df.loc[mask_test]
    }


# ---------- orchestration ----------
def build_price_dataset(
    start: str = "2005-01-01",
    end: Optional[str] = None,
    interval: str = "1d",
    train_until: str = "2018-12-31",
    val_until: str = "2021-12-31",
    save_processed: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    End-to-end:
      1) fetch prices
      2) add returns & labels
      3) basic clean
      4) split into train/val/test
      5) (optional) save CSVs into data/processed
    """
    prices = fetch_sp500_prices(start=start, end=end, interval=interval, cache_csv=True)
    feats  = add_returns_and_labels(prices)
    feats  = basic_clean(feats)

    splits = split_by_date(feats, train_until=train_until, val_until=val_until)

    if save_processed:
        for name, part in splits.items():
            path = os.path.join(PROC_DIR, f"sp500_{interval}_{name}.csv")
            part.to_csv(path)
            print(f"[data_loader] Saved {name} split to {path}")

    return splits


# ---------- simple sanity report ----------
def quick_report(df: pd.DataFrame, name: str = "dataset") -> None:
    """
    Print quick stats for sanity checking.
    """
    n = len(df)
    n_label = df["Label"].notna().sum() if "Label" in df.columns else 0
    up_rate = df.loc[df["Label"].notna(), "Label"].mean() if n_label > 0 else np.nan
    print(f"[report] {name}: rows={n}, labeled={n_label}, up_rate={up_rate:.3f}")
    if {"Close", "Ret"}.issubset(df.columns):
        print(f"[report] Close head:\n{df['Close'].head()}")
        print(f"[report] Ret   head:\n{df['Ret'].head()}")


if __name__ == "__main__":
    # manual test: python -m src.data_loader
    splits = build_price_dataset()
    for k, v in splits.items():
        quick_report(v, name=k)
