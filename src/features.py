# src/features.py
from __future__ import annotations
import pandas as pd
import numpy as np
import re

def _ema(x: pd.Series, span: int) -> pd.Series:
    return x.ewm(span=span, adjust=False).mean()

def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    ret = close.diff()
    up = ret.clip(lower=0.0).rolling(window).mean()
    down = (-ret.clip(upper=0.0)).rolling(window).mean()
    rs = up / (down.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def _first_col(df: pd.DataFrame, pattern: str) -> pd.Series | None:
    # case-insensitive; match substring anywhere
    m = df.filter(regex=re.compile(pattern, re.I), axis=1)
    return None if m.empty else m.iloc[:, 0]

def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # --- robust helpers (match substrings, case-insensitive) ---
    close_s = out["Close"]  if "Close"  in out.columns else _first_col(out, r"adjclose|close")
    ret_s   = out["Ret"]    if "Ret"    in out.columns else _first_col(out, r"ret")
    vol_s   = out["Volume"] if "Volume" in out.columns else _first_col(out, r"volume|vol")

    out["Close"]  = pd.to_numeric(close_s, errors="coerce")
    out["Ret"]    = pd.to_numeric(ret_s,   errors="coerce")
    out["Volume"] = pd.to_numeric(vol_s,   errors="coerce")

    # --- features (with min_periods so they appear quickly) ---
    out["Ret5"]   = out["Ret"].rolling(5,  min_periods=5).sum()
    out["Vol10"]  = out["Ret"].rolling(10, min_periods=10).std()

    sma10 = out["Close"].rolling(10, min_periods=10).mean()
    sma20 = out["Close"].rolling(20, min_periods=20).mean()
    out["SMA_slope"] = (sma10 - sma20) / (out["Close"].rolling(20, min_periods=20).std() + 1e-8)

    ema12 = _ema(out["Close"], 12)
    ema26 = _ema(out["Close"], 26)
    macd  = ema12 - ema26
    signal = _ema(macd, 9)
    out["MACD"]     = macd
    out["MACDsig"]  = signal
    out["MACDhist"] = macd - signal

    out["RSI14"] = _rsi(out["Close"], 14)

    mid = out["Close"].rolling(20, min_periods=20).mean()
    sd  = out["Close"].rolling(20, min_periods=20).std()
    out["BBwidth"] = (4.0 * sd) / (mid + 1e-8)

    out["VolZ20"] = (out["Volume"] - out["Volume"].rolling(20, min_periods=20).mean()) / (
        out["Volume"].rolling(20, min_periods=20).std() + 1e-8
    )
    return out
