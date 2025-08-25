# main.py
# Hybrid LSTM (Prices + News Sentiment) for S&P 500 direction
# Clean, robust, and minimal.

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score

# --- project imports ---
from src.data_loader import build_price_dataset, quick_report
from src.utils import create_windows, TimeSeriesDataset
from src.models import LSTMClassifier
from src.trainer import train_model, evaluate_model, plot_training_curves, plot_confusion_matrix
from src.sentiment import score_headlines_df, aggregate_daily_sentiment
from src.preprocess import fit_scaler_on_train, transform_splits
from src.features import add_technical_features


RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# -------------------- helpers --------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns to single-level 'a__b' names (no-op if already flat)."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = ["__".join(map(str, c)).strip() for c in df.columns.values]
    return df


def resolve_price_feature_columns(df: pd.DataFrame):
    """
    Robustly find the price columns (Close/AdjClose, Ret, Volume) after flattening.
    Returns (close_col, ret_col, volume_col).
    """
    cols = list(df.columns)

    def find_col(patterns):
        pats = [p.lower() for p in patterns]
        for c in cols:
            name = str(c).lower().replace(" ", "").replace("-", "_")
            for p in pats:
                if p in name:
                    return c
        return None

    close_col = find_col(["close", "adjclose"])
    ret_col   = find_col(["ret"])
    vol_col   = find_col(["volume", "vol"])

    missing = [n for n, c in [("Close/AdjClose", close_col), ("Ret", ret_col), ("Volume", vol_col)] if c is None]
    if missing:
        raise ValueError(f"Could not resolve columns {missing}. Available: {cols}")

    print(f"[resolver] Using -> Close: {close_col} | Ret: {ret_col} | Volume: {vol_col}")
    return close_col, ret_col, vol_col


def ensure_standard_label(splits_dict: dict, ret_col_name: str) -> None:
    """
    Make sure each split has a column 'Label' (1 if next day's return > 0, else 0).
    If a label-like or nextret-like column exists, use it; otherwise build from Ret.shift(-1).
    """
    for name, df in splits_dict.items():
        cols = list(df.columns)
        labels = [c for c in cols if "label" in str(c).lower()]
        if labels:
            df["Label"] = df[labels[0]].astype(float)
        else:
            nextrets = [c for c in cols if "nextret" in str(c).lower()]
            if nextrets:
                df["Label"] = (df[nextrets[0]] > 0).astype(float)
            else:
                if ret_col_name not in df.columns:
                    raise ValueError(f"[ensure_standard_label] Ret '{ret_col_name}' not found in {name}")
                df["Label"] = (df[ret_col_name].shift(-1) > 0).astype(float)
        splits_dict[name] = df


def merge_sentiment(price_df: pd.DataFrame, daily_sent: pd.DataFrame) -> pd.DataFrame:
    """
    Robustly merge daily sentiment by plain 'Date' (avoid index-level merges),
    cap forward-fill to 3 days, neutral-fill otherwise.
    """
    out = flatten_cols(price_df.copy())

    # clean Date column from index (handles DatetimeIndex or MultiIndex)
    date_idx = out.index.get_level_values(0) if hasattr(out.index, "levels") else out.index
    date_col = pd.to_datetime(date_idx).tz_localize(None)

    out_reset = out.reset_index(drop=True)
    out_reset.insert(0, "Date", date_col.values)

    ds = daily_sent.copy().reset_index().rename(columns={"date": "Date"})
    ds["Date"] = pd.to_datetime(ds["Date"]).dt.tz_localize(None)
    if isinstance(ds.columns, pd.MultiIndex):
        ds.columns = ["__".join(map(str, c)).strip() for c in ds.columns.values]

    merged = pd.merge(out_reset, ds, on="Date", how="left").sort_values("Date")

    sent_cols = ["sent_neg", "sent_neu", "sent_pos", "sent_comp"]
    for c in sent_cols:
        if c not in merged.columns:
            merged[c] = np.nan

    merged[sent_cols] = merged[sent_cols].ffill(limit=3)
    merged[sent_cols] = merged[sent_cols].fillna(
        {"sent_neg": 0.0, "sent_neu": 1.0, "sent_pos": 0.0, "sent_comp": 0.0}
    )

    merged = merged.set_index("Date")
    return merged


def select_usable_features(df: pd.DataFrame, feature_cols: list[str], min_rows: int = 50) -> list[str]:
    """
    Keep only features with at least `min_rows` finite values in TRAIN.
    Prevents one bad column from killing the scaler fit.
    """
    keep, counts = [], {}
    for c in feature_cols:
        if c not in df.columns:
            counts[c] = 0
            continue
        cnt = int(np.isfinite(pd.to_numeric(df[c], errors="coerce")).sum())
        counts[c] = cnt
        if cnt >= min_rows:
            keep.append(c)
    print("[features] finite rows per feature:", {k: counts.get(k, 0) for k in feature_cols})
    if not keep:
        raise RuntimeError(f"No usable features (≥{min_rows} finite rows). Check features or lower min_rows.")
    if len(keep) < len(feature_cols):
        dropped = [c for c in feature_cols if c not in keep]
        print(f"[features] dropping unusable features: {dropped}")
    return keep


def probs_from_loader(model, loader, device):
    model.eval()
    ps, ys = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            logits = model(X)
            p = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            ps.append(p)
            ys.append(y.numpy())
    return np.concatenate(ps), np.concatenate(ys)


def backtest_from_preds(df_test: pd.DataFrame, ret_col: str, window_size: int, y_pred: np.ndarray):
    """
    Apply next-day return from the day after each window end.
    """
    start = window_size
    end = start + len(y_pred)
    ret_seq = df_test[ret_col].iloc[start:end].values

    L = min(len(y_pred), len(ret_seq))
    y_pred = y_pred[:L]
    ret_seq = ret_seq[:L]

    strat = (y_pred == 1).astype(float) * ret_seq
    mu = strat.mean()
    sd = strat.std(ddof=1) if len(strat) > 1 else 0.0
    sharpe = (mu / sd) * np.sqrt(252) if sd > 0 else 0.0
    cum_ret = float(np.exp(strat.sum()) - 1.0)
    return sharpe, cum_ret


# -------------------- main --------------------
if __name__ == "__main__":
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Prices: download & split
    splits = build_price_dataset(
        start="2005-01-01",
        end=None,
        interval="1d",
        train_until="2018-12-31",
        val_until="2021-12-31",
        save_processed=True,
    )
    # flatten price columns (handles baseline path too)
    splits = {k: flatten_cols(df) for k, df in splits.items()}
    for name, df in splits.items():
        quick_report(df, name)

    # 2) News -> FinBERT sentiment (expects data/news_headlines.csv)
    news_path = "data/news_headlines.csv"
    if not os.path.exists(news_path):
        raise FileNotFoundError(
            f"Missing {news_path}. Add a CSV with columns: date,headline (see scripts/prepare_kaggle_news.py)."
        )
    news_df = pd.read_csv(news_path)
    if not {"date", "headline"}.issubset(news_df.columns):
        raise ValueError("news_headlines.csv must have columns: date, headline")

    print(f"\n[news] loaded {len(news_df)} headlines")
    scored = score_headlines_df(news_df, text_col="headline", batch_size=16)
    daily_sent = aggregate_daily_sentiment(scored)
    print(f"[news] daily sentiment rows: {len(daily_sent)}")

    # 3) Merge sentiment into each split
    splits_sent = {k: merge_sentiment(df, daily_sent) for k, df in splits.items()}
    for name, df in splits_sent.items():
        miss = df[["sent_neg", "sent_neu", "sent_pos", "sent_comp"]].isna().sum().sum()
        print(f"[merge] {name}: shape={df.shape}, missing_sent_vals={miss}")

    # 4) Add technical features (prices only; sentiment already merged)
    splits_feat = {k: add_technical_features(df) for k, df in splits_sent.items()}

    # 5) Resolve columns & ensure labels
    close_col, ret_col, vol_col = resolve_price_feature_columns(splits_feat["train"])
    ensure_standard_label(splits_feat, ret_col_name=ret_col)

    # 6) Pick feature set (technicals + sentiment), filter unusable
    feature_cols = [
        ret_col,
        "Ret5", "Vol10",
        "SMA_slope", "MACD", "MACDsig", "MACDhist",
        "RSI14", "BBwidth", "VolZ20",
        "sent_neg", "sent_neu", "sent_pos", "sent_comp",
    ]
    feature_cols = select_usable_features(splits_feat["train"], feature_cols, min_rows=50)
    label_col = "Label"
    window_size = 30

    # 7) Scale (fit on train), transform, then window
    scaler = fit_scaler_on_train(splits_feat["train"], feature_cols)
    splits_scaled = transform_splits(splits_feat, feature_cols, scaler)

    X_train, y_train = create_windows(splits_scaled["train"], feature_cols, label_col, window_size)
    X_val, y_val = create_windows(splits_scaled["val"], feature_cols, label_col, window_size)
    X_test, y_test = create_windows(splits_scaled["test"], feature_cols, label_col, window_size)
    print(f"\n[windowing] Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")

    # 8) DataLoaders
    train_ds = TimeSeriesDataset(X_train, y_train)
    val_ds = TimeSeriesDataset(X_val, y_val)
    test_ds = TimeSeriesDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    # 9) Model + class weights
    model = LSTMClassifier(input_dim=len(feature_cols), hidden_dim=256, num_layers=2, dropout=0.3)
    pos_rate = float(y_train.mean()) if len(y_train) else 0.5
    pos_rate = min(max(pos_rate, 1e-6), 1 - 1e-6)
    w_neg = 0.5 / (1 - pos_rate)
    w_pos = 0.5 / pos_rate
    class_w = torch.tensor([w_neg, w_pos], dtype=torch.float32, device=device)
    print(f"[weights] class weights -> neg={w_neg:.3f}, pos={w_pos:.3f}")

    # 10) Train
    history = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=30,
        lr=3e-4,
        device=device,
        class_weights=class_w,
    )
    plot_training_curves(history)

    # 11) Choose threshold on VAL (maximize F1), then evaluate TEST
    p_val, y_val_true = probs_from_loader(model, val_loader, device)
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.linspace(0.3, 0.7, 41):
        y_hat = (p_val >= thr).astype(int)
        f1 = f1_score(y_val_true, y_hat)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    print(f"[thresh] chosen val threshold={best_thr:.3f}, val_f1={best_f1:.3f}")

    p_test, y_test_true = probs_from_loader(model, test_loader, device)
    y_pred = (p_test >= best_thr).astype(int)

    print("\nClassification Report (Hybrid on Test Set):")
    print(classification_report(y_test_true, y_pred, digits=3))
    plot_confusion_matrix(y_test_true, y_pred)

    # 12) Backtest
    sharpe, cum_ret = backtest_from_preds(splits_scaled["test"], ret_col, window_size, y_pred)
    print(f"\nBacktest: Sharpe={sharpe:.3f}, CumReturn={cum_ret:.3f}")

    # 13) Save weights
    model_path = os.path.join(RESULTS_DIR, "lstm_hybrid.pt")
    torch.save(model.state_dict(), model_path)
    print(f"\n Saved model to {model_path}")
