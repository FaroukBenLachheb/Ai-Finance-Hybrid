# scripts/ab_compare.py
# A/B comparison: baseline (prices+technicals) vs hybrid (prices+technicals+sentiment)

# --- make 'src' importable whether run as module or file
import os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# --- project imports ---
from src.data_loader import build_price_dataset
from src.utils import create_windows, TimeSeriesDataset
from src.models import LSTMClassifier
from src.trainer import train_model
from src.sentiment import score_headlines_df, aggregate_daily_sentiment
from src.preprocess import fit_scaler_on_train, transform_splits
from src.features import add_technical_features


RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# -------------------- helpers --------------------
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns to single-level 'a__b' strings."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = ["__".join(map(str, c)).strip() for c in df.columns.values]
    return df


def resolve_price_feature_columns(df: pd.DataFrame):
    """
    Find Close/AdjClose, Ret, Volume columns after flattening.
    """
    cols = list(df.columns)
    def find_col(pats):
        pats = [p.lower() for p in pats]
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
        raise ValueError(f"Missing {missing}. Available: {cols}")
    print(f"[resolver] Using -> Close: {close_col} | Ret: {ret_col} | Volume: {vol_col}")
    return close_col, ret_col, vol_col


def ensure_standard_label(splits_dict: dict, ret_col: str):
    """
    Guarantee a clean 'Label' in each split:
      - Use existing label-like column if present;
      - Else if 'NextRet'-like exists, Label = (NextRet > 0);
      - Else Label = (Ret.shift(-1) > 0).
    """
    for name, df in splits_dict.items():
        labels = [c for c in df.columns if "label" in str(c).lower()]
        if labels:
            df["Label"] = df[labels[0]].astype(float)
        else:
            nextrets = [c for c in df.columns if "nextret" in str(c).lower()]
            if nextrets:
                df["Label"] = (df[nextrets[0]] > 0).astype(float)
            else:
                df["Label"] = (df[ret_col].shift(-1) > 0).astype(float)
        splits_dict[name] = df


def merge_sentiment(price_df: pd.DataFrame, daily_sent: pd.DataFrame) -> pd.DataFrame:
    """
    Merge daily sentiment by plain 'Date', cap ffill to 3 days, neutral-fill otherwise.
    """
    out = flatten_cols(price_df.copy())

    # clean Date column (works for DatetimeIndex or MultiIndex)
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
    Keep only features with at least `min_rows` finite values in TRAIN
    to avoid scaler fitting on zero rows.
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
    model.eval(); ps, ys = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            logits = model(X)
            p = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            ps.append(p); ys.append(y.numpy())
    return np.concatenate(ps), np.concatenate(ys)


def backtest_from_preds(df_test: pd.DataFrame, ret_col: str, window_size: int, y_pred: np.ndarray):
    """
    Each prediction corresponds to the window ending at index (window_size-1),
    so we apply next-day returns starting at index = window_size.
    """
    start = window_size
    end   = start + len(y_pred)
    ret_seq = df_test[ret_col].iloc[start:end].values

    L = min(len(y_pred), len(ret_seq))
    y_pred = y_pred[:L]; ret_seq = ret_seq[:L]

    strat = (y_pred == 1).astype(float) * ret_seq
    mu = strat.mean()
    sd = strat.std(ddof=1) if len(strat) > 1 else 0.0
    sharpe = (mu / sd) * np.sqrt(252) if sd > 0 else 0.0
    cum_ret = float(np.exp(strat.sum()) - 1.0)
    return sharpe, cum_ret


# -------------------- experiment runner --------------------
def run_once(use_sentiment: bool, device: str = None, epochs: int = 20, window_size: int = 30):
    set_seed(42)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Prices: build splits and flatten columns
    splits = build_price_dataset(
        start="2005-01-01", end=None, interval="1d",
        train_until="2018-12-31", val_until="2021-12-31", save_processed=False
    )
    splits = {k: flatten_cols(df) for k, df in splits.items()}

    # 2) Optional sentiment merge
    if use_sentiment:
        news_path = os.path.join(PROJECT_ROOT, "data", "news_headlines.csv")
        if not os.path.exists(news_path):
            raise FileNotFoundError("data/news_headlines.csv missing (needs columns: date,headline).")
        news_df = pd.read_csv(news_path)
        if not {"date", "headline"}.issubset(news_df.columns):
            raise ValueError("news_headlines.csv must have columns: date, headline")
        scored = score_headlines_df(news_df, text_col="headline", batch_size=16)
        daily_sent = aggregate_daily_sentiment(scored)
        splits = {k: merge_sentiment(df, daily_sent) for k, df in splits.items()}

    # 3) Add technical features
    splits_feat = {k: add_technical_features(df) for k, df in splits.items()}
    splits_feat = {k: flatten_cols(df) for k, df in splits_feat.items()}  # ensure single-level columns

    # 4) Resolve columns & ensure labels
    close_col, ret_col, vol_col = resolve_price_feature_columns(splits_feat["train"])
    ensure_standard_label(splits_feat, ret_col)

    # 5) Feature set (technicals; add sentiment for hybrid)
    feature_cols = [
        ret_col,
        "Ret5", "Vol10",
        "SMA_slope", "MACD", "MACDsig", "MACDhist",
        "RSI14", "BBwidth", "VolZ20",
    ]
    if use_sentiment:
        feature_cols += ["sent_neg", "sent_neu", "sent_pos", "sent_comp"]
    feature_cols = select_usable_features(splits_feat["train"], feature_cols, min_rows=50)
    label_col = "Label"

    # 6) Scale on TRAIN, transform all splits
    scaler = fit_scaler_on_train(splits_feat["train"], feature_cols)
    splits_scaled = transform_splits(splits_feat, feature_cols, scaler)

    # 7) Windowing
    def _window_all(win):
        Xtr, ytr = create_windows(splits_scaled["train"], feature_cols, label_col, win)
        Xva, yva = create_windows(splits_scaled["val"],   feature_cols, label_col, win)
        Xte, yte = create_windows(splits_scaled["test"],  feature_cols, label_col, win)
        return Xtr, ytr, Xva, yva, Xte, yte

    X_train, y_train, X_val, y_val, X_test, y_test = _window_all(window_size)
    if min(len(X_train), len(X_val), len(X_test)) == 0:
        print(f"[warn] zero samples with window={window_size}; retrying with window=20")
        X_train, y_train, X_val, y_val, X_test, y_test = _window_all(20)

    # 8) DataLoaders
    train_ds = TimeSeriesDataset(X_train, y_train)
    val_ds   = TimeSeriesDataset(X_val, y_val)
    test_ds  = TimeSeriesDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=64, shuffle=False)

    # 9) Model + class weights
    model = LSTMClassifier(input_dim=len(feature_cols), hidden_dim=256, num_layers=2, dropout=0.3)
    pos_rate = float(y_train.mean()) if len(y_train) else 0.5
    pos_rate = min(max(pos_rate, 1e-6), 1 - 1e-6)
    w_neg = 0.5 / (1 - pos_rate)
    w_pos = 0.5 / pos_rate
    class_w = torch.tensor([w_neg, w_pos], dtype=torch.float32, device=device)
    print(f"[weights] class weights -> neg={w_neg:.3f}, pos={w_pos:.3f}")

    history = train_model(
        model, train_loader, val_loader,
        num_epochs=epochs, lr=3e-4, device=device, class_weights=class_w
    )

    # 10) Choose threshold on VAL (maximize F1), then evaluate TEST
    p_val, y_val_true = probs_from_loader(model, val_loader, device)
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.linspace(0.3, 0.7, 41):
        y_hat = (p_val >= thr).astype(int)
        f1 = f1_score(y_val_true, y_hat)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    print(f"[thresh] best val threshold={best_thr:.3f}, val_f1={best_f1:.3f}")

    p_test, y_test_true = probs_from_loader(model, test_loader, device)
    y_pred = (p_test >= best_thr).astype(int)

    acc  = accuracy_score(y_test_true, y_pred)
    f1   = f1_score(y_test_true, y_pred)
    prec = precision_score(y_test_true, y_pred)
    rec  = recall_score(y_test_true, y_pred)

    # use original (unscaled) returns, but filtered to rows that survived scaling
    df_ret_test = splits_feat["test"].loc[splits_scaled["test"].index]
    sharpe, cum_ret = backtest_from_preds(df_ret_test, ret_col, window_size, y_pred)


    model_tag = "hybrid" if use_sentiment else "baseline"
    out = {
        "model": model_tag,
        "epochs": epochs,
        "features": len(feature_cols),
        "accuracy": round(float(acc), 4),
        "f1": round(float(f1), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "sharpe": round(float(sharpe), 4),
        "cum_return": round(float(cum_ret), 4),
    }

    # 11) Save row to CSV + weights
    csv_path = os.path.join(RESULTS_DIR, "ab_results.csv")
    pd.DataFrame([out]).to_csv(csv_path, mode="a", index=False, header=not os.path.exists(csv_path))
    print(f"[results] appended to {csv_path}: {out}")

    weights_path = os.path.join(RESULTS_DIR, f"lstm_{model_tag}.pt")
    torch.save(model.state_dict(), weights_path)
    print(f"[weights] saved {weights_path}")

    return out


# -------------------- entrypoint --------------------
if __name__ == "__main__":
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n=== Running BASELINE (prices+technicals) ===")
    base = run_once(use_sentiment=False, device=device, epochs=30)

    print("\n=== Running HYBRID (prices+technicals+sentiment) ===")
    hyb  = run_once(use_sentiment=True, device=device, epochs=30)

    print("\n=== Summary ===")
    print(pd.DataFrame([base, hyb]))
    print(f"\nCSV -> {os.path.join(RESULTS_DIR, 'ab_results.csv')}")
