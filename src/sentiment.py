# src/sentiment.py
from __future__ import annotations
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from typing import Optional, List
from tqdm import tqdm


FINBERT_MODEL_NAME = "ProsusAI/finbert"


def _load_finbert(device: str = "cpu"):
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL_NAME)
    model.to(device)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def score_headlines_df(
    df: pd.DataFrame,
    text_col: str = "headline",
    batch_size: int = 16,
    device: Optional[str] = None,
) -> pd.DataFrame:
    """
    Input df with at least [date, headline].
    Returns df with added columns: sent_neg, sent_neu, sent_pos, sent_label
    """
    assert text_col in df.columns, f"Missing column '{text_col}'"
    if "date" not in df.columns:
        raise ValueError("Input df must contain a 'date' column in YYYY-MM-DD or datetime format")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer, model = _load_finbert(device)

    texts = df[text_col].astype(str).tolist()
    probs_list: List[np.ndarray] = []
    labels: List[str] = []

    for i in tqdm(range(0, len(texts), batch_size), desc="[FinBERT] Scoring"):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}

        logits = model(**enc).logits  # [B, 3] -> order: {negative, neutral, positive}
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        probs_list.append(probs)

        pred_ids = probs.argmax(axis=1)
        id2label = {0: "negative", 1: "neutral", 2: "positive"}
        labels.extend([id2label[i] for i in pred_ids])

    probs_all = np.vstack(probs_list)
    out = df.copy()
    out["sent_neg"] = probs_all[:, 0]
    out["sent_neu"] = probs_all[:, 1]
    out["sent_pos"] = probs_all[:, 2]
    out["sent_label"] = labels

    # ensure date is normalized to date (no time)
    out["date"] = pd.to_datetime(out["date"]).dt.date
    return out


def aggregate_daily_sentiment(df_scored: pd.DataFrame) -> pd.DataFrame:
    """
    Group multiple headlines per day -> daily mean probs.
    Adds 'sent_comp' = sent_pos - sent_neg.
    Output index is DatetimeIndex (normalized to daily).
    """
    g = (
        df_scored
        .groupby("date")[["sent_neg", "sent_neu", "sent_pos"]]
        .mean()
        .reset_index()
    )
    g["sent_comp"] = g["sent_pos"] - g["sent_neg"]
    g["date"] = pd.to_datetime(g["date"])
    g = g.set_index("date").sort_index()
    return g
