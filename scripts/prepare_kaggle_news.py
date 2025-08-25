# scripts/prepare_kaggle_news.py
import os, sys, pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
in_path  = os.path.join(PROJECT_ROOT, "data", "external", "Combined_News_DJIA.csv")
out_path = os.path.join(PROJECT_ROOT, "data", "news_headlines.csv")

if not os.path.exists(in_path):
    raise FileNotFoundError(f"Missing {in_path}\nDownload Kaggle 'Combined_News_DJIA.csv' and place it there.")

df = pd.read_csv(in_path, encoding="latin1")
# expected columns: Date, Label, Top1..Top25 (sometimes Top1..Top30)
headline_cols = [c for c in df.columns if str(c).lower().startswith("top")]
if not headline_cols:
    raise ValueError("No TopN headline columns found.")

# melt to rows: one headline per row
m = df.melt(id_vars=["Date"], value_vars=headline_cols, var_name="slot", value_name="headline")
m = m.dropna(subset=["headline"])
m["headline"] = m["headline"].astype(str).str.strip()
m = m[m["headline"].str.len() > 0]
m = m[["Date", "headline"]].rename(columns={"Date": "date"}).sort_values("date")

# normalize date to YYYY-MM-DD
m["date"] = pd.to_datetime(m["date"]).dt.date

os.makedirs(os.path.dirname(out_path), exist_ok=True)
m.to_csv(out_path, index=False)
print(f"[prepare_kaggle_news] wrote {len(m):,} rows to {out_path}")
