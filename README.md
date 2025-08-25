# AI×Finance — Hybrid Price + News Sentiment (S&P 500)

End-to-end pipeline that predicts **next-day S&P 500 direction** with:
- **Price technicals** (returns, MACD, RSI, SMA slope, BB width, volume z-score)
- **Daily FinBERT sentiment** from news headlines
- **LSTM** classifier with class-weighted loss, clean scaling/windowing, and validation-threshold selection  
- A/B script to compare **baseline (prices only)** vs **hybrid (prices + sentiment)**

> Educational project. Not financial advice.

---

## Highlights
- Clean Python layout: `src/` (loader, features, model, trainer) + `scripts/` (A/B + data prep)
- Robust handling (MultiIndex → flat columns, NaN-safe features, alignment-safe backtest)
- Metrics: Accuracy, F1, plus a simple long-only backtest (Sharpe & cumulative return)

---

## Quickstart

```bash
# 1) create & activate env
python -m venv venv
# Windows:  .\venv\Scripts\Activate.ps1
# macOS/Linux: source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

# 2) (recommended) prepare headlines (Kaggle DJIA → news_headlines.csv)
python scripts/prepare_kaggle_news.py

# 3) A/B comparison: baseline vs hybrid
python -m scripts.ab_compare

# 4) Single hybrid run with plots/backtest
python main.py



##Repo Structure
.
├─ main.py                         # single hybrid run (plots + backtest)
├─ scripts/
│  ├─ ab_compare.py               # A/B: baseline vs hybrid
│  └─ prepare_kaggle_news.py      # DJIA headlines → data/news_headlines.csv
├─ src/
│  ├─ data_loader.py  features.py  models.py  preprocess.py
│  ├─ sentiment.py    trainer.py   utils.py
├─ data/
│  ├─ external/Combined_News_DJIA.csv  (put Kaggle CSV here)
│  ├─ news_headlines.csv  (generated: date,headline)
│  ├─ raw/  processed/
├─ results/
└─ requirements.txt

📄 **Paper (PDF):** [Hybrid Price + News Sentiment for S&P 500](./docs/AixFinance_paper.pdf)
