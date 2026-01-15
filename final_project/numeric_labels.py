# numeric_labels.py
import numpy as np
import pandas as pd
import yfinance as yf

#  Load the text dataset (already has the yf_ticker)
df = pd.read_csv("financial_text_dataset_with_risk.csv")

print("Loaded financial_text_dataset_with_risk.csv")
print("Columns:", df.columns.tolist())
print(df[["company", "ticker", "yf_ticker", "filing_date"]].head())

def get_price_series(prices: pd.DataFrame, ticker: str) -> pd.Series | None:
    cols = prices.columns 
    if isinstance(cols, pd.MultiIndex): #multi-indexed columns
        candidates = [("Adj Close", ticker), ("Close", ticker)]
        for key in candidates:
            if key in cols:
                return prices.loc[:, key]
    else: #regular columns
        if "Adj Close" in cols:
            return prices["Adj Close"]
        if "Close" in cols:
            return prices["Close"]

    # If nothing matches, return None
    return None


def realized_vol(ticker, start_date, months=12):

    if pd.isna(ticker):
        return np.nan

    ticker = str(ticker).strip()
    if ticker == "":
        return np.nan

    start_date = pd.to_datetime(start_date, errors="coerce")
    if pd.isna(start_date):
        return np.nan

    end_date = start_date + pd.DateOffset(months=months)

    try:
        prices = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            auto_adjust=False,
            progress=False,
        )
    except Exception as e:
        print(f"yfinance error for {ticker}: {e}")
        return np.nan

    if prices is None or prices.empty:
        return np.nan

    price_series = get_price_series(prices, ticker)
    if price_series is None:
        print(f"Could not find Adj Close or Close for {ticker} in columns: {prices.columns.tolist()}")
        return np.nan

    returns = price_series.pct_change().dropna()
    if returns.empty:
        return np.nan

    vol = returns.std() * np.sqrt(252)  # annualized volatility
    return vol


vols = []
for i, row in df.iterrows():
    rv = realized_vol(row["yf_ticker"], row["filing_date"])
    vols.append(rv)

df["future_vol"] = vols

print("Final non-null future_vol count:", df["future_vol"].notna().sum())
print(df[["company", "yf_ticker", "filing_date", "future_vol"]].head(10))

df.to_csv("financial_text_dataset_with_risk.csv", index=False)
print("Wrote updated financial_text_dataset_with_risk.csv")
