"""
Fetch daily OHLCV data. Priority: Polygon > Finnhub > Yahoo Finance.
Polygon/Finnhub = more reliable (no 401s). yfinance = free fallback.
"""
import os
from datetime import datetime, timedelta

import pandas as pd


def _get_finnhub_key() -> str | None:
    """Return Finnhub API key from env or config.yaml."""
    key = os.environ.get("FINNHUB_API_KEY")
    if key:
        return key
    try:
        import yaml
        cfg = os.path.join(os.path.dirname(__file__), "config.yaml")
        if os.path.exists(cfg):
            with open(cfg) as f:
                c = yaml.safe_load(f) or {}
            k = c.get("finnhub_api_key") or c.get("FINNHUB_API_KEY")
            if k:
                return str(k).strip()
    except Exception:
        pass
    return None


def _fetch_finnhub(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch from Finnhub stock/candle API. Daily only. Max 1 year per request; chunks if needed."""
    key = _get_finnhub_key()
    if not key:
        return pd.DataFrame()
    try:
        import urllib.request
        import json

        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        all_rows = []
        # Finnhub daily: max 1 year per request
        chunk_end = end_dt
        while chunk_end >= start_dt:
            chunk_start = max(start_dt, chunk_end - timedelta(days=365))
            from_ts = int(chunk_start.timestamp())
            to_ts = int(chunk_end.timestamp())
            url = (
                f"https://finnhub.io/api/v1/stock/candle?symbol={ticker}"
                f"&resolution=D&from={from_ts}&to={to_ts}&token={key}"
            )
            with urllib.request.urlopen(url, timeout=15) as r:
                data = json.loads(r.read().decode())
            if not data.get("t"):
                break
            ts = data["t"]
            vols = data.get("v") or [0] * len(ts)
            for i, t in enumerate(ts):
                all_rows.append({
                    "date": pd.Timestamp.utcfromtimestamp(t).normalize(),
                    "Open": data["o"][i],
                    "High": data["h"][i],
                    "Low": data["l"][i],
                    "Close": data["c"][i],
                    "Volume": vols[i] if i < len(vols) else 0,
                })
            chunk_end = chunk_start - timedelta(days=1)
            if chunk_end >= start_dt:
                import time
                time.sleep(0.5)  # Respect 60/min rate limit
        if not all_rows:
            return pd.DataFrame()
        df = pd.DataFrame(all_rows).set_index("date").sort_index()
        df = df[~df.index.duplicated(keep="first")]
        return df
    except Exception:
        return pd.DataFrame()


def _fetch_polygon(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch from Polygon.io. Returns empty DataFrame on error."""
    key = os.environ.get("POLYGON_API_KEY")
    if not key:
        return pd.DataFrame()
    try:
        import urllib.request
        # Aggregate bars endpoint
        url = (
            f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/"
            f"{start_date}/{end_date}?adjusted=true&sort=asc&limit=50000&apiKey={key}"
        )
        with urllib.request.urlopen(url, timeout=15) as r:
            import json
            data = json.loads(r.read().decode())
        if not data.get("results"):
            return pd.DataFrame()
        rows = data["results"]
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["t"], unit="ms")
        df = df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
        df = df.set_index("date")[["Open", "High", "Low", "Close", "Volume"]]
        return df
    except Exception:
        return pd.DataFrame()


def fetch_vix(start_date: str, end_date: str) -> pd.Series:
    """
    Fetch VIX close from yfinance. Used as macro feature for market stress.
    Returns Series indexed by date; empty on error.
    """
    try:
        import yfinance as yf
        df = yf.download("^VIX", start=start_date, end=end_date, progress=False)
        if df.empty or "Close" not in df.columns:
            return pd.Series(dtype=float)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        return df["Close"]
    except Exception:
        return pd.Series(dtype=float)


def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches stock data from Yahoo Finance, saves to CSV, returns DataFrame.

    Args:
        ticker: Stock ticker symbol.
        start_date: Start date 'YYYY-MM-DD'.
        end_date: End date 'YYYY-MM-DD'.

    Returns:
        DataFrame with OHLCV data, or empty DataFrame on error.
    """
    raw_data_dir = os.path.join(os.path.dirname(__file__), "data", "raw")
    os.makedirs(raw_data_dir, exist_ok=True)
    file_path = os.path.join(raw_data_dir, f"{ticker}_{start_date}_{end_date}.csv")

    data = _fetch_polygon(ticker, start_date, end_date)
    if data.empty:
        data = _fetch_finnhub(ticker, start_date, end_date)
    if data.empty:
        try:
            import yfinance as yf
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        except Exception:
            return pd.DataFrame()

    try:
        if data.empty or data is None:
            return pd.DataFrame()

        # Flatten MultiIndex columns (yfinance returns MultiIndex for single ticker)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)

        data.to_csv(file_path)
        return data
    except Exception:
        return pd.DataFrame()


if __name__ == "__main__":
    fetch_stock_data("AAPL", "2022-01-01", "2024-01-01")
