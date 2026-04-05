"""
Add technical indicators (RSI, EMA, MACD) to OHLCV data.
Expects columns: Open, High, Low, Close, Volume.
Pattern features: day_of_week, month, vix_regime, is_fomc_day (from pattern_research).
"""
import os
import pandas as pd
import numpy as np


def _load_fomc_dates() -> set:
    """Load FOMC dates from data/economic_events.yaml. Returns set of date strings YYYY-MM-DD."""
    path = os.path.join(os.path.dirname(__file__), "data", "economic_events.yaml")
    if os.path.exists(path):
        try:
            import yaml
            with open(path) as f:
                cfg = yaml.safe_load(f) or {}
            return set(cfg.get("fomc", []))
        except Exception:
            pass
    # Fallback: built-in 2022-2024
    dates = [
        "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31",
        "2024-09-18", "2024-11-07", "2024-12-18",
        "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26",
        "2023-09-20", "2023-11-01", "2023-12-13",
        "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15", "2022-07-27",
        "2022-09-21", "2022-11-02", "2022-12-14",
    ]
    return set(dates)


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range - volatility measure using High, Low, Close."""
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _bollinger(close: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands: middle (SMA), upper, lower, bandwidth."""
    middle = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    bandwidth = (upper - lower) / middle
    return pd.DataFrame({"bb_upper": upper, "bb_lower": lower, "bb_middle": middle, "bb_bandwidth": bandwidth})


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "macd_signal": signal_line, "macd_hist": histogram})


def _add_candlestick_patterns(df: pd.DataFrame) -> None:
    """
    Add candlestick pattern features (0/1) from OHLC. No external deps.
    Based on CANDLESTICK_PATTERNS.md: hammer, inverted_hammer, doji,
    bullish_engulfing, bearish_engulfing. In-place.
    Skips if Open/High/Low missing (e.g. Close-only); fills with 0.
    """
    req = ["Open", "High", "Low", "Close"]
    if not all(c in df.columns for c in req):
        n = len(df)
        for col in ["cdl_doji", "cdl_hammer", "cdl_inverted_hammer",
                    "cdl_bullish_engulfing", "cdl_bearish_engulfing",
                    "cdl_hammer_5d", "cdl_doji_5d"]:
            df[col] = 0
        return
    o = df["Open"].values
    h = df["High"].values
    l_ = df["Low"].values
    c = df["Close"].values
    n = len(df)
    body = np.abs(c - o)
    range_ = h - l_
    range_ = np.where(range_ < 1e-10, 1e-10, range_)
    upper_wick = h - np.maximum(o, c)
    lower_wick = np.minimum(o, c) - l_

    # Doji: body very small relative to range (body/range < 0.1)
    df["cdl_doji"] = (body / range_ < 0.1).astype(int)

    # Hammer: small body at top, long lower wick (>= 2x body), little upper wick
    small_body = body < 0.3 * range_
    long_lower = lower_wick >= 2 * np.maximum(body, 1e-10)
    little_upper = upper_wick < 0.3 * range_
    df["cdl_hammer"] = (small_body & long_lower & little_upper).astype(int)

    # Inverted Hammer: small body at bottom, long upper wick, little lower wick
    little_lower = lower_wick < 0.3 * range_
    long_upper = upper_wick >= 2 * np.maximum(body, 1e-10)
    df["cdl_inverted_hammer"] = (small_body & long_upper & little_lower).astype(int)

    # Bullish/Bearish Engulfing (current bar engulfs prior)
    bull_engulf = np.zeros(n, dtype=int)
    bear_engulf = np.zeros(n, dtype=int)
    for i in range(1, n):
        prev_open, prev_close = o[i - 1], c[i - 1]
        curr_open, curr_close = o[i], c[i]
        prev_bull = prev_close > prev_open
        curr_bull = curr_close > curr_open
        curr_lo, curr_hi = min(curr_open, curr_close), max(curr_open, curr_close)
        prev_lo, prev_hi = min(prev_open, prev_close), max(prev_open, prev_close)
        curr_engulfs = curr_lo <= prev_lo and curr_hi >= prev_hi
        if curr_bull and not prev_bull and curr_engulfs:
            bull_engulf[i] = 1
        if not curr_bull and prev_bull and curr_lo <= prev_lo and curr_hi >= prev_hi:
            bear_engulf[i] = 1
    df["cdl_bullish_engulfing"] = bull_engulf
    df["cdl_bearish_engulfing"] = bear_engulf

    # Rolling: count of each pattern in last 5 bars (captures recent pattern density)
    df["cdl_hammer_5d"] = df["cdl_hammer"].rolling(5, min_periods=1).sum()
    df["cdl_doji_5d"] = df["cdl_doji"].rolling(5, min_periods=1).sum()


def add_features(df: pd.DataFrame, vix: pd.Series | None = None) -> pd.DataFrame:
    """Add RSI, EMA, MACD, ATR, Bollinger Bands, volatility, volume, prior_return, and optionally VIX."""
    df = df.copy()
    close = df["Close"]
    high = df["High"] if "High" in df.columns else close
    low = df["Low"] if "Low" in df.columns else close
    df["RSI"] = _rsi(close)
    df["EMA_20"] = _ema(close, 20)
    df["EMA_50"] = _ema(close, 50)
    macd_df = _macd(close)
    df["MACD"] = macd_df["macd"]
    df["MACD_signal"] = macd_df["macd_signal"]
    df["MACD_hist"] = macd_df["macd_hist"]
    df["ATR"] = _atr(high, low, close)
    bb_df = _bollinger(close)
    df["bb_bandwidth"] = bb_df["bb_bandwidth"]
    df["bb_position"] = (close - bb_df["bb_lower"]) / (bb_df["bb_upper"] - bb_df["bb_lower"] + 1e-10)
    df["volatility"] = close.rolling(20).std()
    # Prior 1-day return (momentum). Cap at ±15% to reduce impact of extreme days (COVID, flash crashes)
    raw_ret = (close - close.shift(1)) / (close.shift(1) + 1e-10)
    df["prior_return"] = raw_ret.clip(-0.15, 0.15)
    # Volume: ratio vs 20d average (high volume often confirms moves)
    if "Volume" in df.columns:
        vol_ma = df["Volume"].rolling(20).mean().replace(0, np.nan)
        df["volume_ma_ratio"] = df["Volume"] / vol_ma
        df["volume_ma_ratio"] = df["volume_ma_ratio"].fillna(1.0).replace([np.inf, -np.inf], 1.0)
    else:
        df["volume_ma_ratio"] = 1.0
    # VIX: market stress proxy; high VIX often precedes mean reversion
    if vix is not None and not vix.empty:
        vix_aligned = vix.reindex(df.index).ffill().bfill()
        df["VIX"] = vix_aligned.fillna(20.0)
        raw_vix_chg = (df["VIX"] - df["VIX"].shift(1)) / (df["VIX"].shift(1) + 1e-10)
        df["vix_change"] = raw_vix_chg.clip(-0.5, 0.5)  # Cap extreme VIX spikes (COVID 20→80)
    else:
        df["VIX"] = 20.0
        df["vix_change"] = 0.0

    # Pattern features (from pattern_research.py findings)
    # day_of_week: 0=Mon ... 4=Fri. Mon best avg return, Thu worst
    df["day_of_week"] = np.clip(np.array(df.index.dayofweek), 0, 4)
    # month: 1-12. Nov best, Apr worst
    df["month"] = df.index.month
    # vix_regime: 0=low(<20), 1=mid(20-30), 2=high(>30). Low VIX = better for longs
    v = df["VIX"].values
    df["vix_regime"] = np.where(v < 20, 0, np.where(v <= 30, 1, 2))
    # is_fomc_day: 1 if FOMC announcement day, else 0. Sector effect handled per-ticker
    fomc_dates = _load_fomc_dates()
    df["is_fomc_day"] = df.index.strftime("%Y-%m-%d").isin(fomc_dates).astype(int)

    # Candlestick patterns (from CANDLESTICK_PATTERNS.md; no external deps)
    _add_candlestick_patterns(df)

    return df


def load_raw_csv(path: str) -> pd.DataFrame:
    """Load CSV from market_fetcher output. Handles multi-level headers."""
    data = pd.read_csv(path, index_col=0, parse_dates=True)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(0)
    return data
