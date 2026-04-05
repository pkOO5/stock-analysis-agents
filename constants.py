"""Shared feature columns and model factory."""
# Technical features
FEATURE_COLS = [
    "RSI", "EMA_20", "EMA_50", "MACD", "MACD_signal", "MACD_hist",
    "ATR", "bb_bandwidth", "bb_position", "volatility", "prior_return",
    "volume_ma_ratio", "VIX", "vix_change",
    # Pattern features (from pattern_research.py)
    "day_of_week",   # 0=Mon..4=Fri; Mon best, Thu worst
    "month",         # 1-12; Nov best, Apr worst
    "vix_regime",    # 0=low(<20), 1=mid(20-30), 2=high(>30)
    "is_fomc_day",   # 1=FOMC day, 0=not
    # Candlestick patterns (from CANDLESTICK_PATTERNS.md)
    "cdl_doji", "cdl_hammer", "cdl_inverted_hammer",
    "cdl_bullish_engulfing", "cdl_bearish_engulfing",
    "cdl_hammer_5d", "cdl_doji_5d",
]

# Pattern identifiers: ticker -> sector (for FOMC day behavior)
# Tech/defensive: tend positive on FOMC. Energy/financial: tend negative.
SECTOR_MAP = {
    "tech": ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMD", "INTC", "QQQ"],
    "energy": ["XOM", "CVX", "COP", "OXY"],
    "financial": ["JPM", "BAC", "GS"],
    "defensive": ["COST", "HD", "WMT", "JNJ", "UNH"],
    "industrial": ["BA", "CAT", "HON"],
    "consumer": ["NKE", "MCD", "F", "AAL"],
    "healthcare": ["PFE", "ABBV"],
}
# Tickers that tend to outperform on FOMC days (pattern_research)
FOMC_FAVORED = {"NVDA", "AAPL", "BA", "COST", "HD", "INTC"}


# Crisis windows: downweight these dates (e.g. 0.3) to reduce overfitting to extremes
CRISIS_WINDOWS = [
    ("2020-02-20", "2020-03-23"),  # COVID crash
    ("2018-10-01", "2018-12-24"),  # Q4 2018 selloff
]


def get_crisis_weights(dates: "pd.DatetimeIndex", downweight: float = 0.3) -> "np.ndarray":
    """Return sample weights: 1.0 normally, downweight in crisis windows."""
    import numpy as np
    weights = np.ones(len(dates))
    for start, end in CRISIS_WINDOWS:
        mask = (dates >= start) & (dates <= end)
        weights[mask] = downweight
    return weights


def get_model(config: dict):
    """Return configured classifier. RF regularized to reduce overfitting to crises."""
    model_name = (config or {}).get("model", "random_forest")
    if model_name == "xgboost":
        try:
            from xgboost import XGBClassifier
            return XGBClassifier(n_estimators=100, max_depth=5, random_state=42)
        except ImportError:
            pass
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators=100, max_depth=6, min_samples_leaf=15, random_state=42
    )
