# Candlestick pattern reference

This file is the **source-of-truth description** for candlestick features used in this repo. The implementations live in `feature_engineering.py` (`_add_candlestick_patterns`); definitions here match that code.

## Feature columns (`constants.FEATURE_COLS`)

| Column | Meaning |
|--------|---------|
| `cdl_doji` | 1 if the bar is classified as a doji on that day, else 0 |
| `cdl_hammer` | 1 if hammer, else 0 |
| `cdl_inverted_hammer` | 1 if inverted hammer, else 0 |
| `cdl_bullish_engulfing` | 1 if bullish engulfing vs prior bar, else 0 |
| `cdl_bearish_engulfing` | 1 if bearish engulfing vs prior bar, else 0 |
| `cdl_hammer_5d` | Count of `cdl_hammer` in the last 5 bars (inclusive) |
| `cdl_doji_5d` | Count of `cdl_doji` in the last 5 bars (inclusive) |

If `Open`, `High`, or `Low` are missing (close-only data), all candle columns are set to **0**.

## Definitions (algorithmic)

Notation per bar: open **O**, high **H**, low **L**, close **C**, range **R = H − L** (floored to a tiny epsilon to avoid division by zero). Body size **B = |C − O|**. Upper wick **U = H − max(O, C)**. Lower wick **Lo = min(O, C) − L**.

### Doji

- **Rule:** `B / R < 0.1` (very small body relative to full range).
- **Interpretation:** Indecision; context (trend, volume) matters for trading meaning.

### Hammer

- **Rule:** all of:
  - `B < 0.3 * R` (small body),
  - `Lo >= 2 * max(B, tiny)` (lower wick at least twice the body),
  - `U < 0.3 * R` (small upper wick).
- **Classic idea:** Potential bullish reversal after a decline (not validated here—only geometry).

### Inverted hammer

- **Rule:** all of:
  - `B < 0.3 * R`,
  - `U >= 2 * max(B, tiny)` (long upper wick),
  - `Lo < 0.3 * R` (small lower wick).

### Bullish engulfing

- **Rule:** prior bar bearish (`C_prev < O_prev`), current bar bullish (`C > O`), and current real-body range **fully covers** prior real body: `min(O, C) <= min(O_prev, C_prev)` and `max(O, C) >= max(O_prev, C_prev)`.

### Bearish engulfing

- **Rule:** prior bar bullish, current bar bearish, same full engulfment of prior body.

## Where patterns are used

1. **ML (`pipeline/nodes/technical.py`, `backtest.py`)**  
   Candle columns are part of `FEATURE_COLS`. The classifier is trained to predict **next-day direction** (whether the next close is above today’s close). The model does **not** optimize dollar PnL or Sharpe directly; it learns associations between features (including candles) and that label.

2. **Pattern LLM agent (`pipeline/nodes/pattern.py`)**  
   Sends the last ~20 OHLCV rows plus latest-bar flags to the model described in `pipeline/prompts/templates.py` (`PATTERN_SYSTEM` / `PATTERN_PROMPT`).

3. **Decision agent (`pipeline/nodes/decision.py`)**  
   Combines technical (ML + technical LLM), pattern LLM output, and institutional signals into BUY / SELL / HOLD.

## Trading model and “maximizing profit”

**What the stack actually does**

- **Supervised ML:** Random Forest or XGBoost (see `constants.get_model`) on `FEATURE_COLS`, walk-forward evaluated in `backtest.py`.
- **Objective:** Binary label = next day up vs down. Training uses optional **crisis downweighting** (`get_crisis_weights`) so the fit is less dominated by crash windows.
- **Signals:** Confidence comes from `predict_proba`; trades are gated by a **confidence threshold** (default `0.53` in code paths; backtest: `--threshold`).

**What it does *not* do**

- It is **not** reinforcement learning and **does not** maximize cumulative return or Sharpe during training.
- Candlestick flags are **heuristic geometry**, not broker-grade pattern recognition (no trend qualification, volume confirmation, or multi-bar strict Nison rules).

**How to steer toward better *economic* outcomes (honest workflow)**

1. **Tune the trade filter** — Run `python backtest.py --threshold 0.52` (or 0.54–0.60) and compare `total_return_pct`, `avg_return_per_trade_pct`, and number of trades. Higher threshold usually means fewer trades and often higher average quality, but not always.
2. **Validate out-of-sample** — Keep walk-forward windows honest; if performance jumps only in-sample, treat it as overfit.
3. **Paper trade** — `paper_trader.py` logs forward outcomes; use `python paper_trader.py review` to see realized edge.
4. **Confluence** — The decision step is designed so **technical + pattern + smart money** agree before high conviction. Dry-run / fallback ranking nudges confidence when pattern **trend** agrees with the ML side (see `decision.py`).

## Maintenance

When you change thresholds or add patterns in `feature_engineering.py`, update this document so comments in `constants.py` / `feature_engineering.py` remain accurate.
