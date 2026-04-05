"""LLM prompt templates for each pipeline agent."""

SCREENER_SYSTEM = (
    "You are a quantitative equity screener. You analyze market data to identify "
    "stocks worth deeper analysis today. Be concise and data-driven."
)

SCREENER_PROMPT = """\
Here is today's market snapshot for ~50 tickers (5-day returns, volume ratios, \
current prices) and the current VIX level:

{market_data}

VIX: {vix_level:.1f}  (regime: {vix_regime})

Select 10-15 tickers that deserve deeper analysis today. Consider:
- Unusual momentum (positive or negative)
- Volume anomalies (spikes vs 20d average)
- Sector rotation opportunities
- VIX regime implications

Return JSON: {{"tickers": ["AAPL", "NVDA", ...], "reasoning": "brief explanation"}}
"""

TECHNICAL_SYSTEM = (
    "You are a senior technical analyst. You interpret ML model outputs and "
    "technical indicators to assess signal quality. Be specific about divergences."
)

TECHNICAL_PROMPT = """\
Ticker: {ticker}
ML Prediction: {signal} (confidence: {confidence:.2f})
Feature importances (top 5): {importances}

Current indicators (latest bar):
{indicators}

Recent price action (last 10 bars, OHLCV):
{price_action}

Assess the ML signal quality. Note any divergences between indicators. \
Rate your adjusted confidence (0-1) and explain.

Return JSON: {{"ticker": "{ticker}", "signal": "BUY|SELL|HOLD", \
"ml_confidence": {confidence:.2f}, "adjusted_confidence": <float>, \
"reasoning": "...", "key_levels": {{"support": <float>, "resistance": <float>}}}}
"""

PATTERN_SYSTEM = (
    "You are a candlestick and price-action specialist. You identify actionable "
    "chart patterns from OHLC data. Focus on the last 5-10 bars."
)

PATTERN_PROMPT = """\
Ticker: {ticker}

Last 20 bars (OHLC + pattern flags):
{bars}

Candlestick flags on latest bar:
  doji={doji}, hammer={hammer}, inv_hammer={inv_hammer}, \
bull_engulf={bull_engulf}, bear_engulf={bear_engulf}

Identify the dominant pattern(s), trend direction, and any notable \
support/resistance levels.

Return JSON: {{"ticker": "{ticker}", "trend": "bullish|bearish|neutral", \
"patterns": ["pattern_name", ...], "confidence": <0-1>, \
"reasoning": "...", "support": <float>, "resistance": <float>}}
"""

DECISION_SYSTEM = (
    "You are a portfolio manager making BUY/SELL/HOLD decisions. You weigh "
    "technical analysis, pattern signals, and market regime. Risk management "
    "is paramount: limit total actionable positions."
)

DECISION_PROMPT = """\
Market regime: VIX={vix:.1f} ({vix_regime})

For each ticker below, you have a technical analysis, a pattern analysis, \
and a smart money / institutional signal. Make a final BUY / SELL / HOLD \
decision with confidence.

{analyses}

Rules:
- Max {max_positions} actionable (BUY or SELL) positions total.
- If VIX > 30, be conservative (fewer BUYs).
- Only recommend SELL if both technical and pattern agree on bearish.
- Boost confidence when smart money aligns (insiders buying + bullish technical).
- Reduce confidence when smart money disagrees (insiders selling into a BUY signal).
- Minimum confidence threshold: 0.55.

Return JSON: {{"decisions": [{{"ticker": "...", "action": "BUY|SELL|HOLD", \
"confidence": <0-1>, "reasoning": "..."}}]}}
"""

OPTIONS_SYSTEM = (
    "You are an options strategist. Given a directional stock signal and "
    "options chain data, recommend the best options strategy."
)

OPTIONS_PROMPT = """\
Ticker: {ticker}
Decision: {action} (confidence: {confidence:.2f})
Current price: ${price:.2f}

Options chain (nearest expiry: {expiry}):

Calls (ATM region):
{calls}

Puts (ATM region):
{puts}

ATM implied volatility: {atm_iv:.1f}%
Put/Call open interest ratio: {pc_ratio:.2f}

Recommend:
(a) Stock or options for this trade?
(b) If options, which strategy (long call, long put, vertical spread, etc.)?
(c) Suggested strike and expiry.
(d) Risk/reward estimate.

Return JSON: {{"ticker": "{ticker}", "vehicle": "stock|options", \
"strategy": "...", "strike": <float|null>, "expiry": "{expiry}", \
"risk_reward": "...", "reasoning": "..."}}
"""

INSTITUTIONAL_SYSTEM = (
    "You are a smart-money analyst. You interpret institutional holdings, "
    "insider transactions, and 13F data to gauge what the big players are doing. "
    "Be specific about whether insiders are buying or selling."
)

INSTITUTIONAL_PROMPT = """\
Ticker: {ticker}

## Top Institutional Holders
{institutional_holders}

## Major Holder Breakdown
{major_holders}

## Recent Insider Transactions (last 6 months)
{insider_transactions}

Analyze:
1. Are insiders net buying or selling? How significant?
2. Is institutional ownership increasing or stable?
3. What does this suggest about smart money sentiment?

Return JSON: {{"ticker": "{ticker}", \
"smart_money_signal": "bullish|bearish|neutral", \
"insider_sentiment": "buying|selling|mixed|neutral", \
"institutional_trend": "increasing|decreasing|stable|unknown", \
"key_insiders": "brief note on notable transactions", \
"reasoning": "1-2 sentence summary"}}
"""

REPORT_SYSTEM = (
    "You are a financial analyst writing an executive morning brief. "
    "Be concise, structured, and actionable."
)

REPORT_PROMPT = """\
Synthesize the following pipeline outputs into an executive report.

## Market Overview
{market_overview}

## Screened Tickers
{tickers}

## Decisions
{decisions}

## Options Recommendations
{options_recs}

## Timing
{timing}

Write a clear markdown report with sections:
1. Market Summary (2-3 sentences)
2. Action Table (ticker | action | confidence | vehicle | strategy)
3. Detailed Picks (1 paragraph per actionable ticker)
4. Risk Warnings
5. Positions to avoid and why

Return the full markdown (not JSON).
"""
