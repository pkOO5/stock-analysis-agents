# Stock Analysis Agents

A multi-agent stock analysis pipeline that screens the market, analyzes technicals and patterns, makes buy/sell decisions, evaluates options strategies, and generates an executive report — all running locally on your Mac with no cloud API keys.

## Architecture

```
Step 1: SCREENER        → Scans ~50 tickers, LLM picks 10-15 for deep analysis
Step 2: DATA COLLECTOR  → Fetches 3 years of OHLCV data + computes 30 technical features
Step 3: ANALYSIS        → Two agents per ticker (parallel):
         ├─ Technical Agent  (ML model + LLM reasoning)
         └─ Pattern Agent    (candlestick analysis + LLM reasoning)
Step 4: DECISION        → LLM synthesizes all signals into BUY / SELL / HOLD
Step 5: OPTIONS         → Fetches live options chains, LLM recommends strategy
Step 6: REPORT          → Generates executive markdown brief
```

**Stack:** LangGraph (orchestration) · Ollama + llama3.1:8b (local LLM) · scikit-learn / XGBoost (ML) · yfinance (market data + options chains)

## Quick Start

### Prerequisites

- **Python 3.11+**
- **Ollama** — install from [ollama.com](https://ollama.com), then pull the model:

```bash
ollama pull llama3.1:8b
```

### Setup

```bash
git clone https://github.com/pkOO5/stock-analysis-agents.git
cd stock-analysis-agents

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Optional: copy and fill in API keys for Finnhub/Polygon (yfinance works without keys)
cp .env.example .env
```

### Run

```bash
# Make sure Ollama is running (the script auto-starts it on macOS)
./run_pipeline.sh

# Or directly:
python -m pipeline.run_pipeline
```

The report is written to `data/pipeline_report_YYYY-MM-DD.md`.

### Scheduled runs (auto-pilot)

Run the pipeline every N minutes during market hours, Mon-Fri:

```bash
# Run every 60 minutes during 8:30 AM - 3:00 PM (your local time)
python scheduler.py --interval 60

# Every 30 minutes
python scheduler.py --interval 30

# Every 10 minutes (aggressive — each run takes ~6 min)
python scheduler.py --interval 10

# Custom window
python scheduler.py --interval 30 --start 09:00 --end 14:00

# Single run, then exit
python scheduler.py --once
```

To auto-start every weekday morning via macOS launchd:

```bash
./setup_schedule.sh            # install (Mon-Fri at 8:25 AM, hourly until 3 PM)
./setup_schedule.sh uninstall  # remove
```

Ctrl+C stops the scheduler gracefully after the current run finishes.

## Project Structure

```
stock-analysis-agents/
├── pipeline/
│   ├── graph.py              # LangGraph StateGraph orchestrator
│   ├── state.py              # Shared pipeline state (TypedDict)
│   ├── utils.py              # Ollama wrapper, timing, config
│   ├── run_pipeline.py       # CLI entry point
│   ├── nodes/
│   │   ├── screener.py       # Step 1: market scan + LLM selection
│   │   ├── data_collector.py # Step 2: parallel data fetch + features
│   │   ├── technical.py      # Step 3a: ML + LLM technical analysis
│   │   ├── pattern.py        # Step 3b: candlestick + LLM pattern analysis
│   │   ├── decision.py       # Step 4: BUY/SELL/HOLD decisions
│   │   ├── options.py        # Step 5: options chain + strategy
│   │   └── report.py         # Step 6: executive report
│   └── prompts/
│       └── templates.py      # LLM prompt templates
├── market_fetcher.py         # OHLCV data (Polygon → Finnhub → yfinance)
├── feature_engineering.py    # RSI, MACD, Bollinger, candlestick patterns
├── constants.py              # Feature columns, model factory
├── backtest.py               # Walk-forward ML backtest
├── simulate_portfolio.py     # Dollar-amount portfolio simulation
├── paper_trader.py           # Paper trade logger + outcome reviewer
├── scheduler.py              # Recurring runs during market hours
├── setup_schedule.sh         # macOS launchd auto-start installer
├── config.yaml               # Pipeline and model configuration
├── run_pipeline.sh           # Shell wrapper with auto-start and timeout
└── requirements.txt
```

## Configuration

Edit `config.yaml` to adjust:

- `pipeline.model` — Ollama model name (default: `llama3.1:8b`)
- `pipeline.max_tickers` — how many tickers to analyze (default: 15)
- `pipeline.max_positions` — max actionable BUY/SELL calls (default: 5)
- `pipeline.timeout_minutes` — hard time limit (default: 15)

## Backtesting

Validate the ML model's edge before trusting the pipeline's decisions:

```bash
# Backtest all tickers in config.yaml (walk-forward, 252-day training window)
python backtest.py

# Specific tickers
python backtest.py --tickers AAPL NVDA TSLA

# Override model
python backtest.py --model xgboost --threshold 0.55
```

Outputs per-ticker win rate, total return, avg return per trade, best/worst trades.

## Paper Trading

Every pipeline run automatically logs actionable decisions (with entry prices) to `data/paper_trades.jsonl`. Review outcomes after 1, 5, or 10 trading days:

```bash
# Check how past picks performed
python paper_trader.py review

# Check only 5-day outcomes
python paper_trader.py review --days 5
```

This builds a real track record over time — no backtesting bias, just forward results.

## Performance

On an Apple M3 (16 GB RAM) with `llama3.1:8b`:

| Step | ~Time |
|------|-------|
| Screener | 25-30s |
| Data Collector | 1-3s |
| Technical Analysis | 60-90s |
| Pattern Analysis | 60-90s |
| Decision | 30-40s |
| Options Strategy | 45-60s |
| Report | 40-50s |
| **Total** | **~5-7 min** |

## License

MIT
