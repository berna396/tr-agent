# tr-agent

A local AI trading agent that runs entirely on your machine. It scans 48 liquid large-caps every morning, selects the best setups, and paper-trades them throughout the day using a 4-stage pipeline: technical signals → risk check → LLM confirmation → order execution. A LightGBM model runs in the background, learning from every trade and improving its confidence scores over time.

**Status:** v0.3 — paper trading live, ML layer active, dynamic screener running.

---

## How it works

Every trading day the agent runs three types of jobs:

```
09:00 ET  ─── ML daily check
              └── If 10+ new closed trades since last training → retrain LightGBM → deploy
09:15 ET  ─── Pre-market screener
              └── Score 48 large-caps → pick top 12 by signal quality, trend, volume, ML confidence
              └── Save to data/active_watchlist.json → Telegram: "Today's watchlist: GOOGL, NVDA, ..."
09:30–16:00 ET  ─── Trading cycle (every 30 min)
              └── For each stock in today's watchlist → run 4-stage pipeline
10:00 ET (Sun)  ─── Weekly ML analysis
              └── 30-day P&L report + SHAP feature analysis → Ollama generates narrative → Telegram
```

### 4-stage trading pipeline

```
Stage 1 — Signal detection  (pure Python, no LLM)
  ├── Fetch 3 months of daily OHLCV (yfinance)
  ├── Compute RSI(14), MACD(12/26/9), SMA(20/50)
  ├── Compute 14 ML features: ATR, Bollinger Bands, ROC, ADX, volume ratio, ...
  ├── Query LightGBM → ml_confidence (0–1, probability of profitable outcome)
  └── Fire BUY/SELL when ≥2 of 3 indicators align · NEUTRAL → stop

Stage 2 — Risk check  (pure Python, no LLM)
  ├── Max 20% of cash per single trade
  ├── Max 60% of portfolio invested at once
  └── SELL requires open position · Rejected → Telegram alert

Stage 3 — LLM confirmation  (Ollama · qwen2.5:7b)
  ├── Receives: signal + all indicators + ML confidence + portfolio state + trade history
  ├── Returns JSON: {confirmed, quantity, reasoning}
  └── Conservative by design — rejects anything ambiguous

Stage 4 — Order execution  (PaperBroker)
  ├── Fill at current market price ± 0.1% slippage
  ├── Persist to data/portfolio_state.json
  ├── Log everything to data/journal.db
  └── Send Telegram notification with P&L summary
```

---

## ML layer

The ML layer runs alongside the pipeline and improves automatically.

**Model:** LightGBM binary classifier — predicts whether the current market conditions will produce a profitable trade.

**14 features per signal:**

| Category | Features |
|---|---|
| Momentum | RSI(14), MACD, MACD histogram, MACD signal, ROC(5d), ROC(10d), ROC(20d) |
| Trend | SMA ratio (SMA20/SMA50), ADX(14) |
| Volatility | ATR(14)/price, Bollinger Band position, Bollinger Band width |
| Volume | Volume / 20-day average volume |
| Time | Day of week |

**Training data:** Bootstrapped from 2 years of historical data (501 labeled samples across all 5 original tickers). As paper trades close, the model retrains on real P&L outcomes — the labels get cleaner and AUC improves over time.

**Deployment:** ML confidence is injected into the LLM prompt as context:
- `> 60%` → "supports trade"
- `40–60%` → neutral
- `< 40%` → "warns against trade"

The LLM makes the final call; ML is one voice in the room, not a veto.

**Model files:** `data/models/signal_model.pkl` (current), `signal_model_vN.pkl` (versioned history).

---

## Pre-market screener

Every morning at 9:15 AM ET, the screener evaluates the full 48-ticker candidate pool and selects the 12 best setups for the day.

**Scoring (0–5.5 per ticker):**
- Signal fires (BUY or SELL): +2.0
- ADX > 25 (trending, not choppy): +1.0
- Volume ratio > 1.2 (above-average interest): +1.0
- ML confidence > 55%: +1.0 / < 40%: −0.5
- RSI extreme (< 30 or > 70): +0.5

**Candidate pool (48 tickers):**

| Sector | Tickers |
|---|---|
| Tech | AAPL MSFT GOOGL AMZN META NVDA TSLA AMD INTC QCOM CRM ADBE NFLX MU AMAT ORCL PYPL UBER |
| Finance | JPM BAC GS MS V MA AXP BLK |
| Health | JNJ UNH LLY ABBV MRK PFE |
| Consumer | WMT HD COST NKE MCD SBUX |
| Energy | XOM CVX COP SLB |
| Industrial | BA CAT HON GE T VZ |

The selected watchlist is saved to `data/active_watchlist.json`. The trading cycle reads from it automatically.

---

## Stack

| Layer | Technology |
|---|---|
| LLM runtime | Ollama (`qwen2.5:7b`, local) |
| ML model | LightGBM + scikit-learn + SHAP |
| Scheduling | APScheduler (cron) |
| Process management | supervisord |
| Market data | yfinance |
| Technical indicators | `ta` library |
| Trade journal | SQLite (`data/journal.db`) |
| Config | pydantic-settings + `.env` |
| CLI | Typer + Rich |
| Notifications | Telegram Bot API |

---

## Setup

### Prerequisites
- Python 3.12+
- [uv](https://github.com/astral-sh/uv)
- [Ollama](https://ollama.ai) running locally

```bash
# Pull the LLM model
ollama pull qwen2.5:7b

# Install dependencies (including ML stack)
uv sync

# Configure
cp .env.example .env
# Edit .env — set TELEGRAM_TOKEN and TELEGRAM_CHAT_ID at minimum
```

### `.env` reference

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b

PAPER_MODE=true
PAPER_INITIAL_CAPITAL=10000.0
PAPER_SLIPPAGE=0.001

TELEGRAM_TOKEN=<your-bot-token>
TELEGRAM_CHAT_ID=<your-chat-id>

# Optional — defaults shown
ML_MIN_NEW_SAMPLES=10
ML_BACKTEST_PERIOD=2y
SCREENER_TOP_N=12
SCREENER_MIN_PRICE=10.0
SCREENER_MIN_AVG_VOLUME=500000
```

### First run

```bash
# 1. Train the initial ML model (downloads 2y of history, ~30s)
uv run python -m tr_agent.main ml bootstrap

# 2. Run the screener to pick today's watchlist
uv run python -m tr_agent.main screen

# 3. Run a single test cycle
uv run python -m tr_agent.main trade

# 4. Start the full scheduler (stays running)
uv run python -m tr_agent.main scheduler
```

### Production (supervisord)

```bash
# Start (launches tr-agent automatically, restarts on crash)
.venv/bin/supervisord -c supervisord.conf

# Control
.venv/bin/supervisorctl -c supervisord.conf status
.venv/bin/supervisorctl -c supervisord.conf restart tr-agent
.venv/bin/supervisorctl -c supervisord.conf tail -f tr-agent

# Stop everything
.venv/bin/supervisorctl -c supervisord.conf shutdown
```

Logs rotate at 10 MB (3 backups) in `data/tr-agent.log`.

---

## CLI reference

```bash
# Trading
uv run python -m tr_agent.main trade               # one cycle, all tickers
uv run python -m tr_agent.main trade --tickers AAPL,GOOGL
uv run python -m tr_agent.main scheduler           # continuous (30 min intervals)
uv run python -m tr_agent.main portfolio           # show current paper portfolio

# Screener
uv run python -m tr_agent.main screen              # run screener, save watchlist
uv run python -m tr_agent.main screen --dry-run    # run screener, don't save
uv run python -m tr_agent.main screen --top-n 20   # pick more tickers

# ML
uv run python -m tr_agent.main ml bootstrap        # train initial model from 2y history
uv run python -m tr_agent.main ml status           # show model version, AUC, training date
uv run python -m tr_agent.main ml analyze          # run Ollama analysis on journal data
```

---

## Data files

All runtime data lives in `data/` (gitignored — back it up):

| File | Contents |
|---|---|
| `journal.db` | SQLite: every signal, risk check, LLM decision, outcome, P&L |
| `portfolio_state.json` | Paper portfolio: cash, positions, trade history |
| `active_watchlist.json` | Today's screener picks (refreshed daily at 9:15 AM ET) |
| `models/signal_model.pkl` | Current LightGBM model |
| `models/signal_model_vN.pkl` | Versioned model history (last 3 kept) |
| `models/training_history.json` | All training runs with AUC and sample counts |
| `tr-agent.log` | Agent logs |

### Journal schema

```sql
-- Every decision step logged
cycle_events(id, ts, ticker, event_type, data_json)
  event_type: "signal" | "risk" | "llm_decision" | "order"
  signal data includes: rsi, macd_hist, sma_20, sma_50, close, reasoning, ml_features{}

-- Closed trades for ML training and memory
trade_outcomes(id, ticker, buy_ts, sell_ts, buy_price, sell_price, quantity, pnl, pnl_pct)
```

---

## Scheduled jobs

| Job | Schedule | What it does |
|---|---|---|
| `Pre-market screener` | Mon–Fri 9:15 AM ET | Score 48 tickers, pick top 12, Telegram |
| `Trading cycle` | Mon–Fri 9:30–16:00 ET (every 30 min) | 4-stage pipeline on active watchlist |
| `ML daily check` | Mon–Fri 9:00 AM ET | Retrain model if ≥10 new live trades |
| `ML weekly analysis` | Sunday 10:00 AM ET | 30-day P&L + SHAP + Ollama insights → Telegram |

---

## Project structure

```
src/tr_agent/
├── main.py              CLI entry point (trade, scheduler, portfolio, screen, ml)
├── scheduler.py         APScheduler orchestration + refresh_watchlist + run_cycle
├── screener.py          Pre-market screener: score and rank candidate pool
├── config.py            Pydantic settings from .env
├── journal.py           SQLite trade journal (read/write all events)
├── memory.py            Build LLM context from historical trade outcomes
├── risk.py              Risk validator (position sizing rules)
├── notifier.py          Telegram notifications
├── signals/
│   ├── technical.py     Technical analysis: fetch OHLCV, compute indicators, derive signal
│   └── rules.py         Rule engine (for backtesting/reference)
├── agent/
│   ├── core.py          LLM confirmation via Ollama
│   └── prompts.py       System prompt + ML confidence line builder
├── broker/
│   ├── paper.py         Paper broker: simulate orders with slippage
│   └── trade_republic.py  Stub for live trading (v0.4)
├── portfolio/
│   ├── tracker.py       In-memory portfolio state
│   └── persistence.py   Load/save portfolio_state.json
└── ml/
    ├── features.py      14-feature engineering from OHLCV DataFrame
    ├── dataset.py       Historical bootstrap + live data from journal
    ├── signal_model.py  LightGBM wrapper (train, predict, save, load)
    ├── trainer.py       Walk-forward CV, deploy gate, training history
    ├── analyzer.py      SHAP importances, performance report, Ollama insights
    └── auto_improve.py  Daily retrain + weekly analysis scheduler hooks
```

---

## Next steps

### v0.4 — Live trading (Trade Republic)
- Wire up `broker/trade_republic.py` using the `pytr` library
- Toggle via `PAPER_MODE=false` in `.env`
- Add fractional share handling and order confirmation flow
- Risk: start with very small position sizes

### v0.5 — Smarter signals
- **Market regime filter**: if SPY/QQQ is in a downtrend (SMA20 < SMA50), suppress BUY signals system-wide
- **Earnings blackout**: avoid entering positions within 3 days of earnings (fetch calendar from yfinance)
- **Sector rotation**: track which sectors have the most buy signals as a confirmation signal
- **Stop-loss tracking**: close positions that drop X% from entry price regardless of sell signal

### v0.6 — Better ML
- The current LightGBM model improves automatically as live trade outcomes accumulate — just keep trading
- Once 200+ real P&L-labeled trades exist, retrain with a longer lookback and tune hyperparameters
- Add inter-ticker correlation features (e.g., NVDA vs. AMD signal alignment)
- Experiment with a small LSTM on sequences of signals (needs ~500+ trades)

### v0.7 — Observability
- Web dashboard (FastAPI + htmx) showing live portfolio, recent signals, model performance
- Backtesting CLI: replay journal signals against historical prices to measure strategy performance
- Alert on model degradation (AUC drops below baseline over rolling 30-day window)
