# tr-agent

A local AI trading agent that runs entirely on your machine. It scans 48 liquid large-caps every morning, selects the best setups, and paper-trades them throughout the day using a 4-stage pipeline: technical signals → risk check → LLM confirmation → order execution. A LightGBM model runs in the background, learning from every trade and improving its confidence scores over time.

**Status:** v0.6 — paper trading live, ML layer active (16 features, hyperparameter tuning, SPY correlation), regime filter + earnings blackout + stop-loss active.

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

### Trading pipeline

```
Stage 0 — Stop-loss enforcement  (runs before everything else)
  ├── Check every open position against current price
  ├── If loss ≥ stop_loss_pct (default 5%) → sell immediately, bypass pipeline
  └── Telegram alert · Skip that ticker for the rest of the cycle

Stage 1 — Signal detection  (pure Python, no LLM)
  ├── Fetch 1 year of daily OHLCV (yfinance)
  ├── Compute RSI(14), MACD(12/26/9), SMA(20/50/200)
  ├── Compute 16 ML features: ATR, Bollinger Bands, ROC, ADX, volume ratio, SPY correlation, ...
  ├── Query LightGBM → ml_confidence (0–1, probability of profitable outcome)
  └── Fire BUY when ≥2 of {RSI<30, MACD hist>0, SMA20>SMA50} · SELL symmetric · else NEUTRAL

Stage 1b — BUY guards  (applied before risk check, BUY signals only)
  ├── Market regime filter: SPY SMA50 < SMA200 (death cross) → suppress BUY
  └── Earnings blackout: earnings within 3 days → suppress BUY · fails closed on API error

Stage 2 — Risk check  (pure Python, no LLM)
  ├── Max 20% of cash per single trade
  ├── Max 60% of portfolio invested at once
  └── SELL requires open position · Rejected → Telegram alert

Stage 3 — LLM confirmation  (Ollama · qwen2.5:7b)
  ├── Receives: signal + indicators (incl. SMA200) + ML confidence + market regime + portfolio state + trade history
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

**16 features per signal:**

| Category | Features |
|---|---|
| Momentum | RSI(14), MACD, MACD histogram, MACD signal, ROC(5d), ROC(10d), ROC(20d) |
| Trend | SMA ratio (SMA20/SMA50), ADX(14) |
| Volatility | ATR(14)/price, Bollinger Band position, Bollinger Band width |
| Volume | Volume / 20-day average volume |
| Time | Day of week |
| Market-relative | `rel_roc_5` (ticker ROC5 − SPY ROC5), `spy_corr_60` (60-day return correlation with SPY) |

**Training:** Bootstrapped from 2 years of historical data (~480 labeled samples). Hyperparameters tuned via `GridSearchCV + TimeSeriesSplit` on every training run. As paper trades close, the model retrains on real P&L outcomes automatically.

**Deployment gate:** CV AUC must exceed 0.40 (model must show some discrimination vs random). Current model: v2, AUC 0.453, tuned params `n_estimators=100, learning_rate=0.01`.

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

# v0.5 safety guards
STOP_LOSS_PCT=0.05          # 5% drawdown triggers auto-close (set to 0 to disable)
EARNINGS_BLACKOUT_DAYS=3    # suppress BUY within N days of earnings (set to 0 to disable)
REGIME_FILTER_ENABLED=true  # suppress BUY when SPY SMA50 < SMA200 (death cross)
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
├── market_regime.py     SPY SMA50/SMA200 golden/death cross regime detection
├── guards.py            Earnings blackout check (fail-closed on API error)
├── config.py            Pydantic settings from .env
├── journal.py           SQLite trade journal (read/write all events)
├── memory.py            Build LLM context from historical trade outcomes
├── risk.py              Risk validator (position sizing rules)
├── notifier.py          Telegram notifications
├── signals/
│   ├── technical.py     Technical analysis: fetch OHLCV, compute indicators, derive signal
│   └── rules.py         Rule engine (for backtesting/reference)
├── agent/
│   ├── core.py          LLM confirmation via Ollama (includes regime context)
│   └── prompts.py       System prompt + ML confidence + regime line builders
├── broker/
│   ├── paper.py         Paper broker: simulate orders with slippage
│   └── trade_republic.py  Stub for live trading (v0.4)
├── portfolio/
│   ├── tracker.py       In-memory portfolio state
│   └── persistence.py   Load/save portfolio_state.json
└── ml/
    ├── features.py      16-feature engineering from OHLCV + SPY DataFrame
    ├── dataset.py       Historical bootstrap + live data from journal
    ├── signal_model.py  LightGBM wrapper (train, predict, save, load)
    ├── trainer.py       Walk-forward CV, deploy gate, training history
    ├── analyzer.py      SHAP importances, performance report, Ollama insights
    └── auto_improve.py  Daily retrain + weekly analysis scheduler hooks
```

---

## Roadmap

### ✅ v0.5 — Smarter signals (done)
- RSI thresholds tightened: BUY < 30 (was 35), SELL > 70 (was 65)
- Market regime filter: SPY SMA50 < SMA200 (golden/death cross, upgraded from SMA20/SMA50)
- Earnings blackout: avoids BUY within 3 days of earnings; fails closed on API error
- SMA200 added to signal analysis and injected into LLM prompt
- Stop-loss: Stage 0 auto-closes positions that drop ≥5% from entry
- Screener: returns empty watchlist on quiet days instead of picking neutral noise

### ✅ v0.6 — Better ML (done)
- 16-feature vector: added `rel_roc_5` (excess return vs SPY) and `spy_corr_60` (60-day correlation with SPY)
- SPY data downloaded once per day and shared across all ticker feature computations (cached)
- Hyperparameter tuning: `GridSearchCV + TimeSeriesSplit` runs on every training call; best params stored in training history
- Label consistency fix: `_is_buy_signal()` threshold aligned with live signal (RSI < 30)
- Walk-forward CV now reports precision and recall alongside AUC
- Model v2: AUC 0.453, tuned params `n_estimators=100, learning_rate=0.01`

### v0.4 — Live trading (Trade Republic)
- Wire up `broker/trade_republic.py` using the `pytr` library
- Toggle via `PAPER_MODE=false` in `.env`
- Add fractional share handling and order confirmation flow
- Risk: start with very small position sizes

### v0.7 — Observability
- Web dashboard (FastAPI + htmx) showing live portfolio, recent signals, model performance
- Backtesting CLI: replay journal signals against historical prices to measure strategy performance
- Alert on model degradation (AUC drops below baseline over rolling 30-day window)

### v0.8 — Advanced ML (needs 200+ closed trades)
- Retrain with longer lookback and tuned hyperparameters on real P&L-labeled data
- LSTM on sequences of signals (needs ~500+ trades)
- RSI entry quality analysis: did buys at RSI < 30 outperform 30–35 entries?
