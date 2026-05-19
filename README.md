# tr-agent

A local AI trading agent that runs entirely on your machine. It scans 48 liquid large-caps every morning, selects the best setups, and paper-trades them throughout the day using a 5-stage pipeline: technical signals → news analysis → risk check → LLM confirmation → order execution. A LightGBM model runs in the background, learning from every trade and improving its confidence scores over time.

**Status:** v0.8 — paper trading live. Multi-horizon labels, Kelly sizing, news analyst LLM, Slack trade alerts, web dashboard.

---

## How it works

Every trading day the agent runs three types of jobs:

```
09:00 ET  ─── ML daily check
              └── If 10+ new closed trades since last training → retrain LightGBM → deploy
09:15 ET  ─── Pre-market screener
              └── Score 48 large-caps → pick top 12 by signal quality, trend, volume, ML confidence
              └── Save to data/active_watchlist.json
09:30–16:00 ET  ─── Trading cycle (every 30 min)
              └── For each stock in today's watchlist → run 5-stage pipeline
10:00 ET (Sun)  ─── Weekly ML analysis
              └── 30-day P&L report + SHAP feature analysis → Ollama generates narrative
```

### Trading pipeline

```
Stage 0 — Stop-loss enforcement  (runs before everything else)
  ├── Check every open position against current price
  ├── ATR-based stop: stop_price = entry − (2 × ATR(14)) stored at BUY time
  ├── Falls back to fixed 5% stop for positions without an ATR stop
  └── Slack SELL alert on trigger

Stage 1 — Signal detection  (pure Python, no LLM)
  ├── Fetch 1 year of daily OHLCV + 5 days of 15-min intraday bars (yfinance)
  ├── RSI(14), MACD(12/26/9), SMA(20/50) computed from intraday bars (live during session)
  ├── SMA(200) and all 20 ML features computed from daily bars
  ├── Query LightGBM → ml_confidence (0–1, probability of profitable outcome)
  └── Fire BUY when ≥2 of {RSI<30, MACD hist>0, SMA20>SMA50} · SELL symmetric · else NEUTRAL

Stage 1b — BUY guards  (applied before news analysis, BUY signals only)
  ├── Market regime filter: SPY SMA50 < SMA200 (death cross) → suppress BUY
  ├── Earnings blackout: earnings within 3 days → suppress BUY · fails closed on API error
  └── News risk gate: Ollama risk_level="high" AND sentiment < −0.5 → suppress BUY

Stage 1.5 — News analysis  (Ollama · qwen2.5:7b)
  ├── Receives: ticker + recent headlines + days until earnings
  ├── Returns structured NewsContext: sentiment_score, risk_level, flags, summary
  └── NewsContext is passed to Stage 3 as structured context (replaces raw headlines)

Stage 2 — Risk check  (pure Python, no LLM)
  ├── Kelly criterion half-Kelly sizing: f* = (p×b − (1−p)) / b, half-Kelly = f*/2
  ├── Falls back to fixed 20% of cash when no historical stats available
  ├── Hard cap: never exceed 20% of cash in a single trade
  └── Max 60% of portfolio invested at any time

Stage 3 — LLM confirmation  (Ollama · qwen2.5:7b)
  ├── Receives: signal + indicators + ML confidence + regime + portfolio state + trade history
  ├── + Structured NewsContext from Stage 1.5 (risk level, flags, summary)
  ├── + Learned rules from data/llm_rules.md (generated weekly by Ollama)
  ├── Returns JSON: {confirmed, quantity, reasoning}
  └── Conservative by design — rejects anything ambiguous

Stage 4 — Order execution  (PaperBroker)
  ├── Fill at current market price ± 0.1% slippage
  ├── Compute ATR-based stop_price and store on position
  ├── Persist to data/portfolio_state.json
  ├── Log everything to data/journal.db
  └── Slack notification: BUY or SELL with % of cash used
```

---

## ML layer

The ML layer runs alongside the pipeline and improves automatically.

**Model:** LightGBM binary classifier — predicts whether the current market conditions will produce a profitable trade.

**20 features per signal:**

| Category | Features |
|---|---|
| Momentum | RSI(14), MACD, MACD histogram, MACD signal, ROC(5d), ROC(10d), ROC(20d) |
| Trend | SMA ratio (SMA20/SMA50), ADX(14) |
| Volatility | ATR(14)/price, Bollinger Band position, Bollinger Band width |
| Volume | Volume / 20-day average volume |
| Time | Day of week |
| Market-relative | `rel_roc_5` (ticker ROC5 − SPY ROC5), `spy_corr_60` (60-day return correlation with SPY) |
| Sentiment | `news_sentiment` (VADER compound score, −1 to 1) |
| Options flow | `iv_rank` (ATM IV vs 52-week range), `put_call_ratio` (put/call volume ratio) |
| Short interest | `short_ratio` (days-to-cover) |

The last 4 features default to 0.0/1.0 in historical bootstrap — live values are injected at trade time and the model learns them over time.

**Labels (multi-horizon ensemble):** A sample is labeled positive only if ≥2 of 3 forward return horizons exceed their threshold:

| Horizon | Threshold |
|---|---|
| 5 days | ≥ 0.5% |
| 10 days | ≥ 0.5% |
| 20 days | ≥ 1.0% |

A whipsaw filter rejects the sample entirely if the 5-day forward drawdown exceeds −3%, even when the net return is positive.

**Training:** Bootstrapped from 2 years of historical data (~485 labeled samples). Hyperparameters tuned via `GridSearchCV + TimeSeriesSplit` on every training run. As paper trades close, the model retrains on real P&L outcomes automatically.

**Deployment gate:** CV AUC must exceed 0.40. Current model: v3, AUC 0.483, 485 samples, 20 features.

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

## Web dashboard

A Flask dashboard runs on port 8080 (accessible on the local network).

**Features:**
- Process status (running / stopped) with uptime, start/stop/restart controls
- Live log streaming via Server-Sent Events
- Portfolio metrics: cash, total value, return %, open positions with live prices
- Portfolio equity curve (Chart.js) — reconstructed from trade log; shows cost-basis history with live unrealized P&L at the current point
- Cycle events feed (signals, risk checks, LLM decisions, orders) from `journal.db`
- Closed trade history with P&L

```bash
# Dashboard URL (replace with your machine's IP)
http://192.168.1.x:8080
```

---

## Notifications

Only actual trade executions produce a notification — no noise from screener runs, risk rejections, or cycle summaries.

**Slack** (buy/sell only):

```
🟢 BUY AAPL
$195.20  ·  8% of cash  (~$800 · 4.1 sh)
Stop: $185.44  (-5.0%)

🔴 SELL NVDA
$245.00  ·  9.0 shares
P&L: +0.18%

🔴 SELL NVDA
$215.00  ·  9.0 shares  ·  stop-loss
P&L: -3.0%
```

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
| Sentiment | vaderSentiment (news headlines → compound score) |
| Trade journal | SQLite (`data/journal.db`) |
| Config | pydantic-settings + `.env` |
| CLI | Typer + Rich |
| Web dashboard | Flask + Chart.js (CDN) |
| Notifications | Slack Incoming Webhooks |

---

## Setup

### Prerequisites
- Python 3.12+
- [uv](https://github.com/astral-sh/uv)
- [Ollama](https://ollama.ai) running locally

```bash
# Pull the LLM model
ollama pull qwen2.5:7b

# Install dependencies
uv sync

# Configure
cp .env.example .env
# Edit .env — set SLACK_WEBHOOK_URL at minimum
```

### `.env` reference

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b

PAPER_MODE=true
PAPER_INITIAL_CAPITAL=10000.0
PAPER_SLIPPAGE=0.001

# Slack — create at api.slack.com → Your Apps → Incoming Webhooks
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL

# Telegram (optional, legacy)
TELEGRAM_TOKEN=
TELEGRAM_CHAT_ID=

# Optional — defaults shown
ML_MIN_NEW_SAMPLES=10
ML_BACKTEST_PERIOD=2y
SCREENER_TOP_N=12
SCREENER_MIN_PRICE=10.0
SCREENER_MIN_AVG_VOLUME=500000

# Safety guards
STOP_LOSS_PCT=0.05              # 5% drawdown triggers auto-close (0 = disabled)
STOP_LOSS_ATR_MULTIPLIER=2.0    # ATR-based stop = entry − (mult × ATR14)
EARNINGS_BLACKOUT_DAYS=3        # suppress BUY within N days of earnings (0 = disabled)
REGIME_FILTER_ENABLED=true      # suppress BUY when SPY SMA50 < SMA200

# v0.8
KELLY_SIZING=true               # half-Kelly position sizing when historical stats available
NEWS_RISK_GATE=true             # block BUY when Ollama rates news risk "high" + sentiment < -0.5
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
# Start both tr-agent and the web dashboard
.venv/bin/supervisord -c supervisord.conf

# Control
./agent.sh status
./agent.sh restart
./agent.sh logs
./agent.sh web-logs

# Stop everything
.venv/bin/supervisorctl -c supervisord.conf shutdown
```

Logs rotate at 10 MB (3 backups) in `data/tr-agent.log` and `data/tr-agent-web.log`.

---

## CLI reference

```bash
# Trading
uv run python -m tr_agent.main trade               # one cycle, all tickers
uv run python -m tr_agent.main trade --tickers AAPL,GOOGL
uv run python -m tr_agent.main scheduler           # continuous (30 min intervals)
uv run python -m tr_agent.main portfolio           # show current paper portfolio
uv run python -m tr_agent.main web                 # start web dashboard only

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
| `portfolio_state.json` | Paper portfolio: cash, positions, full trade history |
| `active_watchlist.json` | Today's screener picks (refreshed daily at 9:15 AM ET) |
| `models/signal_model.pkl` | Current LightGBM model (20 features) |
| `models/signal_model_vN.pkl` | Versioned model history (last 3 kept) |
| `models/training_history.json` | All training runs with AUC and sample counts |
| `llm_rules.md` | Auto-generated weekly by Ollama — learned rules injected into every LLM confirmation |
| `tr-agent.log` | Agent logs |
| `tr-agent-web.log` | Web dashboard logs |

### Journal schema

```sql
-- Every decision step logged
cycle_events(id, ts, ticker, event_type, data_json)
  event_type: "signal" | "risk" | "llm_decision" | "order"
  signal data: rsi, macd_hist, sma_20, sma_50, close, reasoning, ml_features{}

-- Closed trades for ML training
trade_outcomes(id, ticker, buy_ts, sell_ts, buy_price, sell_price, quantity, pnl, pnl_pct)
```

---

## Scheduled jobs

| Job | Schedule | What it does |
|---|---|---|
| `ML daily check` | Mon–Fri 9:00 AM ET | Retrain model if ≥10 new live trades |
| `Pre-market screener` | Mon–Fri 9:15 AM ET | Score 48 tickers, pick top 12, save watchlist |
| `Trading cycle` | Mon–Fri 9:30–16:00 ET (every 30 min) | 5-stage pipeline on active watchlist |
| `ML weekly analysis` | Sunday 10:00 AM ET | 30-day P&L + SHAP + Ollama insights → `llm_rules.md` |

---

## Project structure

```
src/tr_agent/
├── main.py              CLI entry point (trade, scheduler, portfolio, screen, ml, web)
├── scheduler.py         APScheduler orchestration + refresh_watchlist + run_cycle
├── screener.py          Pre-market screener: score and rank candidate pool
├── market_regime.py     SPY SMA50/SMA200 golden/death cross regime detection
├── guards.py            Earnings blackout check + days_until_earnings
├── config.py            Pydantic settings from .env
├── journal.py           SQLite trade journal (read/write all events)
├── memory.py            Build LLM context from historical trade outcomes
├── risk.py              Risk validator (Kelly sizing + portfolio limits)
├── notifier.py          Slack trade notifications + Telegram (legacy)
├── news.py              yfinance headline fetch + VADER sentiment scoring
├── web.py               Flask dashboard (port 8080): metrics, logs, equity chart
├── signals/
│   ├── technical.py     Technical analysis: OHLCV, indicators, signal, 20 ML features
│   └── rules.py         Rule engine (for backtesting/reference)
├── agent/
│   ├── core.py          LLM trade confirmation via Ollama
│   ├── news_analyst.py  Ollama news analysis → structured NewsContext
│   └── prompts.py       Prompt builders (signal, regime, news, ML confidence)
├── broker/
│   ├── paper.py         Paper broker: simulate orders with slippage
│   └── trade_republic.py  Stub for live trading (future)
├── portfolio/
│   ├── tracker.py       In-memory portfolio state + equity reconstruction
│   └── persistence.py   Load/save portfolio_state.json
└── ml/
    ├── features.py      20-feature engineering from OHLCV + SPY DataFrame
    ├── dataset.py       Multi-horizon labels + whipsaw filter + live data from journal
    ├── kelly.py         Half-Kelly position sizing from historical win/loss stats
    ├── signal_model.py  LightGBM wrapper (train, predict, save, load)
    ├── trainer.py       Walk-forward CV, deploy gate, training history
    ├── analyzer.py      SHAP importances, performance report, Ollama insights
    └── auto_improve.py  Daily retrain + weekly analysis scheduler hooks
```

---

## Changelog

### ✅ v0.8 (current)
- **Multi-horizon labels**: ensemble of 5d/10d/20d forward returns (majority vote) + whipsaw filter; more robust than single-horizon
- **20 ML features**: added `news_sentiment` (VADER), `iv_rank`, `put_call_ratio`, `short_ratio`; model is v3 (AUC 0.483)
- **News analyst LLM**: dedicated Ollama call before trade confirmation produces structured `NewsContext`; news risk gate blocks BUY on high-risk negative news
- **Kelly criterion sizing**: half-Kelly position sizing when ≥10 closed trades available; falls back to fixed 20%
- **Slack notifications**: buy/sell only — price, % of cash, stop level; no noise
- **Web dashboard**: port 8080, process control, live SSE logs, real-time portfolio with equity curve chart
- **Watchlist fix**: trading cycle now reads `active_watchlist.json` each run instead of using the hardcoded default list

### ✅ v0.7
- Intraday signals: RSI/MACD/SMA from 15-min bars (falls back to daily outside market hours)
- ATR-based stop-loss stored per position (adapts to each stock's volatility)
- LLM feedback rules: weekly Ollama analysis synthesises performance into `llm_rules.md`
- News injection: recent headlines passed to LLM Stage 3

### ✅ v0.6
- 16 ML features: added `rel_roc_5` (excess return vs SPY) and `spy_corr_60`
- SPY data cached per session; hyperparameter tuning via `GridSearchCV + TimeSeriesSplit`

### ✅ v0.5
- RSI thresholds tightened (BUY < 30, SELL > 70)
- Market regime filter (SPY golden/death cross)
- Earnings blackout guard (fail-closed)
- SMA200 in signal and LLM prompt
- Fixed 5% stop-loss Stage 0

---

## Roadmap

### Live broker
Interactive Brokers is the recommended next step — official Python SDK (`ib_insync`), EU accounts supported, free paper trading account for testing. Implement as `broker/ibkr.py`, toggle via `PAPER_MODE=false`.

### Backtesting CLI
Replay `journal.db` signals against historical prices to measure strategy performance without running live cycles.

### Model degradation alerts
Alert when rolling 30-day AUC drops below the deployment baseline — signals the model needs fresh training data.
