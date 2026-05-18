# tr-agent

Local AI trading agent using Ollama + LangChain. Analyzes a watchlist of stocks every 30 minutes during NYSE market hours, applies technical analysis, and executes paper trades when signals are strong enough.

## Architecture

```
APScheduler (every 30 min, Mon-Fri 9:30–16:00 ET)
    │
    ▼
Stage 1 — Signal detection (pure Python, no LLM)
    RSI + MACD + SMA → BUY / SELL / NEUTRAL
    │
    ├── NEUTRAL → skip
    │
    ▼
Stage 2 — Risk check (pure Python, no LLM)
    Max 20% cash per trade · Max 60% portfolio invested
    │
    ├── rejected → Telegram alert
    │
    ▼
Stage 3 — LLM confirmation (Ollama · qwen2.5:7b)
    Receives signal + portfolio + historical memory
    Outputs JSON: {confirmed, quantity, reasoning}
    │
    ├── rejected → skip
    │
    ▼
Stage 4 — Execute order (PaperBroker)
    Fill at market price + slippage
    Save to portfolio_state.json
    Log to journal.db
    │
    ▼
Telegram notification
```

## Stack

| Layer | Tech |
|---|---|
| LLM runtime | Ollama (`qwen2.5:7b`) |
| Scheduling | APScheduler |
| Process management | supervisord (in `.venv`) |
| Market data | yfinance |
| Technical indicators | `ta` library |
| Trade journal | SQLite (`data/journal.db`) |
| Config | pydantic-settings + `.env` |
| CLI | Typer + Rich |

## Setup

```bash
# 1. Pull the LLM model
ollama pull qwen2.5:7b

# 2. Install dependencies
uv sync

# 3. Configure environment
cp .env.example .env
# Edit .env — set TELEGRAM_TOKEN and TELEGRAM_CHAT_ID

# 4. Run a one-shot test (single cycle, no scheduler)
uv run python -m tr_agent.main trade
```

## Running continuously with supervisord

```bash
# Start (auto-restarts on crash)
uv run supervisord -c supervisord.conf

# Control
uv run supervisorctl -c supervisord.conf status
uv run supervisorctl -c supervisord.conf stop tr-agent
uv run supervisorctl -c supervisord.conf start tr-agent
uv run supervisorctl -c supervisord.conf restart tr-agent
uv run supervisorctl -c supervisord.conf tail -f tr-agent

# Stop everything
uv run supervisorctl -c supervisord.conf shutdown
```

Logs are written to `data/tr-agent.log` and rotate automatically.

## CLI commands

```bash
# Run one cycle immediately
uv run python -m tr_agent.main trade

# Start the scheduler
uv run python -m tr_agent.main scheduler

# Show current portfolio state
uv run python -m tr_agent.main portfolio
```

## Data

All runtime data lives in `data/` (gitignored — back it up):

| File | Contents |
|---|---|
| `portfolio_state.json` | Paper portfolio: cash, positions, trade history |
| `journal.db` | SQLite: every signal, risk check, LLM decision and outcome |
| `tr-agent.log` | Agent logs from supervisord |

## Default watchlist

`AAPL · MSFT · META · TSLA · NVDA`

Chosen for balanced signal frequency and liquidity. Override with `--tickers`:
```bash
uv run python -m tr_agent.main trade --tickers AAPL,GOOGL
```

## Iterations

- **v0.1 — current:** Paper trading, technical signals, LLM confirmation, trade journal, memory
- **v0.2 — planned:** Live Trade Republic integration via `pytr`
- **v0.3 — planned:** ML signal model trained on journal data
