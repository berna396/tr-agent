"""Web dashboard for tr-agent: process control, portfolio metrics, and live logs."""

import json
import sqlite3
import subprocess
import time
from datetime import datetime
from pathlib import Path

import pytz
from flask import Flask, Response, jsonify, render_template_string, request
from tr_agent import yf_utils

BASE_DIR = Path(__file__).parents[2]
CONF = str(BASE_DIR / "supervisord.conf")
LOG_PATH = BASE_DIR / "data" / "tr-agent.log"
DB_PATH = BASE_DIR / "data" / "journal.db"
PORTFOLIO_PATH = BASE_DIR / "data" / "portfolio_state.json"
CTL = str(BASE_DIR / ".venv/bin/supervisorctl")
PROC = "tr-agent"
ET = pytz.timezone("America/New_York")

app = Flask(__name__)


def ctl_run(*args: str) -> str:
    try:
        result = subprocess.run(
            [CTL, "-c", CONF, *args],
            capture_output=True, text=True, timeout=15,
        )
        return (result.stdout + result.stderr).strip()
    except Exception as exc:
        return str(exc)


def market_open() -> bool:
    now = datetime.now(ET)
    return (
        now.weekday() < 5
        and (now.hour > 9 or (now.hour == 9 and now.minute >= 30))
        and now.hour < 16
    )


@app.get("/")
def index():
    return render_template_string(TEMPLATE)


@app.get("/api/status")
def api_status():
    raw = ctl_run("status", PROC)
    parts = raw.split()
    status = parts[1] if len(parts) > 1 else "UNKNOWN"
    pid, uptime = "", ""
    if "pid" in raw:
        try:
            i = parts.index("pid")
            pid = parts[i + 1].rstrip(",")
            # output: "... pid 65059, uptime 0:00:07"
            j = parts.index("uptime", i)
            uptime = parts[j + 1] if len(parts) > j + 1 else ""
        except (ValueError, IndexError):
            pass
    return jsonify({
        "status": status,
        "pid": pid,
        "uptime": uptime,
        "market_open": market_open(),
        "crypto_active": True,
        "server_time_et": datetime.now(ET).strftime("%H:%M:%S ET"),
    })


@app.post("/api/control")
def api_control():
    data = request.get_json(silent=True) or {}
    action = data.get("action", "")
    if action not in ("start", "stop", "restart"):
        return jsonify({"ok": False, "error": "unknown action"}), 400
    out = ctl_run(action, PROC)
    return jsonify({"ok": True, "output": out})


@app.get("/api/portfolio")
def api_portfolio():
    try:
        with open(PORTFOLIO_PATH) as f:
            state = json.load(f)
    except FileNotFoundError:
        return jsonify({"cash": 0, "invested": 0, "total_value": 0,
                        "return_pct": 0, "num_trades": 0, "positions": []})
    except Exception as e:
        return jsonify({"error": str(e)})

    cash = state.get("cash", 0.0)
    initial = state.get("initial_capital", 10_000.0)
    positions = state.get("positions", {})
    trade_log = state.get("trade_log", [])

    pos_list = []
    invested = 0.0

    if positions:
        try:
            import yfinance as yf
            prices = {}
            for t in positions:
                try:
                    prices[t] = float(yf_utils.ticker(t).fast_info.last_price)
                except Exception:
                    pass
        except Exception:
            prices = {}

        for ticker, pos in positions.items():
            qty = pos["quantity"]
            avg = pos["avg_price"]
            current = prices.get(ticker, avg)
            mkt_val = qty * current
            invested += mkt_val
            pnl_pct = (current - avg) / avg * 100 if avg else 0
            pos_list.append({
                "ticker": ticker,
                "quantity": round(qty, 4),
                "avg_price": round(avg, 4),
                "current_price": round(current, 4),
                "market_value": round(mkt_val, 2),
                "pnl_pct": round(pnl_pct, 2),
            })

    total_value = cash + invested
    return_pct = (total_value - initial) / initial * 100 if initial else 0
    return jsonify({
        "cash": round(cash, 2),
        "invested": round(invested, 2),
        "total_value": round(total_value, 2),
        "return_pct": round(return_pct, 2),
        "num_trades": len(trade_log),
        "positions": pos_list,
    })


@app.get("/api/events")
def api_events():
    try:
        con = sqlite3.connect(DB_PATH)
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT ts, ticker, event_type, data FROM cycle_events ORDER BY id DESC LIMIT 80"
        ).fetchall()
        con.close()
        result = []
        for r in rows:
            try:
                d = json.loads(r["data"])
            except Exception:
                d = {}
            result.append({
                "ts": r["ts"][:19].replace("T", " "),
                "ticker": r["ticker"],
                "type": r["event_type"],
                "data": d,
            })
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})


@app.get("/api/trades")
def api_trades():
    try:
        con = sqlite3.connect(DB_PATH)
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT ticker, buy_ts, sell_ts, buy_price, sell_price, quantity, pnl, pnl_pct "
            "FROM trade_outcomes ORDER BY id DESC LIMIT 20"
        ).fetchall()
        con.close()
        return jsonify([dict(r) for r in rows])
    except Exception as e:
        return jsonify({"error": str(e)})


@app.get("/api/equity")
def api_equity():
    """
    Reconstruct portfolio equity curve from the trade log.

    Each data point represents the portfolio value immediately after an order:
      value = cash_after_order + sum(position.quantity * position.avg_price)
    i.e. open positions are marked at cost (no unrealized P&L) except at the
    final "current" point where live prices are used.
    """
    try:
        with open(PORTFOLIO_PATH) as f:
            state = json.load(f)
    except FileNotFoundError:
        return jsonify({"initial": 10_000.0, "points": []})
    except Exception as e:
        return jsonify({"error": str(e)})

    initial = state.get("initial_capital", 10_000.0)
    trade_log = state.get("trade_log", [])

    # Sort chronologically
    trades_sorted = sorted(trade_log, key=lambda t: t.get("timestamp", ""))

    # Simulate cash + cost_basis after each order
    cash = initial
    positions: dict[str, dict] = {}  # ticker -> {quantity, avg_price}

    points: list[dict] = []

    for t in trades_sorted:
        side = t.get("side", "")
        ticker = t.get("ticker", "")
        qty = t.get("quantity", 0.0)
        price = t.get("fill_price", 0.0)
        ts = t.get("timestamp", "")[:10]  # YYYY-MM-DD

        if side == "buy":
            cash -= qty * price
            if ticker in positions:
                old = positions[ticker]
                total_qty = old["quantity"] + qty
                avg = (old["quantity"] * old["avg_price"] + qty * price) / total_qty
                positions[ticker] = {"quantity": total_qty, "avg_price": avg}
            else:
                positions[ticker] = {"quantity": qty, "avg_price": price}
        elif side == "sell":
            cash += qty * price
            if ticker in positions:
                remaining = positions[ticker]["quantity"] - qty
                if remaining <= 0:
                    del positions[ticker]
                else:
                    positions[ticker]["quantity"] = remaining

        cost_basis = sum(p["quantity"] * p["avg_price"] for p in positions.values())
        points.append({"ts": ts, "value": round(cash + cost_basis, 2)})

    # Current point: replace cost_basis with live market values
    today = datetime.now(ET).strftime("%Y-%m-%d")
    current_positions = state.get("positions", {})
    current_cash = state.get("cash", cash)

    if current_positions:
        import yfinance as yf
        mkt_val = 0.0
        for ticker, pos in current_positions.items():
            try:
                live_price = float(yf_utils.ticker(ticker).fast_info.last_price)
            except Exception:
                live_price = pos["avg_price"]
            mkt_val += pos["quantity"] * live_price
        current_value = round(current_cash + mkt_val, 2)
    else:
        current_value = round(current_cash, 2)

    # Replace last point if same date, otherwise append
    if points and points[-1]["ts"] == today:
        points[-1] = {"ts": today, "value": current_value, "current": True}
    else:
        points.append({"ts": today, "value": current_value, "current": True})

    # Always prepend the starting point (day before first trade, or today if no trades)
    if points:
        first_ts = points[0]["ts"]
    else:
        first_ts = today
    points.insert(0, {"ts": first_ts, "value": initial, "start": True})

    return jsonify({"initial": initial, "points": points})


@app.get("/api/logs/stream")
def api_logs_stream():
    def generate():
        try:
            with open(LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
                for line in lines[-200:]:
                    yield f"data: {json.dumps(line.rstrip())}\n\n"
                while True:
                    line = f.readline()
                    if line:
                        yield f"data: {json.dumps(line.rstrip())}\n\n"
                    else:
                        time.sleep(0.3)
                        yield ": keepalive\n\n"
        except GeneratorExit:
            pass

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>tr-agent</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
  <style>
    body { background: #0f172a; }
    #logs { height: 420px; overflow-y: auto; scroll-behavior: smooth; }
    #events-body { max-height: 280px; overflow-y: auto; }
    .log-line { white-space: pre-wrap; word-break: break-all; }
    .btn { @apply px-3 py-1.5 text-sm rounded font-medium transition-colors duration-150 cursor-pointer; }
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #1e293b; }
    ::-webkit-scrollbar-thumb { background: #475569; border-radius: 3px; }
  </style>
</head>
<body class="text-slate-200 min-h-screen">

  <!-- Header -->
  <header class="bg-slate-800 border-b border-slate-700 sticky top-0 z-10">
    <div class="max-w-screen-xl mx-auto px-5 py-3 flex items-center justify-between gap-4">
      <div class="flex items-center gap-4 min-w-0">
        <span class="text-lg font-bold text-white whitespace-nowrap">🤖 tr-agent</span>
        <div class="flex items-center gap-2">
          <div id="s-dot" class="w-2.5 h-2.5 rounded-full bg-slate-500 shrink-0"></div>
          <span id="s-text" class="text-sm font-medium text-slate-400">—</span>
          <span id="s-uptime" class="text-xs text-slate-500 hidden sm:inline"></span>
        </div>
        <div class="flex items-center gap-1.5 text-xs text-slate-500">
          <span id="market-badge" class="px-2 py-0.5 rounded-full bg-slate-700 text-slate-400">—</span>
          <span id="server-time"></span>
        </div>
      </div>
      <div class="flex gap-1.5 shrink-0">
        <button onclick="control('start')"
          class="px-3 py-1.5 text-xs bg-emerald-700 hover:bg-emerald-600 rounded font-medium transition-colors">
          Start
        </button>
        <button onclick="control('stop')"
          class="px-3 py-1.5 text-xs bg-rose-800 hover:bg-rose-700 rounded font-medium transition-colors">
          Stop
        </button>
        <button onclick="control('restart')"
          class="px-3 py-1.5 text-xs bg-blue-700 hover:bg-blue-600 rounded font-medium transition-colors">
          Restart
        </button>
      </div>
    </div>
  </header>

  <main class="max-w-screen-xl mx-auto px-5 py-5 space-y-5">

    <!-- Metric cards -->
    <div class="grid grid-cols-2 sm:grid-cols-4 gap-3">
      <div class="bg-slate-800 rounded-xl p-4 border border-slate-700">
        <p class="text-xs text-slate-400 uppercase tracking-wider mb-1">Cash</p>
        <p id="m-cash" class="text-2xl font-bold text-white">—</p>
      </div>
      <div class="bg-slate-800 rounded-xl p-4 border border-slate-700">
        <p class="text-xs text-slate-400 uppercase tracking-wider mb-1">Portfolio</p>
        <p id="m-total" class="text-2xl font-bold text-white">—</p>
        <p id="m-invested" class="text-xs text-slate-500 mt-0.5">— invested</p>
      </div>
      <div class="bg-slate-800 rounded-xl p-4 border border-slate-700">
        <p class="text-xs text-slate-400 uppercase tracking-wider mb-1">Return</p>
        <p id="m-return" class="text-2xl font-bold">—</p>
      </div>
      <div class="bg-slate-800 rounded-xl p-4 border border-slate-700">
        <p class="text-xs text-slate-400 uppercase tracking-wider mb-1">Trades</p>
        <p id="m-trades" class="text-2xl font-bold text-white">—</p>
      </div>
    </div>

    <!-- Equity Chart -->
    <div class="bg-slate-800 rounded-xl border border-slate-700 p-4">
      <div class="flex items-center justify-between mb-3">
        <h2 class="text-xs font-semibold text-slate-400 uppercase tracking-wider">Portfolio Performance</h2>
        <span id="equity-label" class="text-xs text-slate-500"></span>
      </div>
      <div class="relative" style="height:200px">
        <canvas id="equity-chart"></canvas>
      </div>
    </div>

    <!-- Positions + Events row -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-5">

      <!-- Open Positions -->
      <div class="bg-slate-800 rounded-xl border border-slate-700 p-4">
        <h2 class="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">Open Positions</h2>
        <div id="positions-body">
          <p class="text-slate-500 text-sm">Loading…</p>
        </div>
      </div>

      <!-- Recent Events -->
      <div class="bg-slate-800 rounded-xl border border-slate-700 p-4">
        <h2 class="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">Recent Events</h2>
        <div id="events-body">
          <p class="text-slate-500 text-sm">Loading…</p>
        </div>
      </div>
    </div>

    <!-- Trade History (shown only if trades exist) -->
    <div id="trades-section" class="hidden bg-slate-800 rounded-xl border border-slate-700 p-4">
      <h2 class="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">Trade History</h2>
      <div id="trades-body" class="overflow-x-auto"></div>
    </div>

    <!-- Log viewer -->
    <div class="bg-slate-800 rounded-xl border border-slate-700 p-4">
      <div class="flex items-center justify-between mb-3">
        <h2 class="text-xs font-semibold text-slate-400 uppercase tracking-wider">Logs</h2>
        <div class="flex items-center gap-4">
          <label class="flex items-center gap-1.5 text-xs text-slate-400 cursor-pointer select-none">
            <input type="checkbox" id="autoscroll" checked class="accent-blue-500">
            Auto-scroll
          </label>
          <button onclick="clearLogs()"
            class="text-xs text-slate-500 hover:text-slate-300 transition-colors">
            Clear
          </button>
        </div>
      </div>
      <div id="logs" class="bg-slate-900 rounded-lg p-3 font-mono text-xs"></div>
    </div>

  </main>

<script>
const fmt2 = (n) => n.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2});
const fmtUSD = (n) => '$' + fmt2(n);
const sign = (n) => (n >= 0 ? '+' : '') + fmt2(n);

// ── Status ────────────────────────────────────────────────────────────────────
async function fetchStatus() {
  try {
    const d = await fetch('/api/status').then(r => r.json());
    const dot  = document.getElementById('s-dot');
    const text = document.getElementById('s-text');
    const up   = document.getElementById('s-uptime');
    const mkt  = document.getElementById('market-badge');
    document.getElementById('server-time').textContent = d.server_time_et || '';
    text.textContent = d.status;
    up.textContent = d.uptime ? `uptime ${d.uptime}` : '';
    up.classList.toggle('hidden', !d.uptime);
    if (d.status === 'RUNNING') {
      dot.className = 'w-2.5 h-2.5 rounded-full bg-emerald-400 shrink-0 shadow-sm shadow-emerald-400';
      text.className = 'text-sm font-medium text-emerald-400';
    } else if (d.status === 'STOPPED') {
      dot.className = 'w-2.5 h-2.5 rounded-full bg-rose-400 shrink-0';
      text.className = 'text-sm font-medium text-rose-400';
    } else {
      dot.className = 'w-2.5 h-2.5 rounded-full bg-amber-400 shrink-0';
      text.className = 'text-sm font-medium text-amber-400';
    }
    if (d.market_open) {
      mkt.textContent = '● NYSE open · ₿ 24/7';
      mkt.className = 'px-2 py-0.5 rounded-full bg-emerald-900 text-emerald-400 text-xs';
    } else {
      mkt.textContent = '○ NYSE closed · ₿ 24/7';
      mkt.className = 'px-2 py-0.5 rounded-full bg-slate-700 text-amber-400 text-xs';
    }
  } catch(e) {}
}

// ── Portfolio ─────────────────────────────────────────────────────────────────
async function fetchPortfolio() {
  try {
    const d = await fetch('/api/portfolio').then(r => r.json());
    if (d.error) return;
    document.getElementById('m-cash').textContent    = fmtUSD(d.cash);
    document.getElementById('m-total').textContent   = fmtUSD(d.total_value);
    document.getElementById('m-invested').textContent = fmtUSD(d.invested) + ' invested';
    document.getElementById('m-trades').textContent  = d.num_trades;

    const ret = document.getElementById('m-return');
    ret.textContent = sign(d.return_pct) + '%';
    ret.className = 'text-2xl font-bold ' + (d.return_pct >= 0 ? 'text-emerald-400' : 'text-rose-400');

    const posDiv = document.getElementById('positions-body');
    if (!d.positions.length) {
      posDiv.innerHTML = '<p class="text-slate-500 text-sm">No open positions</p>';
    } else {
      posDiv.innerHTML = `<table class="w-full text-sm">
        <thead><tr class="text-xs text-slate-500 border-b border-slate-700">
          <th class="text-left pb-2 font-medium">Ticker</th>
          <th class="text-right pb-2 font-medium">Qty</th>
          <th class="text-right pb-2 font-medium">Avg</th>
          <th class="text-right pb-2 font-medium">Current</th>
          <th class="text-right pb-2 font-medium">Value</th>
          <th class="text-right pb-2 font-medium">P&amp;L</th>
        </tr></thead>
        <tbody>
        ${d.positions.map(p => {
          const isCrypto = p.ticker.includes('-USD') || p.ticker.includes('-USDT') || p.ticker.includes('-BTC') || p.ticker.includes('-ETH');
          const cryptoBadge = isCrypto ? '<span class="ml-1 text-xs text-amber-400 font-normal">₿</span>' : '';
          return `<tr class="border-b border-slate-700/50">
          <td class="font-semibold py-2 text-white">${p.ticker}${cryptoBadge}</td>
          <td class="text-right text-slate-300 py-2">${p.quantity}</td>
          <td class="text-right text-slate-300 py-2">$${p.avg_price}</td>
          <td class="text-right text-slate-300 py-2">$${p.current_price}</td>
          <td class="text-right text-slate-300 py-2">${fmtUSD(p.market_value)}</td>
          <td class="text-right py-2 font-medium ${p.pnl_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'}">
            ${sign(p.pnl_pct)}%
          </td>
        </tr>`;}).join('')}
        </tbody></table>`;
    }
  } catch(e) {}
}

// ── Events ────────────────────────────────────────────────────────────────────
const SIG_COLOR = { BUY: 'text-emerald-400', SELL: 'text-rose-400', NEUTRAL: 'text-slate-400' };

function eventDetail(e) {
  const d = e.data;
  switch (e.type) {
    case 'signal': {
      const sig = (d.signal || '').toUpperCase();
      const col = SIG_COLOR[sig] || 'text-slate-400';
      return `<span class="${col} font-semibold">${sig}</span>
        <span class="text-slate-400"> RSI <span class="text-slate-200">${(d.rsi||0).toFixed(0)}</span>
        MACD <span class="text-slate-200">${(d.macd_hist||0).toFixed(2)}</span></span>`;
    }
    case 'llm_decision':
      return d.confirmed
        ? `<span class="text-emerald-400">✓ confirmed</span> <span class="text-slate-500 text-xs">${(d.reasoning||'').slice(0,60)}</span>`
        : `<span class="text-rose-400">✗ rejected</span> <span class="text-slate-500 text-xs">${(d.reasoning||'').slice(0,60)}</span>`;
    case 'order': {
      const col = d.side === 'buy' ? 'text-emerald-400' : 'text-rose-400';
      return `<span class="${col} font-semibold">${(d.side||'').toUpperCase()}</span>
        <span class="text-slate-300"> ${(d.quantity||0).toFixed(2)} @ $${(d.fill_price||0).toFixed(2)}</span>`;
    }
    case 'risk':
      return d.approved
        ? `<span class="text-emerald-400">✓ approved</span>`
        : `<span class="text-rose-400">✗ blocked</span> <span class="text-slate-500 text-xs">${(d.reason||'').slice(0,50)}</span>`;
    default:
      return `<span class="text-slate-500">${JSON.stringify(d).slice(0, 60)}</span>`;
  }
}

async function fetchEvents() {
  try {
    const events = await fetch('/api/events').then(r => r.json());
    const div = document.getElementById('events-body');
    div.innerHTML = events.map(e => `
      <div class="flex items-baseline gap-2 py-1 border-b border-slate-700/50 text-xs">
        <span class="text-slate-500 shrink-0 tabular-nums">${e.ts.slice(11)}</span>
        <span class="text-blue-400 font-medium shrink-0 w-20">${e.ticker}</span>
        <span class="text-slate-500 shrink-0 w-20">${e.type}</span>
        <span class="min-w-0">${eventDetail(e)}</span>
      </div>`).join('');
  } catch(e) {}
}

// ── Trades ────────────────────────────────────────────────────────────────────
async function fetchTrades() {
  try {
    const trades = await fetch('/api/trades').then(r => r.json());
    if (!trades.length) return;
    document.getElementById('trades-section').classList.remove('hidden');
    document.getElementById('trades-body').innerHTML = `
      <table class="w-full text-sm min-w-[520px]">
        <thead><tr class="text-xs text-slate-500 border-b border-slate-700">
          <th class="text-left pb-2 font-medium">Ticker</th>
          <th class="text-right pb-2 font-medium">Buy $</th>
          <th class="text-right pb-2 font-medium">Sell $</th>
          <th class="text-right pb-2 font-medium">Qty</th>
          <th class="text-right pb-2 font-medium">P&amp;L</th>
          <th class="text-right pb-2 font-medium">P&amp;L %</th>
        </tr></thead>
        <tbody>
        ${trades.map(t => `<tr class="border-b border-slate-700/50">
          <td class="font-semibold py-2 text-white">${t.ticker}</td>
          <td class="text-right text-slate-300 py-2">$${(t.buy_price||0).toFixed(2)}</td>
          <td class="text-right text-slate-300 py-2">$${(t.sell_price||0).toFixed(2)}</td>
          <td class="text-right text-slate-300 py-2">${(t.quantity||0).toFixed(2)}</td>
          <td class="text-right py-2 font-medium ${t.pnl >= 0 ? 'text-emerald-400' : 'text-rose-400'}">
            ${t.pnl >= 0 ? '+' : ''}$${Math.abs(t.pnl||0).toFixed(2)}
          </td>
          <td class="text-right py-2 font-medium ${t.pnl_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'}">
            ${sign(t.pnl_pct||0)}%
          </td>
        </tr>`).join('')}
        </tbody></table>`;
  } catch(e) {}
}

// ── Logs ──────────────────────────────────────────────────────────────────────
const logsDiv = document.getElementById('logs');

function appendLog(text) {
  const el = document.createElement('div');
  el.className = 'log-line leading-5';
  // level colouring
  if (/\bERROR\b/.test(text))   el.style.color = '#f87171';
  else if (/\bWARNING\b/.test(text)) el.style.color = '#fbbf24';
  else if (/\[Cycle\].*BUY/.test(text))  el.style.color = '#4ade80';
  else if (/\[Cycle\].*SELL/.test(text)) el.style.color = '#f87171';
  else el.style.color = '#94a3b8';
  el.textContent = text;
  logsDiv.appendChild(el);
  while (logsDiv.children.length > 600) logsDiv.removeChild(logsDiv.firstChild);
  if (document.getElementById('autoscroll').checked) {
    logsDiv.scrollTop = logsDiv.scrollHeight;
  }
}

function clearLogs() { logsDiv.innerHTML = ''; }

const evtSource = new EventSource('/api/logs/stream');
evtSource.onmessage = e => appendLog(JSON.parse(e.data));
evtSource.onerror   = () => appendLog('— log stream disconnected —');

// ── Process control ───────────────────────────────────────────────────────────
async function control(action) {
  const label = { start: 'Starting', stop: 'Stopping', restart: 'Restarting' }[action] || action;
  appendLog(`[dashboard] ${label} ${action}…`);
  try {
    await fetch('/api/control', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action }),
    });
    setTimeout(fetchStatus, 2000);
    setTimeout(fetchStatus, 4000);
  } catch(e) {}
}

// ── Equity chart ──────────────────────────────────────────────────────────────
let equityChart = null;

async function fetchEquity() {
  try {
    const d = await fetch('/api/equity').then(r => r.json());
    if (d.error || !d.points || d.points.length < 2) return;

    const labels  = d.points.map(p => p.ts);
    const values  = d.points.map(p => p.value);
    const initial = d.initial;
    const last    = values[values.length - 1];
    const gain    = last - initial;
    const gainPct = (gain / initial * 100).toFixed(2);

    document.getElementById('equity-label').textContent =
      (gain >= 0 ? '+' : '') + '$' + Math.abs(gain).toFixed(2) +
      ' (' + (gain >= 0 ? '+' : '') + gainPct + '%) all-time';

    const lineColor = last >= initial ? '#34d399' : '#f87171';

    const canvas = document.getElementById('equity-chart');
    const ctx = canvas.getContext('2d');

    // Gradient fill
    const grad = ctx.createLinearGradient(0, 0, 0, 200);
    grad.addColorStop(0, last >= initial ? 'rgba(52,211,153,0.25)' : 'rgba(248,113,113,0.25)');
    grad.addColorStop(1, 'rgba(0,0,0,0)');

    // Baseline dataset (dashed line at initial capital)
    const baseline = labels.map(() => initial);

    const chartData = {
      labels,
      datasets: [
        {
          data: values,
          borderColor: lineColor,
          borderWidth: 2,
          backgroundColor: grad,
          fill: true,
          tension: 0.3,
          pointRadius: values.map((_, i) => (i === 0 || i === values.length - 1) ? 4 : 2),
          pointBackgroundColor: lineColor,
          order: 1,
        },
        {
          data: baseline,
          borderColor: 'rgba(100,116,139,0.4)',
          borderWidth: 1,
          borderDash: [4, 4],
          backgroundColor: 'transparent',
          fill: false,
          pointRadius: 0,
          tension: 0,
          order: 2,
        },
      ],
    };

    if (equityChart) {
      equityChart.data = chartData;
      equityChart.update('none');
      return;
    }

    equityChart = new Chart(ctx, {
      type: 'line',
      data: chartData,
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { display: false },
          tooltip: {
            filter: item => item.datasetIndex === 0,
            backgroundColor: '#1e293b',
            borderColor: '#334155',
            borderWidth: 1,
            titleColor: '#94a3b8',
            bodyColor: '#e2e8f0',
            callbacks: {
              label: ctx => ' $' + ctx.parsed.y.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2}),
            },
          },
        },
        scales: {
          x: {
            grid: { color: 'rgba(255,255,255,0.04)' },
            ticks: { color: '#64748b', maxTicksLimit: 6, font: { size: 11 } },
          },
          y: {
            grid: { color: 'rgba(255,255,255,0.04)' },
            ticks: {
              color: '#64748b',
              font: { size: 11 },
              callback: v => '$' + v.toLocaleString('en-US', {minimumFractionDigits: 0}),
            },
          },
        },
      },
    });
  } catch(e) {}
}

// ── Init + polling ────────────────────────────────────────────────────────────
fetchStatus();
fetchPortfolio();
fetchEvents();
fetchTrades();
fetchEquity();

setInterval(fetchStatus,    5_000);
setInterval(fetchPortfolio, 30_000);
setInterval(fetchEvents,    10_000);
setInterval(fetchTrades,    30_000);
setInterval(fetchEquity,    60_000);
</script>
</body>
</html>"""


def main(host: str = "0.0.0.0", port: int = 8080):
    app.run(host=host, port=port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
