"""
Trade journal backed by SQLite.

Two tables:
  cycle_events  — every step of every cycle (signal, risk, llm decision, order)
  trade_outcomes — completed buy+sell pairs with realized P&L, used for memory
"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

_DB_PATH = Path(__file__).parents[2] / "data" / "journal.db"


@contextmanager
def _conn(path: Path = _DB_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init(path: Path = _DB_PATH) -> None:
    with _conn(path) as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS cycle_events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          TEXT    NOT NULL,
            ticker      TEXT    NOT NULL,
            event_type  TEXT    NOT NULL,
            data        TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS trade_outcomes (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker           TEXT    NOT NULL,
            buy_ts           TEXT    NOT NULL,
            sell_ts          TEXT    NOT NULL,
            buy_price        REAL    NOT NULL,
            sell_price       REAL    NOT NULL,
            quantity         REAL    NOT NULL,
            pnl              REAL    NOT NULL,
            pnl_pct          REAL    NOT NULL,
            buy_reasoning    TEXT,
            sell_reasoning   TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_events_ticker ON cycle_events(ticker);
        CREATE INDEX IF NOT EXISTS idx_outcomes_ticker ON trade_outcomes(ticker);
        """)


# ── Logging helpers ──────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_signal(ticker: str, signal: str, rsi: float, macd_hist: float,
               sma_20: float, sma_50: float, close: float, reasoning: str,
               path: Path = _DB_PATH) -> int:
    data = dict(signal=signal, rsi=round(rsi, 2), macd_hist=round(macd_hist, 4),
                sma_20=round(sma_20, 2), sma_50=round(sma_50, 2),
                close=round(close, 4), reasoning=reasoning)
    with _conn(path) as con:
        cur = con.execute(
            "INSERT INTO cycle_events(ts, ticker, event_type, data) VALUES(?,?,?,?)",
            (_now(), ticker, "signal", json.dumps(data)),
        )
        return cur.lastrowid


def log_risk(ticker: str, approved: bool, reason: str, max_qty: float,
             path: Path = _DB_PATH) -> None:
    data = dict(approved=approved, reason=reason, max_quantity=round(max_qty, 4))
    with _conn(path) as con:
        con.execute(
            "INSERT INTO cycle_events(ts, ticker, event_type, data) VALUES(?,?,?,?)",
            (_now(), ticker, "risk", json.dumps(data)),
        )


def log_llm_decision(ticker: str, confirmed: bool, quantity: float,
                     reasoning: str, path: Path = _DB_PATH) -> None:
    data = dict(confirmed=confirmed, quantity=round(quantity, 4), reasoning=reasoning)
    with _conn(path) as con:
        con.execute(
            "INSERT INTO cycle_events(ts, ticker, event_type, data) VALUES(?,?,?,?)",
            (_now(), ticker, "llm_decision", json.dumps(data)),
        )


def log_order(ticker: str, side: str, quantity: float, fill_price: float,
              order_id: str, path: Path = _DB_PATH) -> None:
    data = dict(side=side, quantity=round(quantity, 4),
                fill_price=round(fill_price, 4), order_id=order_id)
    with _conn(path) as con:
        con.execute(
            "INSERT INTO cycle_events(ts, ticker, event_type, data) VALUES(?,?,?,?)",
            (_now(), ticker, "order", json.dumps(data)),
        )


def record_outcome(ticker: str, buy_ts: str, sell_ts: str, buy_price: float,
                   sell_price: float, quantity: float, buy_reasoning: str = "",
                   sell_reasoning: str = "", path: Path = _DB_PATH) -> None:
    pnl = (sell_price - buy_price) * quantity
    pnl_pct = (sell_price - buy_price) / buy_price * 100
    with _conn(path) as con:
        con.execute("""
            INSERT INTO trade_outcomes
            (ticker, buy_ts, sell_ts, buy_price, sell_price, quantity, pnl, pnl_pct, buy_reasoning, sell_reasoning)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (ticker, buy_ts, sell_ts, round(buy_price, 4), round(sell_price, 4),
              round(quantity, 4), round(pnl, 2), round(pnl_pct, 2),
              buy_reasoning, sell_reasoning))
    log.info(f"[Journal] Outcome recorded: {ticker} P&L={pnl:+.2f} ({pnl_pct:+.1f}%)")


# ── Query helpers ─────────────────────────────────────────────────────────────

@dataclass
class TickerStats:
    ticker: str
    total_trades: int
    wins: int
    losses: int
    win_rate_pct: float
    avg_win_pct: float
    avg_loss_pct: float
    avg_pnl_pct: float
    recent_outcomes: list[dict]   # last 5, newest first


def get_ticker_stats(ticker: str, path: Path = _DB_PATH) -> Optional[TickerStats]:
    """Returns performance stats for a ticker. None if no history yet."""
    with _conn(path) as con:
        rows = con.execute(
            "SELECT pnl, pnl_pct, buy_ts, sell_ts, buy_price, sell_price FROM trade_outcomes "
            "WHERE ticker=? ORDER BY sell_ts DESC",
            (ticker,),
        ).fetchall()

    if not rows:
        return None

    wins = [r for r in rows if r["pnl"] > 0]
    losses = [r for r in rows if r["pnl"] <= 0]

    return TickerStats(
        ticker=ticker,
        total_trades=len(rows),
        wins=len(wins),
        losses=len(losses),
        win_rate_pct=len(wins) / len(rows) * 100,
        avg_win_pct=sum(r["pnl_pct"] for r in wins) / len(wins) if wins else 0.0,
        avg_loss_pct=sum(r["pnl_pct"] for r in losses) / len(losses) if losses else 0.0,
        avg_pnl_pct=sum(r["pnl_pct"] for r in rows) / len(rows),
        recent_outcomes=[
            {"pnl": r["pnl"], "pnl_pct": r["pnl_pct"],
             "buy_price": r["buy_price"], "sell_price": r["sell_price"],
             "buy_ts": r["buy_ts"][:10], "sell_ts": r["sell_ts"][:10]}
            for r in rows[:5]
        ],
    )


def get_pending_buy(ticker: str, path: Path = _DB_PATH) -> Optional[dict]:
    """
    Returns the most recent buy order for a ticker that has no matching sell outcome.
    Used to record outcomes when a sell order fires.
    """
    with _conn(path) as con:
        # Find last buy order
        buy_row = con.execute("""
            SELECT ts, data FROM cycle_events
            WHERE ticker=? AND event_type='order' AND json_extract(data,'$.side')='buy'
            ORDER BY ts DESC LIMIT 1
        """, (ticker,)).fetchone()

        if not buy_row:
            return None

        buy_data = json.loads(buy_row["data"])
        # Check it's not already in outcomes
        outcome = con.execute(
            "SELECT id FROM trade_outcomes WHERE ticker=? AND buy_ts=?",
            (ticker, buy_row["ts"]),
        ).fetchone()

        if outcome:
            return None  # already recorded

        return {"ts": buy_row["ts"], "fill_price": buy_data["fill_price"]}
