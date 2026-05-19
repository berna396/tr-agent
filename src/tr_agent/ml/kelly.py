"""Half-Kelly position sizing based on ML win probability and historical trade stats."""

import logging
import sqlite3
from pathlib import Path

log = logging.getLogger(__name__)

_MIN_TRADES = 10      # need at least this many closed trades for reliable stats
_FALLBACK_FRACTION = None  # None → caller uses settings.max_trade_pct


def get_historical_stats(db_path: Path, lookback_days: int = 90) -> tuple[float, float] | None:
    """
    Return (avg_win_pct, avg_loss_pct) from the last `lookback_days` of closed trades.
    Returns None if fewer than _MIN_TRADES are available.
    Both values are positive percentages (e.g. 0.03 = 3%).
    """
    db_path = Path(db_path)
    if not db_path.exists():
        return None
    try:
        with sqlite3.connect(db_path) as con:
            rows = con.execute(
                """
                SELECT pnl_pct FROM trade_outcomes
                WHERE sell_ts >= datetime('now', ?)
                """,
                (f"-{lookback_days} days",),
            ).fetchall()
        if len(rows) < _MIN_TRADES:
            return None
        pcts = [r[0] for r in rows]
        wins  = [p for p in pcts if p > 0]
        losses = [abs(p) for p in pcts if p <= 0]
        if not wins or not losses:
            return None
        return sum(wins) / len(wins), sum(losses) / len(losses)
    except Exception as e:
        log.debug(f"[Kelly] Could not load historical stats: {e}")
        return None


def compute_kelly_fraction(
    win_prob: float,
    avg_win_pct: float,
    avg_loss_pct: float,
) -> float:
    """
    Half-Kelly fraction of capital to allocate to a single trade.
    Clamped to [0.01, 0.25] regardless of the formula output.

    Kelly formula:  f* = (p*b - (1-p)) / b   where b = avg_win / avg_loss
    Half-Kelly:     f  = f* / 2
    """
    if avg_loss_pct <= 0 or win_prob <= 0 or win_prob >= 1:
        return 0.05   # conservative fallback
    b = avg_win_pct / avg_loss_pct
    f_star = (win_prob * b - (1.0 - win_prob)) / b
    half_kelly = max(0.0, f_star) / 2.0
    clamped = round(min(max(half_kelly, 0.01), 0.25), 4)
    log.debug(
        f"[Kelly] win_prob={win_prob:.2f} b={b:.2f} f*={f_star:.3f} "
        f"half_kelly={half_kelly:.3f} → {clamped:.4f}"
    )
    return clamped
