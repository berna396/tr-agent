import json
import logging
from pathlib import Path
from typing import Optional

from tr_agent.config import DEFAULT_WATCHLIST, settings
from tr_agent.signals import technical
from tr_agent.signals.technical import Signal, TechnicalAnalysis

log = logging.getLogger(__name__)

CANDIDATE_POOL = [
    # Tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD",
    "INTC", "QCOM", "CRM", "ADBE", "NFLX", "MU", "AMAT", "ORCL", "PYPL", "UBER",
    # Finance
    "JPM", "BAC", "GS", "MS", "V", "MA", "AXP", "BLK",
    # Health
    "JNJ", "UNH", "LLY", "ABBV", "MRK", "PFE",
    # Consumer / Retail
    "WMT", "HD", "COST", "NKE", "MCD", "SBUX",
    # Energy
    "XOM", "CVX", "COP", "SLB",
    # Industrial / Other
    "BA", "CAT", "HON", "GE", "T", "VZ",
]


def score_ticker(analysis: TechnicalAnalysis) -> float:
    """Score a ticker 0–5.5 based on signal quality, trend strength, volume, and ML."""
    score = 0.0

    if analysis.signal != Signal.NEUTRAL:
        score += 2.0

    adx = analysis.ml_features.get("adx", 0.0)
    if adx > 25:
        score += 1.0

    vol_ratio = analysis.ml_features.get("volume_ratio", 1.0)
    if vol_ratio > 1.2:
        score += 1.0

    if analysis.ml_available and analysis.ml_confidence is not None:
        if analysis.ml_confidence > 0.55:
            score += 1.0
        elif analysis.ml_confidence < 0.40:
            score -= 0.5

    if analysis.rsi is not None and (analysis.rsi < 30 or analysis.rsi > 70):
        score += 0.5

    return score


def _passes_filters(analysis: TechnicalAnalysis) -> bool:
    if analysis.close < settings.screener_min_price:
        return False
    vol_ratio = analysis.ml_features.get("volume_ratio", 1.0)
    # volume_ratio < 0.1 likely means very low absolute volume; reject
    if vol_ratio < 0.1:
        return False
    return True


def screen(
    pool: Optional[list[str]] = None,
    top_n: Optional[int] = None,
) -> list[str]:
    """
    Analyze every ticker in pool, score them, return top_n with a non-neutral signal.
    Never raises — returns DEFAULT_WATCHLIST on complete failure.
    """
    pool = pool or CANDIDATE_POOL
    top_n = top_n or settings.screener_top_n

    scored: list[tuple[str, float, TechnicalAnalysis]] = []
    errors = 0

    for ticker in pool:
        try:
            analysis = technical.analyze(ticker)
            if not _passes_filters(analysis):
                log.debug(f"[Screener] {ticker} filtered out (price/volume)")
                continue
            s = score_ticker(analysis)
            rsi_str = f"{analysis.rsi:.1f}" if analysis.rsi is not None else "n/a"
            log.info(
                f"[Screener] {ticker}: {analysis.signal.value.upper()} "
                f"score={s:.1f} rsi={rsi_str} "
                f"adx={analysis.ml_features.get('adx', 0):.1f}"
            )
            scored.append((ticker, s, analysis))
        except Exception as e:
            log.warning(f"[Screener] {ticker} skipped: {e}")
            errors += 1

    if not scored:
        log.warning(f"[Screener] All {len(pool)} tickers failed — falling back to default watchlist")
        return list(DEFAULT_WATCHLIST)

    # Only keep tickers with a non-neutral signal, sort by score desc
    with_signal = [(t, s, a) for t, s, a in scored if a.signal != Signal.NEUTRAL]
    if not with_signal:
        # Nothing signaling today — take the highest-scored ones anyway
        log.info("[Screener] No tickers with active signal today — using top-scored candidates")
        with_signal = scored

    with_signal.sort(key=lambda x: x[1], reverse=True)
    selected = [t for t, _, _ in with_signal[:top_n]]

    log.info(
        f"[Screener] Selected {len(selected)}/{len(pool)} tickers "
        f"(errors={errors}): {', '.join(selected)}"
    )
    return selected


def load_active_watchlist(path: Path) -> list[str]:
    """Return today's active watchlist from JSON, or DEFAULT_WATCHLIST if not found."""
    path = Path(path)
    if not path.exists():
        return list(DEFAULT_WATCHLIST)
    try:
        with open(path) as f:
            data = json.load(f)
        tickers = data.get("tickers", [])
        if tickers:
            return tickers
    except Exception as e:
        log.warning(f"[Screener] Failed to load active watchlist: {e}")
    return list(DEFAULT_WATCHLIST)


def save_active_watchlist(tickers: list[str], path: Path) -> None:
    from datetime import datetime, timezone
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(
            {"tickers": tickers, "updated_at": datetime.now(timezone.utc).isoformat()},
            f,
            indent=2,
        )
    log.info(f"[Screener] Active watchlist saved: {', '.join(tickers)}")
