"""
Fetches recent news headlines per ticker via yfinance.
Results are cached per (ticker, day) to avoid redundant calls during screener runs.
"""

import logging
from datetime import date, datetime, timezone, timedelta
from typing import Optional

import yfinance as yf

log = logging.getLogger(__name__)

_NEWS_CACHE: dict = {}  # keyed by (ticker, date_str)


def fetch_news(
    ticker: str,
    max_articles: int = 3,
    max_age_hours: int = 48,
) -> list[dict]:
    """
    Return recent headlines for ticker. Never raises — returns [] on any error.

    Each item: {title, publisher, age_str}
    """
    today = date.today().isoformat()
    cache_key = (ticker, today)

    if cache_key not in _NEWS_CACHE:
        _NEWS_CACHE.clear()  # drop stale entries from previous days
        try:
            raw = yf.Ticker(ticker).news or []
        except Exception as e:
            log.debug(f"[News] {ticker}: fetch failed — {e}")
            return []
        _NEWS_CACHE[cache_key] = raw

    raw = _NEWS_CACHE[cache_key]
    if not raw:
        return []

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=max_age_hours)
    results = []

    for item in raw:
        try:
            pub_ts = item.get("providerPublishTime") or item.get("pubDate") or 0
            pub_dt = datetime.fromtimestamp(int(pub_ts), tz=timezone.utc)
            if pub_dt < cutoff:
                continue
            age = now - pub_dt
            if age.total_seconds() < 3600:
                age_str = f"{int(age.total_seconds() / 60)}m ago"
            elif age.total_seconds() < 86400:
                age_str = f"{int(age.total_seconds() / 3600)}h ago"
            else:
                age_str = f"{int(age.days)}d ago"

            results.append({
                "title": item.get("title", "").strip(),
                "publisher": item.get("publisher", "").strip(),
                "age_str": age_str,
            })
        except Exception:
            continue

        if len(results) >= max_articles:
            break

    return results
