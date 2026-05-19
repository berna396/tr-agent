import logging
from datetime import date as date_type
from datetime import datetime, timezone

from tr_agent import yf_utils
from tr_agent.assets import is_crypto

log = logging.getLogger(__name__)


def days_until_earnings(ticker: str) -> int | None:
    """
    Return integer days until the next earnings date, or None if unknown.
    Negative means earnings already passed. Never raises.
    """
    if is_crypto(ticker):
        return None
    try:
        cal = yf_utils.get_ticker_attr(ticker, "calendar")
        if cal is None or not hasattr(cal, "get"):
            return None
        raw_dates = cal.get("Earnings Date", [])
        if not raw_dates or (hasattr(raw_dates, "__len__") and len(raw_dates) == 0):
            return None
        now = datetime.now(timezone.utc)
        deltas = []
        for date in raw_dates:
            if hasattr(date, "to_pydatetime"):
                date = date.to_pydatetime()
            if isinstance(date, date_type) and not isinstance(date, datetime):
                date = datetime(date.year, date.month, date.day, tzinfo=timezone.utc)
            if hasattr(date, "tzinfo") and date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)
            deltas.append((date - now).days)
        upcoming = [d for d in deltas if d >= 0]
        return min(upcoming) if upcoming else None
    except Exception:
        return None


def is_earnings_blackout(ticker: str, days_before: int = 3, days_after: int = 1) -> bool:
    """
    Returns True if we are within the earnings blackout window for this ticker.
    Always returns False on any data error — the default is to allow the trade.
    """
    if is_crypto(ticker):
        return False
    try:
        cal = yf_utils.get_ticker_attr(ticker, "calendar")
        if cal is None:
            return False

        # yfinance returns either a DataFrame or a dict depending on version
        if hasattr(cal, "get"):
            raw_dates = cal.get("Earnings Date", [])
        else:
            return False

        if raw_dates is None or (hasattr(raw_dates, "__len__") and len(raw_dates) == 0):
            return False

        now = datetime.now(timezone.utc)
        for date in raw_dates:
            # Normalize pandas Timestamp → datetime
            if hasattr(date, "to_pydatetime"):
                date = date.to_pydatetime()
            # Normalize plain date → midnight UTC datetime
            if isinstance(date, date_type) and not isinstance(date, datetime):
                date = datetime(date.year, date.month, date.day, tzinfo=timezone.utc)
            if hasattr(date, "tzinfo") and date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)

            days_delta = (date - now).days
            if -days_after <= days_delta <= days_before:
                log.info(f"[Guards] {ticker}: earnings blackout (earnings in {days_delta}d)")
                return True

        return False

    except Exception as e:
        log.warning(f"[Guards] Could not check earnings for {ticker}: {e} — blocking trade (fail-closed)")
        return True
