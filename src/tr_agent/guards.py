import logging
from datetime import datetime, timezone

import yfinance as yf

log = logging.getLogger(__name__)


def is_earnings_blackout(ticker: str, days_before: int = 3, days_after: int = 1) -> bool:
    """
    Returns True if we are within the earnings blackout window for this ticker.
    Always returns False on any data error — the default is to allow the trade.
    """
    try:
        cal = yf.Ticker(ticker).calendar
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
            if hasattr(date, "tzinfo") and date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)

            days_delta = (date - now).days
            if -days_after <= days_delta <= days_before:
                log.info(f"[Guards] {ticker}: earnings blackout (earnings in {days_delta}d)")
                return True

        return False

    except Exception as e:
        log.debug(f"[Guards] Could not check earnings for {ticker}: {e}")
        return False
