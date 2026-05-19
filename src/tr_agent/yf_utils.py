"""
yfinance wrappers with request timeouts.

yfinance 1.3+ rejects custom requests.Session (requires curl_cffi).
We add timeouts via concurrent.futures instead — each blocking call runs
in a thread and is cancelled if it exceeds _TIMEOUT seconds.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

import yfinance as yf

log = logging.getLogger(__name__)

_TIMEOUT = 20  # seconds


def download(ticker: str, **kwargs):
    """yf.download with a hard timeout (supported natively via the timeout= param)."""
    import pandas as pd
    kwargs.setdefault("progress", False)
    kwargs.setdefault("auto_adjust", True)
    kwargs["timeout"] = _TIMEOUT
    try:
        return yf.download(ticker, **kwargs)
    except Exception as e:
        log.warning(f"[yf] download({ticker}) failed: {e}")
        return pd.DataFrame()


def get_ticker_attr(symbol: str, attr: str, default=None):
    """Return yf.Ticker(symbol).<attr> with a timeout. Returns default on timeout or error."""
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(lambda: getattr(yf.Ticker(symbol), attr))
            return future.result(timeout=_TIMEOUT)
    except FuturesTimeout:
        log.warning(f"[yf] {symbol}.{attr} timed out after {_TIMEOUT}s")
        return default
    except Exception as e:
        log.debug(f"[yf] {symbol}.{attr} failed: {e}")
        return default


def run_with_timeout(func, default=None):
    """Run an arbitrary callable with a timeout. Returns default on timeout or error."""
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(func).result(timeout=_TIMEOUT)
    except FuturesTimeout:
        log.warning(f"[yf] call timed out after {_TIMEOUT}s")
        return default
    except Exception as e:
        log.debug(f"[yf] call failed: {e}")
        return default


def get_last_price(symbol: str) -> float | None:
    """Return the last price for symbol, or None on timeout/error."""
    fi = get_ticker_attr(symbol, "fast_info")
    if fi is None:
        return None
    try:
        return float(fi.last_price)
    except Exception:
        return None
