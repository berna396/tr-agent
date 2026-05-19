"""
yfinance wrappers with request timeouts.

All yf.download() and yf.Ticker() calls go through here so a stalled
Yahoo Finance response can never freeze the scheduler indefinitely.
"""

import functools
import logging

import requests
import yfinance as yf

log = logging.getLogger(__name__)

_TIMEOUT = 20  # seconds per request


def _session() -> requests.Session:
    """Requests session with a hard timeout on every call."""
    s = requests.Session()
    s.request = functools.partial(s.request, timeout=_TIMEOUT)
    return s


def download(ticker: str, **kwargs) -> "pd.DataFrame":
    """yf.download with a hard timeout. Returns empty DataFrame on failure."""
    import pandas as pd
    kwargs.setdefault("progress", False)
    kwargs.setdefault("auto_adjust", True)
    kwargs["timeout"] = _TIMEOUT
    try:
        return yf.download(ticker, **kwargs)
    except Exception as e:
        log.warning(f"[yf] download({ticker}) failed: {e}")
        return pd.DataFrame()


def ticker(symbol: str) -> yf.Ticker:
    """yf.Ticker pre-wired with a timeout session."""
    return yf.Ticker(symbol, session=_session())
