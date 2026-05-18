from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock

import pandas as pd

from tr_agent.guards import is_earnings_blackout


def _make_calendar(earnings_date: datetime) -> dict:
    return {"Earnings Date": [pd.Timestamp(earnings_date)]}


def test_returns_true_when_earnings_tomorrow():
    tomorrow = datetime.now(timezone.utc) + timedelta(days=1)
    cal = _make_calendar(tomorrow)
    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.calendar = cal
        assert is_earnings_blackout("AAPL", days_before=3) is True


def test_returns_true_when_earnings_today():
    today = datetime.now(timezone.utc)
    cal = _make_calendar(today)
    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.calendar = cal
        assert is_earnings_blackout("AAPL", days_before=3) is True


def test_returns_true_when_earnings_yesterday_within_days_after():
    # Use 12 hours ago (well within days_after=1) to avoid timedelta.days truncation boundary
    twelve_hours_ago = datetime.now(timezone.utc) - timedelta(hours=12)
    cal = _make_calendar(twelve_hours_ago)
    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.calendar = cal
        assert is_earnings_blackout("AAPL", days_before=3, days_after=1) is True


def test_returns_false_when_earnings_far_away():
    far_future = datetime.now(timezone.utc) + timedelta(days=30)
    cal = _make_calendar(far_future)
    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.calendar = cal
        assert is_earnings_blackout("AAPL", days_before=3) is False


def test_returns_false_when_no_calendar():
    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.calendar = None
        assert is_earnings_blackout("AAPL") is False


def test_returns_false_when_empty_dates():
    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.calendar = {"Earnings Date": []}
        assert is_earnings_blackout("AAPL") is False


def test_returns_false_on_exception():
    with patch("yfinance.Ticker", side_effect=RuntimeError("network error")):
        assert is_earnings_blackout("AAPL") is False


def test_returns_false_when_earnings_just_outside_window():
    # 5 days away, window is 3 — should not blackout (uses safe margin to avoid truncation edge)
    five_days = datetime.now(timezone.utc) + timedelta(days=5)
    cal = _make_calendar(five_days)
    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.calendar = cal
        assert is_earnings_blackout("AAPL", days_before=3) is False


def test_returns_false_when_past_earnings_outside_days_after():
    # 3 days ago, days_after=1 — should not blackout
    three_days_ago = datetime.now(timezone.utc) - timedelta(days=3)
    cal = _make_calendar(three_days_ago)
    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.calendar = cal
        assert is_earnings_blackout("AAPL", days_before=3, days_after=1) is False
