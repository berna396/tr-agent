from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

from tr_agent.market_regime import get_regime, MarketRegime


def _make_spy_df(sma20_above_sma50: bool = True) -> pd.DataFrame:
    """Create a fake SPY DataFrame with 60 rows so SMA50 can compute."""
    n = 60
    if sma20_above_sma50:
        # Price trending up — SMA20 > SMA50
        prices = np.linspace(400, 440, n)
    else:
        # Price trending down — SMA20 < SMA50
        prices = np.linspace(440, 400, n)
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    return pd.DataFrame({"Close": prices, "Volume": 1e7}, index=idx)


def test_bullish_regime_when_sma20_above_sma50():
    df = _make_spy_df(sma20_above_sma50=True)
    with patch("yfinance.download", return_value=df):
        regime = get_regime()
    assert regime.bullish is True
    assert regime.label == "BULLISH"
    assert regime.sma20 > regime.sma50


def test_bearish_regime_when_sma20_below_sma50():
    df = _make_spy_df(sma20_above_sma50=False)
    with patch("yfinance.download", return_value=df):
        regime = get_regime()
    assert regime.bullish is False
    assert regime.label == "BEARISH"
    assert regime.sma20 < regime.sma50


def test_defaults_to_bullish_on_empty_df():
    with patch("yfinance.download", return_value=pd.DataFrame()):
        regime = get_regime()
    assert regime.bullish is True
    assert regime.sma20 == 0.0
    assert regime.sma50 == 0.0


def test_defaults_to_bullish_on_short_df():
    short_df = _make_spy_df()[:30]  # only 30 rows — not enough for SMA50
    with patch("yfinance.download", return_value=short_df):
        regime = get_regime()
    assert regime.bullish is True


def test_defaults_to_bullish_on_exception():
    with patch("yfinance.download", side_effect=RuntimeError("network error")):
        regime = get_regime()
    assert regime.bullish is True


def test_source_field_reflects_ticker():
    df = _make_spy_df()
    with patch("yfinance.download", return_value=df):
        regime = get_regime(ticker="QQQ")
    assert regime.source == "QQQ"


def test_market_regime_dataclass_fields():
    regime = MarketRegime(bullish=True, sma20=450.0, sma50=445.0)
    assert regime.source == "SPY"  # default
    assert regime.label == "BULLISH"
