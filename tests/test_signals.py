from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from tr_agent.signals.technical import Signal, TechnicalAnalysis, _derive_signal, analyze
from tr_agent.signals.rules import evaluate, DEFAULT_RULES


def _make_daily_df(n: int = 260) -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    prices = np.linspace(100, 150, n)
    df = pd.DataFrame(
        {"Open": prices, "High": prices * 1.01, "Low": prices * 0.99, "Close": prices, "Volume": 1e6},
        index=idx,
    )
    df.columns = pd.MultiIndex.from_tuples([(c, "TEST") for c in df.columns])
    return df


def _make_intraday_df(n: int = 100) -> pd.DataFrame:
    idx = pd.date_range("2024-01-02 09:30", periods=n, freq="15min")
    prices = np.linspace(148, 155, n)
    df = pd.DataFrame(
        {"Open": prices, "High": prices * 1.002, "Low": prices * 0.998, "Close": prices, "Volume": 5e4},
        index=idx,
    )
    df.columns = pd.MultiIndex.from_tuples([(c, "TEST") for c in df.columns])
    return df


def _make_analysis(**kwargs) -> TechnicalAnalysis:
    defaults = dict(
        ticker="TEST",
        timeframe="3mo",
        close=100.0,
        rsi=50.0,
        macd=0.1,
        macd_signal=0.05,
        macd_hist=0.05,
        sma_20=102.0,
        sma_50=98.0,
        signal=Signal.NEUTRAL,
        reasoning="",
    )
    defaults.update(kwargs)
    return TechnicalAnalysis(**defaults)


class TestDeriveSignal:
    def test_buy_signal_rsi_and_macd_and_sma(self):
        sig, reason = _derive_signal(rsi=25.0, macd_hist=0.1, sma_20=105.0, sma_50=100.0, close=106.0)
        assert sig == Signal.BUY

    def test_sell_signal_rsi_and_sma(self):
        sig, reason = _derive_signal(rsi=72.0, macd_hist=-0.05, sma_20=95.0, sma_50=100.0, close=94.0)
        assert sig == Signal.SELL

    def test_neutral_when_mixed(self):
        # RSI buy, MACD sell, SMA neutral → solo 1 condición de cada lado
        sig, reason = _derive_signal(rsi=28.0, macd_hist=-0.01, sma_20=100.0, sma_50=100.0, close=100.0)
        assert sig == Signal.NEUTRAL

    def test_neutral_when_rsi_normal(self):
        sig, reason = _derive_signal(rsi=50.0, macd_hist=0.01, sma_20=101.0, sma_50=100.0, close=101.0)
        # solo 2 condiciones alcistas → BUY
        assert sig == Signal.BUY

    def test_handles_none_values(self):
        sig, reason = _derive_signal(rsi=None, macd_hist=None, sma_20=None, sma_50=None, close=100.0)
        assert sig == Signal.NEUTRAL


class TestIntradayFallback:
    def test_uses_intraday_when_available(self):
        daily = _make_daily_df()
        intraday = _make_intraday_df(60)

        def mock_download(ticker, period, interval, **kw):
            return intraday if interval == "15m" else daily

        with patch("tr_agent.signals.technical.yf.download", side_effect=mock_download), \
             patch("tr_agent.signals.technical._enrich_with_ml", return_value=(None, False, {})):
            result = analyze("TEST")

        # Intraday close is ~155, daily close ends ~150 — different values confirm intraday was used
        assert result.close == pytest.approx(float(intraday["Close"]["TEST"].iloc[-1]), rel=1e-3)

    def test_falls_back_to_daily_on_empty_intraday(self):
        daily = _make_daily_df()

        def mock_download(ticker, period, interval, **kw):
            return pd.DataFrame() if interval == "15m" else daily

        with patch("tr_agent.signals.technical.yf.download", side_effect=mock_download), \
             patch("tr_agent.signals.technical._enrich_with_ml", return_value=(None, False, {})):
            result = analyze("TEST")

        assert result.close == pytest.approx(float(daily["Close"]["TEST"].iloc[-1]), rel=1e-3)

    def test_falls_back_to_daily_on_insufficient_intraday(self):
        daily = _make_daily_df()
        intraday = _make_intraday_df(10)  # fewer than 30 rows

        def mock_download(ticker, period, interval, **kw):
            return intraday if interval == "15m" else daily

        with patch("tr_agent.signals.technical.yf.download", side_effect=mock_download), \
             patch("tr_agent.signals.technical._enrich_with_ml", return_value=(None, False, {})):
            result = analyze("TEST")

        assert result.close == pytest.approx(float(daily["Close"]["TEST"].iloc[-1]), rel=1e-3)


class TestRules:
    def test_oversold_triggers_buy(self):
        a = _make_analysis(rsi=25.0)
        triggered = evaluate(a, DEFAULT_RULES)
        names = [r.name for r in triggered]
        assert "oversold_buy" in names

    def test_overbought_triggers_sell(self):
        a = _make_analysis(rsi=75.0)
        triggered = evaluate(a, DEFAULT_RULES)
        names = [r.name for r in triggered]
        assert "overbought_sell" in names

    def test_golden_cross(self):
        a = _make_analysis(sma_20=110.0, sma_50=100.0)
        triggered = evaluate(a, DEFAULT_RULES)
        names = [r.name for r in triggered]
        assert "golden_cross_buy" in names

    def test_death_cross(self):
        a = _make_analysis(sma_20=90.0, sma_50=100.0)
        triggered = evaluate(a, DEFAULT_RULES)
        names = [r.name for r in triggered]
        assert "death_cross_sell" in names

    def test_no_rules_triggered(self):
        a = _make_analysis(rsi=50.0, sma_20=100.0, sma_50=100.0)
        # RSI normal, SMA igual → solo death_cross (sma_20 == sma_50 → NOT sma_20 > sma_50)
        triggered = evaluate(a, DEFAULT_RULES)
        names = [r.name for r in triggered]
        assert "oversold_buy" not in names
        assert "overbought_sell" not in names
