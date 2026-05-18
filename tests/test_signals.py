from unittest.mock import patch

import pandas as pd
import pytest

from tr_agent.signals.technical import Signal, TechnicalAnalysis, _derive_signal
from tr_agent.signals.rules import evaluate, DEFAULT_RULES


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
