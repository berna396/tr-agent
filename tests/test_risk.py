from datetime import datetime, timezone

import pytest

from tr_agent.broker.base import OrderSide, Portfolio, Position, Quote
from tr_agent.risk import RiskCheck, evaluate
from tr_agent.signals.technical import Signal, TechnicalAnalysis


def _make_analysis(ticker="AAPL", signal=Signal.BUY, rsi=28.0) -> TechnicalAnalysis:
    return TechnicalAnalysis(
        ticker=ticker, timeframe="3mo", close=150.0,
        rsi=rsi, macd=0.1, macd_signal=0.05, macd_hist=0.05,
        sma_20=155.0, sma_50=145.0, signal=signal,
        reasoning="RSI oversold, MACD positive",
    )


def _make_quote(price=150.0) -> Quote:
    return Quote("AAPL", price, price * 0.9995, price * 1.0005, datetime.now(timezone.utc))


def _make_portfolio(cash=10_000.0, positions=None) -> Portfolio:
    return Portfolio(cash=cash, positions=positions or {}, orders=[])


class TestRiskBuy:
    def test_approved_with_sufficient_cash(self):
        result = evaluate(_make_analysis(), _make_portfolio(), _make_quote())
        assert result.approved
        assert result.side == OrderSide.BUY
        assert result.max_quantity == pytest.approx(10_000 * 0.20 / 150.0, rel=1e-3)

    def test_rejected_when_over_invested(self):
        # 7000 cash, 4000 invested → 4000/11000 = 36% → fine
        # 2000 cash, 9000 invested → 9000/11000 = 81% → rejected
        positions = {"MSFT": Position("MSFT", 60.0, 150.0)}  # cost = 9000
        p = _make_portfolio(cash=2_000.0, positions=positions)
        result = evaluate(_make_analysis(), p, _make_quote())
        assert not result.approved
        assert "invested" in result.reason.lower()

    def test_rejected_when_insufficient_cash(self):
        p = _make_portfolio(cash=0.50)
        result = evaluate(_make_analysis(), p, _make_quote())
        assert not result.approved

    def test_neutral_signal_rejected(self):
        result = evaluate(_make_analysis(signal=Signal.NEUTRAL), _make_portfolio(), _make_quote())
        assert not result.approved


class TestRiskSell:
    def test_approved_with_existing_position(self):
        positions = {"AAPL": Position("AAPL", 10.0, 140.0)}
        p = _make_portfolio(positions=positions)
        result = evaluate(_make_analysis(signal=Signal.SELL), p, _make_quote())
        assert result.approved
        assert result.side == OrderSide.SELL
        assert result.max_quantity == 10.0

    def test_rejected_without_position(self):
        result = evaluate(_make_analysis(signal=Signal.SELL), _make_portfolio(), _make_quote())
        assert not result.approved
        assert "no open position" in result.reason.lower()
