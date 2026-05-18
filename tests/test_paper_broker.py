from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from tr_agent.broker.base import Order, OrderSide, OrderType, Quote
from tr_agent.broker.paper import PaperBroker
from tr_agent.portfolio.tracker import PortfolioTracker


def _make_quote(ticker: str, price: float) -> Quote:
    return Quote(
        ticker=ticker,
        price=price,
        bid=price * 0.9995,
        ask=price * 1.0005,
        timestamp=datetime.now(timezone.utc),
    )


@pytest.fixture
def broker(tmp_path):
    """PaperBroker aislado: usa directorio temporal para no compartir estado entre tests."""
    return PaperBroker(
        initial_capital=10_000.0,
        slippage=0.001,
        state_path=tmp_path / "portfolio_state.json",
    )


def test_initial_portfolio(broker):
    p = broker.get_portfolio()
    assert p.cash == 10_000.0
    assert p.positions == {}
    assert p.orders == []


def test_buy_order(broker):
    with patch.object(broker, "get_quote", return_value=_make_quote("AAPL", 100.0)):
        order = broker.place_order("AAPL", OrderSide.BUY, 5.0)

    assert order.status == "filled"
    assert order.ticker == "AAPL"
    assert order.side == OrderSide.BUY
    assert order.fill_price == pytest.approx(100.0 * 1.0005 * 1.001, rel=1e-3)

    p = broker.get_portfolio()
    assert "AAPL" in p.positions
    assert p.positions["AAPL"].quantity == 5.0
    assert p.cash < 10_000.0


def test_sell_order(broker):
    with patch.object(broker, "get_quote", return_value=_make_quote("AAPL", 100.0)):
        broker.place_order("AAPL", OrderSide.BUY, 5.0)

    with patch.object(broker, "get_quote", return_value=_make_quote("AAPL", 110.0)):
        order = broker.place_order("AAPL", OrderSide.SELL, 5.0)

    assert order.status == "filled"
    p = broker.get_portfolio()
    assert "AAPL" not in p.positions
    assert p.cash > 10_000.0  # ganamos dinero


def test_insufficient_capital(broker):
    with patch.object(broker, "get_quote", return_value=_make_quote("AAPL", 100.0)):
        with pytest.raises(ValueError, match="Capital insuficiente"):
            broker.place_order("AAPL", OrderSide.BUY, 10_000.0)


def test_sell_without_position(broker):
    with patch.object(broker, "get_quote", return_value=_make_quote("AAPL", 100.0)):
        with pytest.raises(ValueError, match="No tienes posición"):
            broker.place_order("AAPL", OrderSide.SELL, 1.0)


def test_cancel_order_returns_false(broker):
    assert broker.cancel_order("any-id") is False


def test_portfolio_tracker_avg_price():
    tracker = PortfolioTracker(initial_capital=10_000.0)
    now = datetime.now(timezone.utc)

    order1 = Order("id1", "AAPL", OrderSide.BUY, OrderType.MARKET, 10.0, 100.0, "filled", now)
    order2 = Order("id2", "AAPL", OrderSide.BUY, OrderType.MARKET, 10.0, 120.0, "filled", now)
    tracker.execute(order1)
    tracker.execute(order2)

    pos = tracker.get_portfolio().positions["AAPL"]
    assert pos.quantity == 20.0
    assert pos.avg_price == pytest.approx(110.0)
