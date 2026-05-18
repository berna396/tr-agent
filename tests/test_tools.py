from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from tr_agent.broker.base import Order, OrderSide, OrderType, Portfolio, Quote
from tr_agent.agent.tools import make_tools


def _make_mock_broker(price: float = 150.0) -> MagicMock:
    broker = MagicMock()
    broker.get_quote.return_value = Quote(
        ticker="AAPL",
        price=price,
        bid=price * 0.9995,
        ask=price * 1.0005,
        timestamp=datetime.now(timezone.utc),
    )
    broker.get_portfolio.return_value = Portfolio(
        cash=9_500.0,
        positions={},
        orders=[],
    )
    broker.place_order.return_value = Order(
        order_id="abc123",
        ticker="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=5.0,
        fill_price=150.15,
        status="filled",
        timestamp=datetime.now(timezone.utc),
    )
    return broker


@pytest.fixture
def tools():
    broker = _make_mock_broker()
    return {t.name: t for t in make_tools(broker)}, broker


def test_get_quote_returns_price(tools):
    tool_map, broker = tools
    result = tool_map["get_quote"].invoke({"ticker": "AAPL"})
    assert "price" in result
    assert result["ticker"] == "AAPL"
    broker.get_quote.assert_called_once_with("AAPL")


def test_get_portfolio_returns_cash(tools):
    tool_map, _ = tools
    result = tool_map["get_portfolio"].invoke({})
    assert result["cash"] == 9_500.0
    assert "positions" in result


def test_place_order_buy(tools):
    tool_map, broker = tools
    result = tool_map["place_order"].invoke(
        {"ticker": "aapl", "side": "buy", "quantity": 5.0}
    )
    assert result["status"] == "filled"
    assert result["ticker"] == "AAPL"
    broker.place_order.assert_called_once()
    call_kwargs = broker.place_order.call_args
    assert call_kwargs.kwargs["side"] == OrderSide.BUY


def test_place_order_invalid_side(tools):
    tool_map, _ = tools
    result = tool_map["place_order"].invoke(
        {"ticker": "AAPL", "side": "hold", "quantity": 1.0}
    )
    assert "error" in result


def test_get_quote_propagates_error(tools):
    tool_map, broker = tools
    broker.get_quote.side_effect = Exception("conexión fallida")
    result = tool_map["get_quote"].invoke({"ticker": "AAPL"})
    assert "error" in result
    assert "conexión fallida" in result["error"]
