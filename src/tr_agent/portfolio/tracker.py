from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from tr_agent.broker.base import Order, OrderSide, Portfolio, Position


@dataclass
class TradeRecord:
    order_id: str
    ticker: str
    side: OrderSide
    quantity: float
    fill_price: float
    timestamp: datetime
    pnl: Optional[float] = None  # solo para ventas


class PortfolioTracker:
    def __init__(self, initial_capital: float):
        self._cash = initial_capital
        self._initial_capital = initial_capital
        self._positions: dict[str, Position] = {}
        self._orders: list[Order] = []
        self._trade_log: list[TradeRecord] = []

    def execute(self, order: Order, stop_price: Optional[float] = None) -> None:
        """Aplica una orden completada al estado del portfolio."""
        if order.fill_price is None:
            raise ValueError(f"La orden {order.order_id} no tiene fill_price")

        cost = order.fill_price * order.quantity

        if order.side == OrderSide.BUY:
            if cost > self._cash:
                raise ValueError(
                    f"Capital insuficiente: necesitas {cost:.2f}, tienes {self._cash:.2f}"
                )
            self._cash -= cost
            if order.ticker in self._positions:
                pos = self._positions[order.ticker]
                total_qty = pos.quantity + order.quantity
                avg_price = (pos.cost_basis + cost) / total_qty
                self._positions[order.ticker] = Position(
                    order.ticker, total_qty, avg_price, stop_price=stop_price
                )
            else:
                self._positions[order.ticker] = Position(
                    order.ticker, order.quantity, order.fill_price, stop_price=stop_price
                )
            pnl = None

        else:  # SELL
            if order.ticker not in self._positions:
                raise ValueError(f"No tienes posición en {order.ticker}")
            pos = self._positions[order.ticker]
            if order.quantity > pos.quantity:
                raise ValueError(
                    f"Intentas vender {order.quantity} pero solo tienes {pos.quantity}"
                )
            pnl = (order.fill_price - pos.avg_price) * order.quantity
            self._cash += cost
            remaining = pos.quantity - order.quantity
            if remaining == 0:
                del self._positions[order.ticker]
            else:
                self._positions[order.ticker] = Position(order.ticker, remaining, pos.avg_price)

        self._orders.append(order)
        self._trade_log.append(
            TradeRecord(
                order_id=order.order_id,
                ticker=order.ticker,
                side=order.side,
                quantity=order.quantity,
                fill_price=order.fill_price,
                timestamp=order.timestamp,
                pnl=pnl,
            )
        )

    def get_portfolio(self) -> Portfolio:
        return Portfolio(
            cash=self._cash,
            positions=dict(self._positions),
            orders=list(self._orders),
        )

    def get_metrics(self, current_prices: dict[str, float]) -> dict:
        """Calcula P&L no realizado y métricas básicas del portfolio."""
        unrealized_pnl = sum(
            (current_prices.get(ticker, pos.avg_price) - pos.avg_price) * pos.quantity
            for ticker, pos in self._positions.items()
        )
        realized_pnl = sum(r.pnl for r in self._trade_log if r.pnl is not None)
        market_value = sum(
            current_prices.get(ticker, pos.avg_price) * pos.quantity
            for ticker, pos in self._positions.items()
        )
        total_value = self._cash + market_value
        return {
            "cash": round(self._cash, 2),
            "market_value": round(market_value, 2),
            "total_value": round(total_value, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "realized_pnl": round(realized_pnl, 2),
            "total_return_pct": round(
                (total_value - self._initial_capital) / self._initial_capital * 100, 2
            ),
            "num_trades": len(self._trade_log),
        }

    def to_dict(self) -> dict:
        return {
            "cash": self._cash,
            "initial_capital": self._initial_capital,
            "positions": {
                ticker: {
                    "quantity": pos.quantity,
                    "avg_price": pos.avg_price,
                    "stop_price": pos.stop_price,
                }
                for ticker, pos in self._positions.items()
            },
            "trade_log": [
                {
                    "order_id": r.order_id,
                    "ticker": r.ticker,
                    "side": r.side.value,
                    "quantity": r.quantity,
                    "fill_price": r.fill_price,
                    "timestamp": r.timestamp.isoformat(),
                    "pnl": r.pnl,
                }
                for r in self._trade_log
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PortfolioTracker":
        tracker = cls(initial_capital=data["initial_capital"])
        tracker._cash = data["cash"]
        for ticker, pos_data in data.get("positions", {}).items():
            tracker._positions[ticker] = Position(
                ticker=ticker,
                quantity=pos_data["quantity"],
                avg_price=pos_data["avg_price"],
                stop_price=pos_data.get("stop_price"),
            )
        return tracker
