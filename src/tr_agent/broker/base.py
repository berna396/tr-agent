from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


@dataclass
class Quote:
    ticker: str
    price: float
    bid: float
    ask: float
    timestamp: datetime


@dataclass
class Order:
    order_id: str
    ticker: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    fill_price: Optional[float]
    status: str  # "filled", "pending", "cancelled", "rejected"
    timestamp: datetime


@dataclass
class Position:
    ticker: str
    quantity: float
    avg_price: float
    stop_price: Optional[float] = None  # ATR-based stop; None → fall back to fixed %

    @property
    def cost_basis(self) -> float:
        return self.quantity * self.avg_price


@dataclass
class Portfolio:
    cash: float
    positions: dict[str, Position]
    orders: list[Order]

    @property
    def total_cost_basis(self) -> float:
        return sum(p.cost_basis for p in self.positions.values())


class BaseBroker(ABC):
    @abstractmethod
    def get_quote(self, ticker: str) -> Quote:
        """Devuelve precio actual del activo."""

    @abstractmethod
    def place_order(
        self,
        ticker: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
    ) -> Order:
        """Envía una orden. En paper mode la simula; en live la envía al broker."""

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancela una orden pendiente. Devuelve True si fue cancelada."""

    @abstractmethod
    def get_portfolio(self) -> Portfolio:
        """Devuelve el estado actual del portfolio."""
