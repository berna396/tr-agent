"""
Stub del broker de Trade Republic (iteración 2).
Implementará BaseBroker usando pytr para conectarse a la API no oficial de TR.
"""
from tr_agent.broker.base import BaseBroker, Order, OrderSide, OrderType, Portfolio, Quote
from typing import Optional


class TradeRepublicBroker(BaseBroker):
    def get_quote(self, ticker: str) -> Quote:
        raise NotImplementedError("TradeRepublicBroker estará disponible en iteración 2")

    def place_order(
        self,
        ticker: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
    ) -> Order:
        raise NotImplementedError("TradeRepublicBroker estará disponible en iteración 2")

    def cancel_order(self, order_id: str) -> bool:
        raise NotImplementedError("TradeRepublicBroker estará disponible en iteración 2")

    def get_portfolio(self) -> Portfolio:
        raise NotImplementedError("TradeRepublicBroker estará disponible en iteración 2")
