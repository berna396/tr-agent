import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from tr_agent import yf_utils
from tr_agent.broker.base import (
    BaseBroker,
    Order,
    OrderSide,
    OrderType,
    Portfolio,
    Quote,
)
from tr_agent.portfolio import persistence
from tr_agent.portfolio.tracker import PortfolioTracker


class PaperBroker(BaseBroker):
    def __init__(
        self,
        initial_capital: float,
        slippage: float = 0.001,
        state_path: Optional[Path] = None,
    ):
        self._slippage = slippage
        self._state_path = state_path or persistence._STATE_FILE
        self._tracker: PortfolioTracker = persistence.load(initial_capital, self._state_path)

    def get_quote(self, ticker: str) -> Quote:
        price = yf_utils.get_last_price(ticker)
        if price is None:
            raise ValueError(f"Could not fetch price for {ticker}")
        spread = price * 0.0005  # spread simulado de 0.05%
        return Quote(
            ticker=ticker,
            price=price,
            bid=round(price - spread, 4),
            ask=round(price + spread, 4),
            timestamp=datetime.now(timezone.utc),
        )

    def place_order(
        self,
        ticker: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> Order:
        quote = self.get_quote(ticker)

        if side == OrderSide.BUY:
            fill_price = round(quote.ask * (1 + self._slippage), 4)
        else:
            fill_price = round(quote.bid * (1 - self._slippage), 4)

        order = Order(
            order_id=str(uuid.uuid4())[:8],
            ticker=ticker,
            side=side,
            order_type=order_type,
            quantity=quantity,
            fill_price=fill_price,
            status="filled",
            timestamp=datetime.now(timezone.utc),
        )
        self._tracker.execute(order, stop_price=stop_price)
        persistence.save(self._tracker, self._state_path)
        return order

    def cancel_order(self, order_id: str) -> bool:
        # En paper mode las órdenes de mercado se rellenan inmediatamente;
        # no hay nada que cancelar.
        return False

    def get_portfolio(self) -> Portfolio:
        return self._tracker.get_portfolio()

    def get_metrics(self, tickers: Optional[list[str]] = None) -> dict:
        current_prices: dict[str, float] = {}
        for ticker in (tickers or list(self._tracker.get_portfolio().positions.keys())):
            try:
                p = yf_utils.get_last_price(ticker)
                if p is not None:
                    current_prices[ticker] = p
            except Exception:
                pass
        return self._tracker.get_metrics(current_prices)
