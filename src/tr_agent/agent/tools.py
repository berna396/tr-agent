from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from tr_agent.broker.base import BaseBroker, OrderSide, OrderType
from tr_agent.signals import technical


# El broker se inyecta en el momento de crear el agente.
# Las tools capturan la instancia via closure.
def make_tools(broker: BaseBroker) -> list:

    class GetQuoteInput(BaseModel):
        ticker: str = Field(description="Ticker del activo (ej: AAPL, MSFT, GOOGL)")

    @tool(args_schema=GetQuoteInput)
    def get_quote(ticker: str) -> dict:
        """Devuelve el precio actual de un activo: precio, bid, ask y timestamp."""
        try:
            quote = broker.get_quote(ticker.upper())
            return {
                "ticker": quote.ticker,
                "price": quote.price,
                "bid": quote.bid,
                "ask": quote.ask,
                "timestamp": quote.timestamp.isoformat(),
            }
        except Exception as e:
            return {"error": str(e)}

    class AnalyzeTechnicalsInput(BaseModel):
        ticker: str = Field(description="Ticker del activo")
        timeframe: str = Field(
            default="3mo",
            description="Periodo de análisis: 1mo, 3mo, 6mo, 1y",
        )

    @tool(args_schema=AnalyzeTechnicalsInput)
    def analyze_technicals(ticker: str, timeframe: str = "3mo") -> dict:
        """Calcula RSI, MACD y medias móviles (SMA20/SMA50) para un activo."""
        try:
            analysis = technical.analyze(ticker.upper(), timeframe)
            return analysis.to_dict()
        except Exception as e:
            return {"error": str(e)}

    @tool
    def get_portfolio() -> dict:
        """Devuelve el estado actual del portfolio: efectivo, posiciones abiertas y órdenes recientes."""
        try:
            portfolio = broker.get_portfolio()
            return {
                "cash": round(portfolio.cash, 2),
                "positions": {
                    ticker: {
                        "quantity": pos.quantity,
                        "avg_price": round(pos.avg_price, 4),
                        "cost_basis": round(pos.cost_basis, 2),
                    }
                    for ticker, pos in portfolio.positions.items()
                },
                "num_orders": len(portfolio.orders),
            }
        except Exception as e:
            return {"error": str(e)}

    class PlaceOrderInput(BaseModel):
        ticker: str = Field(description="Ticker del activo")
        side: str = Field(description="'buy' para comprar, 'sell' para vender")
        quantity: float = Field(description="Número de acciones a operar (puede ser decimal)")
        order_type: str = Field(default="market", description="Tipo de orden: 'market' o 'limit'")
        limit_price: Optional[float] = Field(
            default=None, description="Precio límite (solo para order_type='limit')"
        )

    @tool(args_schema=PlaceOrderInput)
    def place_order(
        ticker: str,
        side: str,
        quantity: float,
        order_type: str = "market",
        limit_price: Optional[float] = None,
    ) -> dict:
        """Ejecuta una orden de compra o venta. En paper mode simula el fill sin dinero real."""
        try:
            order = broker.place_order(
                ticker=ticker.upper(),
                side=OrderSide(side.lower()),
                quantity=quantity,
                order_type=OrderType(order_type.lower()),
                limit_price=limit_price,
            )
            return {
                "order_id": order.order_id,
                "ticker": order.ticker,
                "side": order.side.value,
                "quantity": order.quantity,
                "fill_price": order.fill_price,
                "status": order.status,
                "timestamp": order.timestamp.isoformat(),
            }
        except Exception as e:
            return {"error": str(e)}

    class SearchInstrumentsInput(BaseModel):
        query: str = Field(description="Nombre o ticker de la empresa a buscar")

    @tool(args_schema=SearchInstrumentsInput)
    def search_instruments(query: str) -> dict:
        """Busca instrumentos financieros por nombre o ticker usando yfinance."""
        import yfinance as yf
        try:
            results = yf.Search(query, max_results=5)
            quotes = results.quotes
            if not quotes:
                return {"results": [], "message": f"No se encontraron resultados para '{query}'"}
            return {
                "results": [
                    {
                        "symbol": q.get("symbol", ""),
                        "name": q.get("longname") or q.get("shortname", ""),
                        "exchange": q.get("exchange", ""),
                        "type": q.get("quoteType", ""),
                    }
                    for q in quotes
                ]
            }
        except Exception as e:
            return {"error": str(e)}

    return [get_quote, analyze_technicals, get_portfolio, place_order, search_instruments]
