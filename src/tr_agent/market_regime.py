import logging
from dataclasses import dataclass

import yfinance as yf
from ta.trend import SMAIndicator

log = logging.getLogger(__name__)


@dataclass
class MarketRegime:
    bullish: bool
    sma20: float
    sma50: float
    source: str = "SPY"

    @property
    def label(self) -> str:
        return "BULLISH" if self.bullish else "BEARISH"


def get_regime(ticker: str = "SPY") -> MarketRegime:
    """Return current market regime based on SMA20 vs SMA50 of the given index ticker."""
    try:
        df = yf.download(ticker, period="3mo", interval="1d", progress=False, auto_adjust=True)
        if df.empty or len(df) < 50:
            log.warning(f"[Regime] Insufficient data for {ticker} — defaulting to bullish")
            return MarketRegime(bullish=True, sma20=0.0, sma50=0.0, source=ticker)

        close = df["Close"].squeeze()
        sma20 = float(SMAIndicator(close=close, window=20).sma_indicator().iloc[-1])
        sma50 = float(SMAIndicator(close=close, window=50).sma_indicator().iloc[-1])
        bullish = sma20 > sma50

        log.info(
            f"[Regime] {ticker} SMA20={sma20:.2f} SMA50={sma50:.2f} "
            f"→ {('BULLISH' if bullish else 'BEARISH')}"
        )
        return MarketRegime(bullish=bullish, sma20=sma20, sma50=sma50, source=ticker)

    except Exception as e:
        log.warning(f"[Regime] Failed to get regime for {ticker}: {e} — defaulting to bullish")
        return MarketRegime(bullish=True, sma20=0.0, sma50=0.0, source=ticker)
