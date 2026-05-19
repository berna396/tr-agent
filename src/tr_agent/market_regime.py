import logging
from dataclasses import dataclass

from ta.trend import SMAIndicator
from tr_agent import yf_utils

log = logging.getLogger(__name__)


@dataclass
class MarketRegime:
    bullish: bool
    sma50: float
    sma200: float
    source: str = "SPY"

    @property
    def label(self) -> str:
        return "BULLISH" if self.bullish else "BEARISH"


def get_regime(ticker: str = "SPY") -> MarketRegime:
    """Return current market regime based on golden/death cross (SMA50 vs SMA200)."""
    try:
        df = yf_utils.download(ticker, period="1y", interval="1d")
        if df.empty or len(df) < 200:
            log.warning(f"[Regime] Insufficient data for {ticker} — defaulting to bullish")
            return MarketRegime(bullish=True, sma50=0.0, sma200=0.0, source=ticker)

        close = df["Close"].squeeze()
        sma50 = float(SMAIndicator(close=close, window=50).sma_indicator().iloc[-1])
        sma200 = float(SMAIndicator(close=close, window=200).sma_indicator().iloc[-1])
        bullish = sma50 > sma200

        log.info(
            f"[Regime] {ticker} SMA50={sma50:.2f} SMA200={sma200:.2f} "
            f"→ {('BULLISH' if bullish else 'BEARISH')}"
        )
        return MarketRegime(bullish=bullish, sma50=sma50, sma200=sma200, source=ticker)

    except Exception as e:
        log.warning(f"[Regime] Failed to get regime for {ticker}: {e} — defaulting to bullish")
        return MarketRegime(bullish=True, sma50=0.0, sma200=0.0, source=ticker)
