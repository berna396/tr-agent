from dataclasses import dataclass
from enum import Enum
from typing import Optional

import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator


class Signal(str, Enum):
    BUY = "buy"
    SELL = "sell"
    NEUTRAL = "neutral"


@dataclass
class TechnicalAnalysis:
    ticker: str
    timeframe: str
    close: float
    rsi: Optional[float]
    macd: Optional[float]
    macd_signal: Optional[float]
    macd_hist: Optional[float]
    sma_20: Optional[float]
    sma_50: Optional[float]
    signal: Signal
    reasoning: str

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "timeframe": self.timeframe,
            "close": round(self.close, 4),
            "rsi": round(self.rsi, 2) if self.rsi is not None else None,
            "macd": round(self.macd, 4) if self.macd is not None else None,
            "macd_signal": round(self.macd_signal, 4) if self.macd_signal is not None else None,
            "macd_hist": round(self.macd_hist, 4) if self.macd_hist is not None else None,
            "sma_20": round(self.sma_20, 4) if self.sma_20 is not None else None,
            "sma_50": round(self.sma_50, 4) if self.sma_50 is not None else None,
            "signal": self.signal.value,
            "reasoning": self.reasoning,
        }


def analyze(ticker: str, timeframe: str = "3mo") -> TechnicalAnalysis:
    """
    Descarga histórico diario de `ticker` (vía yfinance) y calcula RSI, MACD y SMAs.

    timeframe: periodo de descarga válido para yfinance (1mo, 3mo, 6mo, 1y).
    Necesita al menos 50 velas para calcular SMA50.
    """
    df = yf.download(ticker, period=timeframe, interval="1d", progress=False, auto_adjust=True)
    if df.empty or len(df) < 50:
        raise ValueError(f"No hay suficientes datos para {ticker} (timeframe={timeframe})")

    close = df["Close"].squeeze()

    rsi_val = float(RSIIndicator(close=close, window=14).rsi().iloc[-1])

    macd_ind = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    macd_val = float(macd_ind.macd().iloc[-1])
    macd_signal_val = float(macd_ind.macd_signal().iloc[-1])
    macd_hist_val = float(macd_ind.macd_diff().iloc[-1])

    sma_20 = float(SMAIndicator(close=close, window=20).sma_indicator().iloc[-1])
    sma_50 = float(SMAIndicator(close=close, window=50).sma_indicator().iloc[-1])

    signal, reasoning = _derive_signal(rsi_val, macd_hist_val, sma_20, sma_50, float(close.iloc[-1]))

    return TechnicalAnalysis(
        ticker=ticker,
        timeframe=timeframe,
        close=float(close.iloc[-1]),
        rsi=rsi_val,
        macd=macd_val,
        macd_signal=macd_signal_val,
        macd_hist=macd_hist_val,
        sma_20=sma_20,
        sma_50=sma_50,
        signal=signal,
        reasoning=reasoning,
    )


def _derive_signal(
    rsi: Optional[float],
    macd_hist: Optional[float],
    sma_20: Optional[float],
    sma_50: Optional[float],
    close: float,
) -> tuple[Signal, str]:
    """Genera señal cuando al menos 2 de 3 indicadores apuntan en la misma dirección."""
    buy_conditions = []
    sell_conditions = []

    if rsi is not None:
        if rsi < 35:
            buy_conditions.append(f"RSI={rsi:.1f} sobrevendido (<35)")
        elif rsi > 65:
            sell_conditions.append(f"RSI={rsi:.1f} sobrecomprado (>65)")

    if macd_hist is not None:
        if macd_hist > 0:
            buy_conditions.append(f"MACD histograma positivo ({macd_hist:.4f})")
        else:
            sell_conditions.append(f"MACD histograma negativo ({macd_hist:.4f})")

    if sma_20 is not None and sma_50 is not None:
        if sma_20 > sma_50:
            buy_conditions.append(f"SMA20({sma_20:.2f}) > SMA50({sma_50:.2f}) — tendencia alcista")
        elif sma_20 < sma_50:
            sell_conditions.append(f"SMA20({sma_20:.2f}) < SMA50({sma_50:.2f}) — tendencia bajista")

    if len(buy_conditions) >= 2:
        return Signal.BUY, "Señal de COMPRA: " + "; ".join(buy_conditions)
    elif len(sell_conditions) >= 2:
        return Signal.SELL, "Señal de VENTA: " + "; ".join(sell_conditions)
    else:
        all_conditions = buy_conditions + sell_conditions
        return Signal.NEUTRAL, "Sin señal clara: " + ("; ".join(all_conditions) or "indicadores mixtos")
