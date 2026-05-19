from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Optional

import logging

import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator

from tr_agent import yf_utils

log = logging.getLogger(__name__)

_SPY_CACHE: dict = {}  # keyed by date string; refreshes automatically each new day


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
    ml_confidence: Optional[float] = None
    ml_available: bool = False
    ml_features: dict = field(default_factory=dict)
    sma_200: Optional[float] = None
    intraday_trend: Optional[str] = None       # "up" / "down" / None if unavailable
    intraday_change_pct: Optional[float] = None  # % change from today's first bar to now
    headlines: list = field(default_factory=list)  # fetched once, reused by LLM stages

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
            "sma_200": round(self.sma_200, 4) if self.sma_200 is not None else None,
            "signal": self.signal.value,
            "reasoning": self.reasoning,
            "ml_confidence": round(self.ml_confidence, 4) if self.ml_confidence is not None else None,
            "ml_available": self.ml_available,
        }


def _fetch_intraday(ticker: str) -> pd.DataFrame:
    """Fetch 5 days of 15-min bars. Returns empty DataFrame on failure."""
    try:
        df = yf_utils.download(ticker, period="5d", interval="15m")
        return df if not df.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _intraday_trend(intraday_df: pd.DataFrame, ticker: str = "") -> tuple[Optional[str], Optional[float]]:
    """
    Compute recent price direction from 15-min bars.
    For equities: compares today's first bar to the latest bar.
    For crypto (24/7): uses the last 48 bars as a rolling window (~12 hours).
    """
    if intraday_df.empty:
        return None, None
    try:
        from tr_agent.assets import is_crypto
        close = intraday_df["Close"].squeeze()
        if is_crypto(ticker):
            if len(close) < 2:
                return None, None
            window = min(48, len(close))
            first = float(close.iloc[-window])
            last = float(close.iloc[-1])
        else:
            idx = intraday_df.index
            today = date.today()
            today_mask = idx.date == today
            if today_mask.sum() < 2:
                return None, None
            first = float(close[today_mask].iloc[0])
            last = float(close[today_mask].iloc[-1])
        change_pct = round((last - first) / first * 100, 2)
        trend = "up" if change_pct >= 0 else "down"
        return trend, change_pct
    except Exception:
        return None, None


def analyze(ticker: str, timeframe: str = "1y") -> TechnicalAnalysis:
    """
    Downloads daily OHLCV and computes RSI, MACD, SMAs — all from daily bars so
    signals stay consistent with how the ML model was trained.

    Intraday 15-min bars are fetched separately and used only to compute today's
    price trend (up/down vs open), which is injected into the LLM prompt as context.
    """
    df = yf_utils.download(ticker, period=timeframe, interval="1d")
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

    sma_200_series = SMAIndicator(close=close, window=200).sma_indicator()
    sma_200_val = None if pd.isna(sma_200_series.iloc[-1]) else float(sma_200_series.iloc[-1])

    signal, reasoning = _derive_signal(rsi_val, macd_hist_val, sma_20, sma_50, float(close.iloc[-1]))

    # Fetch headlines once; reused by both ML enrichment and the LLM confirmation prompt
    from tr_agent.news import fetch_news
    headlines = fetch_news(ticker)

    ml_confidence, ml_available, ml_feats = _enrich_with_ml(df, ticker=ticker, headlines=headlines)

    # Intraday context — direction only, not used in signal logic
    intraday_df = _fetch_intraday(ticker)
    trend, change_pct = _intraday_trend(intraday_df, ticker=ticker)
    if trend:
        log.info(f"[Signal] {ticker}: intraday trend {trend} ({change_pct:+.2f}% from open)")

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
        sma_200=sma_200_val,
        signal=signal,
        reasoning=reasoning,
        ml_confidence=ml_confidence,
        ml_available=ml_available,
        ml_features=ml_feats,
        intraday_trend=trend,
        intraday_change_pct=change_pct,
        headlines=headlines,
    )


def _get_spy_df() -> Optional[pd.DataFrame]:
    """Return today's SPY OHLCV, fetching once and caching for the rest of the day."""
    today = date.today().isoformat()
    if today not in _SPY_CACHE:
        try:
            spy = yf_utils.download("SPY", period="1y", interval="1d")
            if not spy.empty:
                _SPY_CACHE.clear()
                _SPY_CACHE[today] = spy
        except Exception:
            pass
    return _SPY_CACHE.get(today)


def _fetch_alternative_features(ticker: str, headlines: list[dict]) -> dict:
    """
    Compute live alternative-data features that are unavailable for historical dates.
    Never raises — returns neutral defaults on any error.
    """
    from tr_agent.news import get_sentiment_score
    result = {"news_sentiment": 0.0, "iv_rank": 0.0, "put_call_ratio": 1.0, "short_ratio": 0.0}
    try:
        result["news_sentiment"] = get_sentiment_score(headlines)
    except Exception:
        pass
    try:
        t = yf_utils.ticker(ticker)
        info = t.info or {}
        result["short_ratio"] = float(info.get("shortRatio") or 0.0)
    except Exception:
        pass
    try:
        t = yf_utils.ticker(ticker)
        expiries = t.options
        if expiries:
            chain = t.option_chain(expiries[0])
            calls, puts = chain.calls, chain.puts
            call_vol = float(calls["volume"].fillna(0).sum())
            put_vol  = float(puts["volume"].fillna(0).sum())
            result["put_call_ratio"] = round(put_vol / call_vol, 4) if call_vol > 0 else 1.0
            # IV rank: ATM call IV as a simple proxy (normalised to [0, 1] by dividing by 2.0)
            if not calls.empty:
                current_price = float(t.history(period="1d")["Close"].iloc[-1])
                atm_calls = calls.iloc[(calls["strike"] - current_price).abs().argsort()[:1]]
                raw_iv = float(atm_calls["impliedVolatility"].iloc[0]) if not atm_calls.empty else 0.0
                result["iv_rank"] = round(min(raw_iv / 2.0, 1.0), 4)
    except Exception:
        pass
    return result


def _enrich_with_ml(df, ticker: str = "", headlines: list | None = None) -> tuple[Optional[float], bool, dict]:
    """Compute ML feature vector and model confidence. Never raises."""
    try:
        from pathlib import Path
        from tr_agent.assets import is_crypto
        from tr_agent.ml.features import compute_last_row
        from tr_agent.ml.signal_model import SignalModel

        spy_df = _get_spy_df()
        ml_feats = compute_last_row(df, spy_df=spy_df)

        # Overlay live alternative-data features
        alt = _fetch_alternative_features(ticker, headlines or [])
        ml_feats.update(alt)
        log.debug(
            f"[Signal] {ticker}: news_sentiment={alt['news_sentiment']:.2f} "
            f"iv_rank={alt['iv_rank']:.2f} put_call={alt['put_call_ratio']:.2f} "
            f"short_ratio={alt['short_ratio']:.1f}"
        )

        model_name = "crypto_signal_model.pkl" if is_crypto(ticker) else "signal_model.pkl"
        model_path = Path(__file__).parents[3] / "data" / "models" / model_name
        model = SignalModel.load(model_path)
        if model is None:
            return None, False, ml_feats
        confidence = model.predict_proba(ml_feats)
        return confidence, confidence is not None, ml_feats
    except Exception:
        return None, False, {}


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
        if rsi < 30:
            buy_conditions.append(f"RSI={rsi:.1f} sobrevendido (<30)")
        elif rsi > 70:
            sell_conditions.append(f"RSI={rsi:.1f} sobrecomprado (>70)")

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
