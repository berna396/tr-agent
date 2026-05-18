import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator, ROCIndicator
from ta.trend import MACD, SMAIndicator, ADXIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from typing import Optional

FEATURE_NAMES = [
    "rsi",
    "macd",
    "macd_hist",
    "macd_signal",
    "sma_ratio",
    "roc_5",
    "roc_10",
    "roc_20",
    "atr_ratio",
    "bb_position",
    "bb_width",
    "volume_ratio",
    "adx",
    "day_of_week",
    "rel_roc_5",    # ticker 5-day ROC minus SPY 5-day ROC (excess return vs market)
    "spy_corr_60",  # 60-day rolling return correlation with SPY (market dependency)
]


def compute_all_rows(df: pd.DataFrame, spy_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Compute all ML features for every row in an OHLCV DataFrame."""
    if len(df) < 60:
        return pd.DataFrame(columns=FEATURE_NAMES)
    close = df["Close"].squeeze()
    high = df["High"].squeeze()
    low = df["Low"].squeeze()
    volume = df["Volume"].squeeze().replace(0, np.nan)

    rsi = RSIIndicator(close=close, window=14).rsi()

    macd_ind = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    macd_line = macd_ind.macd()
    macd_hist = macd_ind.macd_diff()
    macd_signal = macd_ind.macd_signal()

    sma_20 = SMAIndicator(close=close, window=20).sma_indicator()
    sma_50 = SMAIndicator(close=close, window=50).sma_indicator()
    sma_ratio = sma_20 / sma_50

    roc_5 = ROCIndicator(close=close, window=5).roc()
    roc_10 = ROCIndicator(close=close, window=10).roc()
    roc_20 = ROCIndicator(close=close, window=20).roc()

    atr = AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()
    atr_ratio = atr / close

    bb = BollingerBands(close=close, window=20, window_dev=2)
    bb_upper = bb.bollinger_hband()
    bb_lower = bb.bollinger_lband()
    bb_mid = bb.bollinger_mavg()
    bb_range = (bb_upper - bb_lower).replace(0, np.nan)
    bb_position = (close - bb_lower) / bb_range
    bb_width = (bb_upper - bb_lower) / bb_mid

    vol_ma = volume.rolling(20).mean()
    volume_ratio = (volume / vol_ma).fillna(1.0)

    adx = ADXIndicator(high=high, low=low, close=close, window=14).adx()

    if hasattr(df.index, "dayofweek"):
        dow = pd.Series(df.index.dayofweek, index=df.index, dtype=float)
    else:
        dow = pd.Series(0.0, index=df.index)

    # SPY-relative features
    if spy_df is not None and not spy_df.empty:
        spy_close = spy_df["Close"].squeeze()
        spy_close_aligned = spy_close.reindex(df.index).ffill().bfill()
        spy_roc_5 = ROCIndicator(close=spy_close_aligned, window=5).roc()
        rel_roc_5 = roc_5 - spy_roc_5

        ticker_returns = close.pct_change()
        spy_returns = spy_close_aligned.pct_change()
        spy_corr_60 = ticker_returns.rolling(60).corr(spy_returns).fillna(0.0)
    else:
        rel_roc_5 = pd.Series(0.0, index=df.index)
        spy_corr_60 = pd.Series(0.0, index=df.index)

    result = pd.DataFrame(
        {
            "rsi": rsi,
            "macd": macd_line,
            "macd_hist": macd_hist,
            "macd_signal": macd_signal,
            "sma_ratio": sma_ratio,
            "roc_5": roc_5,
            "roc_10": roc_10,
            "roc_20": roc_20,
            "atr_ratio": atr_ratio,
            "bb_position": bb_position,
            "bb_width": bb_width,
            "volume_ratio": volume_ratio,
            "adx": adx,
            "day_of_week": dow,
            "rel_roc_5": rel_roc_5,
            "spy_corr_60": spy_corr_60,
        },
        index=df.index,
    )
    return result.dropna()


def compute_last_row(df: pd.DataFrame, spy_df: Optional[pd.DataFrame] = None) -> dict:
    """Return the feature vector for the most recent row."""
    feat_df = compute_all_rows(df, spy_df=spy_df)
    if feat_df.empty:
        return {f: 0.0 for f in FEATURE_NAMES}
    return {k: float(v) for k, v in feat_df.iloc[-1].items()}
