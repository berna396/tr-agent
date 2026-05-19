import json
import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from tr_agent import yf_utils
from tr_agent.ml.features import FEATURE_NAMES, compute_all_rows

log = logging.getLogger(__name__)

# Multi-horizon label ensemble: majority vote across 3 horizons
_FORWARD_DAYS  = [5,     10,    20   ]   # horizons
_THRESHOLDS    = [0.005, 0.005, 0.010]   # minimum move per horizon
_WHIPSAW_DD    = 0.03                    # reject label if 5d forward drawdown > 3%


def _is_buy_signal(rsi: float, macd_hist: float, sma_ratio: float) -> bool:
    """Mirror the 2-of-3 rule from technical.py using feature values."""
    hits = 0
    if rsi < 30:
        hits += 1
    if macd_hist > 0:
        hits += 1
    if sma_ratio > 1.0:  # sma_20 > sma_50
        hits += 1
    return hits >= 2


def build_historical_dataset(
    tickers: list[str], period: str = "2y"
) -> tuple[pd.DataFrame, pd.Series]:
    """Download OHLCV history, compute features, label by 5-day forward return."""
    # Download SPY once for correlation features shared across all tickers
    spy_df = None
    try:
        spy_df = yf_utils.download("SPY", period=period, interval="1d")
        if spy_df.empty:
            spy_df = None
        else:
            log.info(f"[ML] SPY downloaded for correlation features ({len(spy_df)} rows)")
    except Exception as e:
        log.warning(f"[ML] Could not download SPY for correlation features: {e}")

    all_X, all_y = [], []

    for ticker in tickers:
        log.info(f"[ML] Bootstrapping {ticker} ({period})...")
        try:
            df = yf_utils.download(ticker, period=period, interval="1d")
            if df.empty or len(df) < 60:
                log.warning(f"[ML] Skipping {ticker} — insufficient data ({len(df)} rows)")
                continue

            feat_df = compute_all_rows(df, spy_df=spy_df)
            close = df["Close"].squeeze().reindex(feat_df.index)

            # Build multi-horizon forward returns
            combined = feat_df.copy()
            for days, thresh in zip(_FORWARD_DAYS, _THRESHOLDS):
                combined[f"_fwd_{days}"] = close.shift(-days) / close - 1
            combined = combined.dropna(subset=[f"_fwd_{d}" for d in _FORWARD_DAYS])

            # Filter to buy-signal days
            signal_mask = combined.apply(
                lambda r: _is_buy_signal(r["rsi"], r["macd_hist"], r["sma_ratio"]),
                axis=1,
            )
            signal_days = combined[signal_mask].copy()
            if signal_days.empty:
                log.warning(f"[ML] {ticker}: no signal days")
                continue

            # Ensemble label: positive if ≥2 of 3 horizons clear the threshold
            votes = sum(
                (signal_days[f"_fwd_{days}"] > thresh).astype(int)
                for days, thresh in zip(_FORWARD_DAYS, _THRESHOLDS)
            )
            label = (votes >= 2).astype(int)

            # Whipsaw filter: suppress positive label when 5d forward drawdown > threshold
            # close.rolling(5).min().shift(-5) = min of the 5 days immediately after each row
            close_full = df["Close"].squeeze()
            fwd_min_5 = close_full.rolling(5).min().shift(-5).reindex(signal_days.index)
            close_aligned = close.reindex(signal_days.index)
            dd_5 = (fwd_min_5 / close_aligned - 1)
            label[dd_5 < -_WHIPSAW_DD] = 0

            X = signal_days[FEATURE_NAMES]
            y = label

            all_X.append(X)
            all_y.append(y)
            log.info(
                f"[ML] {ticker}: {len(y)} samples after ensemble labels "
                f"(pos={int(y.sum())}, neg={int((y==0).sum())})"
            )

        except Exception as e:
            log.error(f"[ML] Failed to bootstrap {ticker}: {e}")

    if not all_X:
        return pd.DataFrame(columns=FEATURE_NAMES), pd.Series(dtype=int)

    return pd.concat(all_X, ignore_index=True), pd.concat(all_y, ignore_index=True)


def build_live_dataset(db_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Extract completed trade outcomes with stored ML features from journal.db."""
    db_path = Path(db_path)
    if not db_path.exists():
        return pd.DataFrame(columns=FEATURE_NAMES), pd.Series(dtype=int)

    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        outcomes = con.execute("SELECT * FROM trade_outcomes ORDER BY buy_ts").fetchall()
        signals = con.execute(
            "SELECT ts, ticker, data FROM cycle_events WHERE event_type='signal'"
        ).fetchall()

    if not outcomes or not signals:
        return pd.DataFrame(columns=FEATURE_NAMES), pd.Series(dtype=int)

    # Index signals by ticker
    sig_by_ticker: dict[str, list[dict]] = {}
    for row in signals:
        parsed = json.loads(row["data"])
        sig_by_ticker.setdefault(row["ticker"], []).append(
            {"ts": row["ts"], **parsed}
        )

    rows = []
    for outcome in outcomes:
        ticker = outcome["ticker"]
        ticker_sigs = sig_by_ticker.get(ticker, [])
        if not ticker_sigs:
            continue

        # Find the signal event just before the buy timestamp
        buy_ts = outcome["buy_ts"]
        pre_buy = [s for s in ticker_sigs if s["ts"] <= buy_ts]
        if not pre_buy:
            continue

        latest = max(pre_buy, key=lambda s: s["ts"])
        ml_features = latest.get("ml_features", {})

        # Only use live records that have the full feature set stored
        if len(ml_features) < len(FEATURE_NAMES):
            continue

        row = {f: float(ml_features.get(f, 0.0)) for f in FEATURE_NAMES}
        row["_label"] = 1 if outcome["pnl_pct"] > 0 else 0
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=FEATURE_NAMES), pd.Series(dtype=int)

    live_df = pd.DataFrame(rows)
    log.info(f"[ML] Live dataset: {len(live_df)} samples from journal")
    return live_df[FEATURE_NAMES], live_df["_label"]


def build_full_dataset(
    tickers: list[str], db_path: Path, period: str = "2y"
) -> tuple[pd.DataFrame, pd.Series]:
    hist_X, hist_y = build_historical_dataset(tickers, period)
    live_X, live_y = build_live_dataset(db_path)

    if hist_X.empty and live_X.empty:
        return pd.DataFrame(columns=FEATURE_NAMES), pd.Series(dtype=int)

    parts_X = [df for df in [hist_X, live_X] if not df.empty]
    parts_y = [s for s in [hist_y, live_y] if not s.empty]

    X = pd.concat(parts_X, ignore_index=True)
    y = pd.concat(parts_y, ignore_index=True)
    log.info(f"[ML] Full dataset: {len(y)} samples total (hist={len(hist_y)}, live={len(live_y)})")
    return X[FEATURE_NAMES], y
