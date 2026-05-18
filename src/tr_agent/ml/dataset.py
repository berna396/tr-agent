import json
import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from tr_agent.ml.features import FEATURE_NAMES, compute_all_rows

log = logging.getLogger(__name__)

_FORWARD_DAYS = 10       # give signal more time to play out
_LABEL_THRESHOLD = 0.005  # 0.5% minimum move to avoid noisy neutral zone


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
        spy_df = yf.download("SPY", period=period, interval="1d", progress=False, auto_adjust=True)
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
            df = yf.download(
                ticker, period=period, interval="1d", progress=False, auto_adjust=True
            )
            if df.empty or len(df) < 60:
                log.warning(f"[ML] Skipping {ticker} — insufficient data ({len(df)} rows)")
                continue

            feat_df = compute_all_rows(df, spy_df=spy_df)
            close = df["Close"].squeeze().reindex(feat_df.index)
            forward_return = close.shift(-_FORWARD_DAYS) / close - 1

            combined = feat_df.copy()
            combined["_fwd"] = forward_return
            combined["_close"] = close.reindex(feat_df.index)
            combined = combined.dropna(subset=["_fwd"])

            # Filter to buy-signal days so the model learns specifically when
            # the rule-based signal tends to be reliable vs. a false positive
            signal_mask = combined.apply(
                lambda r: _is_buy_signal(r["rsi"], r["macd_hist"], r["sma_ratio"]),
                axis=1,
            )
            signal_days = combined[signal_mask]

            # Drop the neutral zone
            non_neutral = signal_days[signal_days["_fwd"].abs() > _LABEL_THRESHOLD]
            if non_neutral.empty:
                log.warning(f"[ML] {ticker}: no signal days with clear forward return")
                continue

            X = non_neutral[FEATURE_NAMES]
            y = (non_neutral["_fwd"] > 0).astype(int)

            all_X.append(X)
            all_y.append(y)
            log.info(f"[ML] {ticker}: {len(y)} labeled samples (pos={y.sum()}, neg={(y==0).sum()})")

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
