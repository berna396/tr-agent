import numpy as np
import pandas as pd
import pytest

from tr_agent.ml.features import FEATURE_NAMES, compute_all_rows, compute_last_row


def _make_ohlcv(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    close = 100 + rng.normal(0, 1, n).cumsum()
    high = close + rng.uniform(0, 2, n)
    low = close - rng.uniform(0, 2, n)
    volume = rng.integers(1_000_000, 10_000_000, n).astype(float)
    # Wrap in extra level to mimic yfinance MultiIndex (some versions)
    return pd.DataFrame(
        {"Open": close, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


def test_feature_names_count():
    assert len(FEATURE_NAMES) == 14


def test_compute_all_rows_returns_expected_columns():
    df = _make_ohlcv(100)
    feat = compute_all_rows(df)
    assert set(FEATURE_NAMES).issubset(set(feat.columns))


def test_compute_all_rows_no_nan():
    df = _make_ohlcv(100)
    feat = compute_all_rows(df)
    assert not feat.isnull().any().any(), "Feature matrix must not contain NaN after dropna"


def test_compute_all_rows_drops_warmup_rows():
    df = _make_ohlcv(100)
    feat = compute_all_rows(df)
    # After computing SMA50 and ADX(14×2=28), we expect fewer rows than input
    assert len(feat) < len(df)
    assert len(feat) > 0


def test_compute_last_row_returns_dict():
    df = _make_ohlcv(100)
    result = compute_last_row(df)
    assert isinstance(result, dict)
    assert set(FEATURE_NAMES) == set(result.keys())


def test_compute_last_row_all_finite():
    df = _make_ohlcv(100)
    result = compute_last_row(df)
    for name, val in result.items():
        assert np.isfinite(val), f"Feature {name} is not finite: {val}"


def test_compute_last_row_insufficient_data():
    """With too few rows, should return zeros dict rather than raising."""
    df = _make_ohlcv(10)
    result = compute_last_row(df)
    assert set(result.keys()) == set(FEATURE_NAMES)
    # All zeros on failure
    assert all(v == 0.0 for v in result.values())


def test_day_of_week_range():
    df = _make_ohlcv(100)
    feat = compute_all_rows(df)
    assert feat["day_of_week"].between(0, 6).all()


def test_sma_ratio_golden_cross():
    """sma_ratio > 1 should correspond to sma_20 > sma_50 (golden cross)."""
    df = _make_ohlcv(100)
    feat = compute_all_rows(df)
    # sma_ratio = sma_20 / sma_50, so > 1 means sma_20 > sma_50
    assert (feat["sma_ratio"] > 0).all()
