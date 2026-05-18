import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from tr_agent.config import DEFAULT_WATCHLIST
from tr_agent.screener import (
    CANDIDATE_POOL,
    load_active_watchlist,
    save_active_watchlist,
    score_ticker,
    screen,
)
from tr_agent.signals.technical import Signal, TechnicalAnalysis


def _make_analysis(
    signal: Signal = Signal.NEUTRAL,
    rsi: float = 50.0,
    ml_confidence: float | None = None,
    adx: float = 15.0,
    volume_ratio: float = 1.0,
) -> TechnicalAnalysis:
    return TechnicalAnalysis(
        ticker="TEST",
        timeframe="3mo",
        close=100.0,
        rsi=rsi,
        macd=0.0,
        macd_signal=0.0,
        macd_hist=0.1,
        sma_20=101.0,
        sma_50=99.0,
        signal=signal,
        reasoning="test",
        ml_confidence=ml_confidence,
        ml_available=ml_confidence is not None,
        ml_features={"adx": adx, "volume_ratio": volume_ratio},
    )


def test_candidate_pool_has_reasonable_size():
    assert 40 <= len(CANDIDATE_POOL) <= 60


def test_candidate_pool_no_duplicates():
    assert len(CANDIDATE_POOL) == len(set(CANDIDATE_POOL))


def test_score_neutral_signal_returns_low():
    a = _make_analysis(signal=Signal.NEUTRAL)
    assert score_ticker(a) < 2.0


def test_score_buy_signal_adds_two_points():
    neutral = _make_analysis(signal=Signal.NEUTRAL)
    buy = _make_analysis(signal=Signal.BUY)
    assert score_ticker(buy) - score_ticker(neutral) == pytest.approx(2.0)


def test_score_strong_trend_adds_point():
    low_adx = _make_analysis(signal=Signal.BUY, adx=10.0)
    high_adx = _make_analysis(signal=Signal.BUY, adx=30.0)
    assert score_ticker(high_adx) > score_ticker(low_adx)


def test_score_high_volume_adds_point():
    low_vol = _make_analysis(signal=Signal.BUY, volume_ratio=0.8)
    high_vol = _make_analysis(signal=Signal.BUY, volume_ratio=1.5)
    assert score_ticker(high_vol) > score_ticker(low_vol)


def test_score_ml_support_adds_point():
    no_ml = _make_analysis(signal=Signal.BUY, ml_confidence=None)
    with_ml = _make_analysis(signal=Signal.BUY, ml_confidence=0.65)
    assert score_ticker(with_ml) > score_ticker(no_ml)


def test_score_ml_warning_subtracts():
    no_ml = _make_analysis(signal=Signal.BUY, ml_confidence=None)
    bearish_ml = _make_analysis(signal=Signal.BUY, ml_confidence=0.30)
    assert score_ticker(bearish_ml) < score_ticker(no_ml)


def test_score_extreme_rsi_adds_half_point():
    normal = _make_analysis(signal=Signal.BUY, rsi=50.0)
    extreme = _make_analysis(signal=Signal.BUY, rsi=28.0)
    assert score_ticker(extreme) == pytest.approx(score_ticker(normal) + 0.5)


def test_screen_returns_at_most_top_n():
    analyses = {
        t: _make_analysis(signal=Signal.BUY, adx=30.0)
        for t in ["A", "B", "C", "D", "E"]
    }

    def mock_analyze(ticker, **kwargs):
        return analyses[ticker]

    with patch("tr_agent.screener.technical.analyze", side_effect=mock_analyze):
        result = screen(pool=list(analyses.keys()), top_n=3)

    assert len(result) <= 3
    assert all(t in analyses for t in result)


def test_screen_fallback_on_all_failures():
    def always_fails(ticker, **kwargs):
        raise ValueError("network error")

    with patch("tr_agent.screener.technical.analyze", side_effect=always_fails):
        result = screen(pool=["FAKE1", "FAKE2"], top_n=5)

    assert result == list(DEFAULT_WATCHLIST)


def test_screen_skips_failed_tickers_but_returns_rest():
    def flaky(ticker, **kwargs):
        if ticker == "BAD":
            raise ValueError("bad ticker")
        return _make_analysis(signal=Signal.BUY)

    with patch("tr_agent.screener.technical.analyze", side_effect=flaky):
        result = screen(pool=["GOOD1", "BAD", "GOOD2"], top_n=5)

    assert "BAD" not in result
    assert "GOOD1" in result
    assert "GOOD2" in result


def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "active_watchlist.json"
    tickers = ["AAPL", "NVDA", "MSFT"]
    save_active_watchlist(tickers, path)
    loaded = load_active_watchlist(path)
    assert loaded == tickers


def test_load_missing_file_returns_default(tmp_path):
    result = load_active_watchlist(tmp_path / "nonexistent.json")
    assert result == list(DEFAULT_WATCHLIST)


def test_load_corrupted_file_returns_default(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("not json {{{{")
    result = load_active_watchlist(bad)
    assert result == list(DEFAULT_WATCHLIST)


def test_save_creates_parent_dirs(tmp_path):
    deep_path = tmp_path / "nested" / "dir" / "watchlist.json"
    save_active_watchlist(["AAPL"], deep_path)
    assert deep_path.exists()
    data = json.loads(deep_path.read_text())
    assert data["tickers"] == ["AAPL"]
    assert "updated_at" in data
