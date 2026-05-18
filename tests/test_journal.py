import pytest

from tr_agent import journal, memory


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "test_journal.db"
    journal.init(path)
    return path


def test_init_creates_tables(db):
    stats = journal.get_ticker_stats("AAPL", db)
    assert stats is None  # empty journal


def test_log_signal(db):
    journal.log_signal("AAPL", "buy", 28.0, 0.05, 155.0, 145.0, 150.0, "RSI oversold", path=db)
    # No error = success; verify via stats after adding an outcome


def test_record_and_query_outcome(db):
    journal.record_outcome(
        ticker="AAPL",
        buy_ts="2025-01-10T10:00:00+00:00",
        sell_ts="2025-01-15T14:00:00+00:00",
        buy_price=150.0,
        sell_price=157.5,
        quantity=5.0,
        buy_reasoning="RSI oversold",
        sell_reasoning="RSI overbought",
        path=db,
    )
    stats = journal.get_ticker_stats("AAPL", db)
    assert stats is not None
    assert stats.total_trades == 1
    assert stats.wins == 1
    assert stats.losses == 0
    assert stats.win_rate_pct == 100.0
    assert stats.avg_pnl_pct == pytest.approx(5.0, rel=1e-2)


def test_multiple_outcomes_win_rate(db):
    trades = [
        ("2025-01-10", "2025-01-15", 100.0, 110.0),  # WIN +10%
        ("2025-01-16", "2025-01-20", 110.0, 105.0),  # LOSS -4.5%
        ("2025-01-21", "2025-01-25", 105.0, 115.0),  # WIN +9.5%
    ]
    for buy_ts, sell_ts, buy_p, sell_p in trades:
        journal.record_outcome("MSFT", buy_ts, sell_ts, buy_p, sell_p, 3.0, path=db)

    stats = journal.get_ticker_stats("MSFT", db)
    assert stats.total_trades == 3
    assert stats.wins == 2
    assert stats.losses == 1
    assert stats.win_rate_pct == pytest.approx(66.67, rel=1e-2)


def test_memory_no_history(db):
    context = memory.build_context("NVDA", "buy", path=db)
    assert context == ""


def test_memory_with_history(db):
    journal.record_outcome("TSLA", "2025-01-10", "2025-01-15", 200.0, 220.0, 2.0, path=db)
    journal.record_outcome("TSLA", "2025-01-20", "2025-01-25", 220.0, 210.0, 2.0, path=db)

    context = memory.build_context("TSLA", "buy", path=db)
    assert "TSLA" in context
    assert "50%" in context  # 1 win / 2 trades
    assert "WIN" in context
    assert "LOSS" in context


def test_pending_buy_found(db):
    journal.log_order("AAPL", "buy", 5.0, 150.0, "order1", db)
    pending = journal.get_pending_buy("AAPL", db)
    assert pending is not None
    assert pending["fill_price"] == 150.0


def test_pending_buy_cleared_after_outcome(db):
    journal.log_order("AAPL", "buy", 5.0, 150.0, "order1", db)
    pending = journal.get_pending_buy("AAPL", db)
    journal.record_outcome("AAPL", pending["ts"], "2025-01-15", 150.0, 160.0, 5.0, path=db)
    assert journal.get_pending_buy("AAPL", db) is None
