from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

import pytest

import tr_agent.news as news_mod
from tr_agent.news import fetch_news


def _make_article(title: str, publisher: str, age_hours: float = 1.0) -> dict:
    ts = int((datetime.now(timezone.utc) - timedelta(hours=age_hours)).timestamp())
    return {"title": title, "publisher": publisher, "providerPublishTime": ts}


@pytest.fixture(autouse=True)
def clear_cache():
    news_mod._NEWS_CACHE.clear()
    yield
    news_mod._NEWS_CACHE.clear()


def test_fetch_news_returns_formatted_list():
    articles = [
        _make_article("NVDA beats Q1 estimates", "Reuters", age_hours=2),
        _make_article("AI chip demand rising", "Bloomberg", age_hours=5),
        _make_article("Old news", "WSJ", age_hours=50),  # too old
    ]
    mock_ticker = MagicMock()
    mock_ticker.news = articles

    with patch("tr_agent.news.yf.Ticker", return_value=mock_ticker):
        result = fetch_news("NVDA", max_articles=3, max_age_hours=48)

    assert len(result) == 2
    assert result[0]["title"] == "NVDA beats Q1 estimates"
    assert result[0]["publisher"] == "Reuters"
    assert "h ago" in result[0]["age_str"]


def test_fetch_news_filters_old_articles():
    articles = [
        _make_article("Old article", "Reuters", age_hours=72),
    ]
    mock_ticker = MagicMock()
    mock_ticker.news = articles

    with patch("tr_agent.news.yf.Ticker", return_value=mock_ticker):
        result = fetch_news("AAPL", max_age_hours=48)

    assert result == []


def test_fetch_news_respects_max_articles():
    articles = [_make_article(f"Article {i}", "Reuters", age_hours=i) for i in range(10)]
    mock_ticker = MagicMock()
    mock_ticker.news = articles

    with patch("tr_agent.news.yf.Ticker", return_value=mock_ticker):
        result = fetch_news("MSFT", max_articles=2)

    assert len(result) == 2


def test_fetch_news_returns_empty_on_exception():
    with patch("tr_agent.news.yf.Ticker", side_effect=Exception("network error")):
        result = fetch_news("TSLA")

    assert result == []


def test_fetch_news_caches_per_day():
    articles = [_make_article("Headline", "Reuters", age_hours=1)]
    mock_ticker = MagicMock()
    mock_ticker.news = articles

    with patch("tr_agent.news.yf.Ticker", return_value=mock_ticker) as mock_yf:
        fetch_news("GOOGL")
        fetch_news("GOOGL")

    # yf.Ticker should be called only once (second call hits cache)
    assert mock_yf.call_count == 1


def test_fetch_news_age_str_minutes():
    articles = [_make_article("Breaking", "CNBC", age_hours=0.25)]  # 15 min ago
    mock_ticker = MagicMock()
    mock_ticker.news = articles

    with patch("tr_agent.news.yf.Ticker", return_value=mock_ticker):
        result = fetch_news("META")

    assert "m ago" in result[0]["age_str"]
