import pytest
from tr_agent.assets import is_crypto, to_binance_symbol


@pytest.mark.parametrize("ticker", ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USDT", "SOL-BTC", "btc-usd"])
def test_is_crypto_true(ticker):
    assert is_crypto(ticker) is True


@pytest.mark.parametrize("ticker", ["AAPL", "MSFT", "NVDA", "JPM", "SPY"])
def test_is_crypto_false(ticker):
    assert is_crypto(ticker) is False


def test_to_binance_symbol_usd_pair():
    assert to_binance_symbol("BTC-USD") == "BTCUSDT"
    assert to_binance_symbol("ETH-USD") == "ETHUSDT"
    assert to_binance_symbol("SOL-USD") == "SOLUSDT"


def test_to_binance_symbol_usdt_pair():
    assert to_binance_symbol("BTC-USDT") == "BTCUSDT"


def test_to_binance_symbol_case_insensitive():
    assert to_binance_symbol("btc-usd") == "BTCUSDT"
