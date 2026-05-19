CRYPTO_SUFFIXES = ("-USD", "-USDT", "-BTC", "-ETH")


def is_crypto(ticker: str) -> bool:
    return any(ticker.upper().endswith(s) for s in CRYPTO_SUFFIXES)


def to_binance_symbol(ticker: str) -> str:
    """Convert yfinance-style ticker to Binance symbol: 'BTC-USD' → 'BTCUSDT'."""
    t = ticker.upper()
    for suffix in ("-USD", "-USDT"):
        if t.endswith(suffix):
            base = t[: -len(suffix)]
            return base + "USDT"
    return t.replace("-", "")
