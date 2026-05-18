from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_WATCHLIST = ["AAPL", "MSFT", "META", "TSLA", "NVDA"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"

    paper_mode: bool = True
    paper_initial_capital: float = 10_000.0
    paper_slippage: float = 0.001  # 0.1%

    # Risk manager
    max_trade_pct: float = 0.20       # max 20% of cash per trade
    max_invested_pct: float = 0.60    # max 60% of portfolio invested

    # Telegram notifications
    telegram_token: str = Field(default="", repr=False)
    telegram_chat_id: str = Field(default="", repr=False)

    # Trade Republic (iteration 2)
    tr_phone: str = Field(default="", repr=False)
    tr_pin: str = Field(default="", repr=False)

    # ML layer
    ml_min_new_samples: int = 10
    ml_backtest_period: str = "2y"

    # Pre-market screener
    screener_top_n: int = 12
    screener_min_price: float = 10.0
    screener_min_avg_volume: int = 500_000

    # v0.5 safety guards
    stop_loss_pct: float = 0.05          # 5% — set to 0 to disable
    earnings_blackout_days: int = 3      # avoid BUY within N days of earnings
    regime_filter_enabled: bool = True   # suppress BUY when SPY is bearish


settings = Settings()
