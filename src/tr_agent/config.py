from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"

    paper_mode: bool = True
    paper_initial_capital: float = 10_000.0
    paper_slippage: float = 0.001  # 0.1%

    # Trade Republic (iteración 2)
    tr_phone: str = Field(default="", repr=False)
    tr_pin: str = Field(default="", repr=False)


settings = Settings()
