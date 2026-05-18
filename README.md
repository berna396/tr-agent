# tr-agent

Agente de trading local usando Ollama + LangChain. Soporta paper trading y (próximamente) Trade Republic.

## Requisitos

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/) con `qwen2.5:7b`

## Setup

```bash
# 1. Descargar el modelo LLM
ollama pull qwen2.5:7b

# 2. Instalar dependencias
uv sync --extra dev

# 3. Configurar variables de entorno
cp .env.example .env
# Editar .env si es necesario

# 4. Ejecutar en modo paper
uv run python -m tr_agent.main --tickers AAPL --mode paper
```

## Arquitectura

```
main.py (CLI)
    └── agent/core.py (AgentExecutor + Ollama)
            ├── agent/tools.py (@tool: get_quote, place_order, ...)
            ├── broker/paper.py (simula trades)
            ├── portfolio/tracker.py (estado cartera)
            └── signals/technical.py (RSI, MACD, SMA)
```

## Iteraciones

- **Iter 1 (actual):** Paper trading con datos de yfinance
- **Iter 2:** Conexión live a Trade Republic vía pytr
- **Iter 3:** Señales ML (XGBoost sobre features técnicas)
- **Iter 4:** Risk management avanzado
