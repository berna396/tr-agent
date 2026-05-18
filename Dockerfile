FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install dependencies first (cached layer)
COPY pyproject.toml .python-version ./
RUN uv sync --no-dev --no-install-project

# Copy source
COPY src/ src/

# Install the project itself
RUN uv sync --no-dev

# Data directory for portfolio state and journal
RUN mkdir -p data

CMD ["uv", "run", "python", "-m", "tr_agent.main", "scheduler"]
