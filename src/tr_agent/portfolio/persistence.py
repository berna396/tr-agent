import json
from pathlib import Path

from tr_agent.portfolio.tracker import PortfolioTracker

_STATE_FILE = Path(__file__).parents[3] / "data" / "portfolio_state.json"


def save(tracker: PortfolioTracker, path: Path = _STATE_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tracker.to_dict(), indent=2))


def load(initial_capital: float, path: Path = _STATE_FILE) -> PortfolioTracker:
    if path.exists():
        data = json.loads(path.read_text())
        return PortfolioTracker.from_dict(data)
    return PortfolioTracker(initial_capital=initial_capital)
