import logging

import typer
from rich.console import Console
from rich.table import Table

from tr_agent.broker.paper import PaperBroker
from tr_agent.config import DEFAULT_WATCHLIST, settings

app = typer.Typer(help="tr-agent: local AI trading agent with Ollama")
console = Console()


@app.command()
def trade(
    tickers: str = typer.Option(
        ",".join(DEFAULT_WATCHLIST), "--tickers", "-t",
        help="Comma-separated tickers to analyze",
    ),
    capital: float = typer.Option(None, "--capital", "-c"),
):
    """Run one trading cycle immediately (useful for testing)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    from tr_agent.scheduler import run_cycle
    ticker_list = [t.strip().upper() for t in tickers.split(",")]
    console.print(f"\n[bold green]tr-agent[/bold green] one-shot | model: {settings.ollama_model}")
    console.print(f"Watchlist: {', '.join(ticker_list)}\n")
    run_cycle(ticker_list)
    _show_portfolio(capital)


@app.command()
def scheduler(
    tickers: str = typer.Option(
        ",".join(DEFAULT_WATCHLIST), "--tickers", "-t",
        help="Comma-separated tickers to watch",
    ),
):
    """Start the scheduler — runs every 30 min during NYSE market hours."""
    from tr_agent.scheduler import start
    ticker_list = [t.strip().upper() for t in tickers.split(",")]
    start(ticker_list)


@app.command()
def portfolio(
    tickers: str = typer.Option(
        ",".join(DEFAULT_WATCHLIST), "--tickers", "-t",
    ),
    capital: float = typer.Option(None, "--capital", "-c"),
):
    """Show current paper portfolio state."""
    _show_portfolio(capital, tickers)


def _show_portfolio(capital: float | None = None, tickers: str | None = None) -> None:
    broker = PaperBroker(initial_capital=capital or settings.paper_initial_capital)
    ticker_list = (
        [t.strip().upper() for t in tickers.split(",") if t.strip()]
        if tickers else DEFAULT_WATCHLIST
    )
    metrics = broker.get_metrics(ticker_list)
    p = broker.get_portfolio()

    table = Table(title="Portfolio (paper trading)")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Cash", f"${metrics['cash']:,.2f}")
    table.add_row("Market value", f"${metrics['market_value']:,.2f}")
    table.add_row("Total value", f"${metrics['total_value']:,.2f}")
    table.add_row("Unrealized P&L", f"${metrics['unrealized_pnl']:,.2f}")
    table.add_row("Realized P&L", f"${metrics['realized_pnl']:,.2f}")
    table.add_row("Total return", f"{metrics['total_return_pct']:+.2f}%")
    table.add_row("# trades", str(metrics["num_trades"]))
    console.print(table)

    if p.positions:
        pos_table = Table(title="Open positions")
        pos_table.add_column("Ticker")
        pos_table.add_column("Qty", justify="right")
        pos_table.add_column("Avg price", justify="right")
        pos_table.add_column("Cost basis", justify="right")
        for ticker, pos in p.positions.items():
            pos_table.add_row(ticker, str(pos.quantity), f"${pos.avg_price:.4f}", f"${pos.cost_basis:.2f}")
        console.print(pos_table)


if __name__ == "__main__":
    app()
