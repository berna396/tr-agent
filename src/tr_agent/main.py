import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from tr_agent.broker.paper import PaperBroker
from tr_agent.config import DEFAULT_WATCHLIST, settings

app = typer.Typer(help="tr-agent: local AI trading agent with Ollama")
ml_app = typer.Typer(help="ML model management")
app.add_typer(ml_app, name="ml")
console = Console()

_DATA_ROOT = Path(__file__).parents[2] / "data"
_MODEL_PATH = _DATA_ROOT / "models" / "signal_model.pkl"
_HISTORY_PATH = _DATA_ROOT / "models" / "training_history.json"
_DB_PATH = _DATA_ROOT / "journal.db"


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


@app.command()
def screen(
    dry_run: bool = typer.Option(False, "--dry-run", help="Show picks without saving"),
    top_n: int = typer.Option(0, "--top-n", "-n", help="Number of tickers to select (0 = use config)"),
):
    """Run the pre-market screener and show today's top opportunities."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    from tr_agent import screener as sc
    from tr_agent.scheduler import _WATCHLIST_PATH

    n = top_n or settings.screener_top_n
    console.print(f"\n[bold green]Pre-market Screener[/bold green] | pool: {len(sc.CANDIDATE_POOL)} tickers → top {n}\n")

    selected = sc.screen(top_n=n)

    table = Table(title=f"Top {len(selected)} picks")
    table.add_column("Ticker", style="bold")
    table.add_column("Rank", justify="right")
    for i, t in enumerate(selected, 1):
        table.add_row(t, str(i))
    console.print(table)

    if not dry_run:
        sc.save_active_watchlist(selected, _WATCHLIST_PATH)
        console.print(f"\n[green]Saved to {_WATCHLIST_PATH}[/green]")
    else:
        console.print("\n[yellow]Dry run — not saved[/yellow]")


@ml_app.command("bootstrap")
def ml_bootstrap(
    tickers: str = typer.Option(",".join(DEFAULT_WATCHLIST), "--tickers", "-t"),
    period: str = typer.Option("2y", "--period", "-p", help="yfinance period (1y, 2y, 5y)"),
):
    """Download historical data, train initial ML model, and report results."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    from tr_agent.ml.trainer import train_and_deploy

    ticker_list = [t.strip().upper() for t in tickers.split(",")]
    console.print(f"\n[bold green]ML Bootstrap[/bold green] | tickers: {', '.join(ticker_list)} | period: {period}\n")

    report = train_and_deploy(ticker_list, _DB_PATH, _MODEL_PATH, _HISTORY_PATH, period, force=True)

    if report.get("deployed"):
        console.print(f"[bold green]Model v{report['version']} trained and deployed[/bold green]")
        console.print(f"  CV AUC:  {report['cv_auc']:.4f}")
        console.print(f"  Samples: {report['n_samples']}")
        console.print(f"  Path:    {_MODEL_PATH}")
    else:
        console.print(f"[yellow]Model not deployed:[/yellow] {report.get('reason', 'unknown')}")
        console.print(f"  Samples available: {report.get('n_samples', 0)}")


@ml_app.command("status")
def ml_status():
    """Show current model version, AUC, training date, and top SHAP features."""
    from tr_agent.ml.signal_model import SignalModel
    from tr_agent.ml.trainer import load_training_history

    model = SignalModel.load(_MODEL_PATH)
    if model is None:
        console.print("[yellow]No model found.[/yellow] Run [bold]tr-agent ml bootstrap[/bold] first.")
        return

    table = Table(title=f"ML Model — v{model.version}")
    table.add_column("Property")
    table.add_column("Value", justify="right")
    table.add_row("Version", str(model.version))
    table.add_row("CV AUC", f"{model.auc:.4f}" if model.auc else "n/a")
    table.add_row("Training samples", str(model.n_samples))
    table.add_row("Trained at", model.train_date[:19].replace("T", " ") if model.train_date else "n/a")
    console.print(table)

    history = load_training_history(_HISTORY_PATH)
    if history:
        console.print(f"\nTraining runs: {len(history)} | Deployed: {sum(1 for r in history if r.get('deployed'))}")


@ml_app.command("analyze")
def ml_analyze(
    tickers: str = typer.Option(",".join(DEFAULT_WATCHLIST), "--tickers", "-t"),
    days: int = typer.Option(30, "--days", "-d"),
):
    """Run Ollama analysis on journal data and print the insight report."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    from tr_agent.ml import analyzer
    from tr_agent.ml.signal_model import SignalModel
    from tr_agent.ml.dataset import build_full_dataset

    ticker_list = [t.strip().upper() for t in tickers.split(",")]
    report = analyzer.build_performance_report(_DB_PATH, days=days)

    shap = {}
    model = SignalModel.load(_MODEL_PATH)
    if model:
        X, _ = build_full_dataset(ticker_list, _DB_PATH, settings.ml_backtest_period)
        if not X.empty:
            shap = analyzer.compute_shap_importances(model, X)

    console.print(f"\n[bold]Performance Report — last {days} days[/bold]")
    if "error" in report:
        console.print(f"[yellow]{report['error']}[/yellow]")
    else:
        console.print(f"Trades: {report.get('total_trades')} | Win rate: {report.get('win_rate')}% | P&L: ${report.get('total_pnl', 0):+.2f}")

    if shap:
        console.print("\n[bold]Top SHAP Features:[/bold]")
        for feat, imp in list(shap.items())[:6]:
            console.print(f"  {feat}: {imp:.4f}")

    if report.get("total_trades", 0) > 0:
        console.print("\n[bold]Generating Ollama insights...[/bold]")
        insights = analyzer.generate_ollama_insights(settings.ollama_model, report, shap)
        console.print(f"\n{insights}")


@app.command()
def web(
    host: str = typer.Option("0.0.0.0", "--host", help="Bind host"),
    port: int = typer.Option(8080, "--port", "-p", help="Port to listen on"),
):
    """Start the web dashboard (process control, metrics, live logs)."""
    from tr_agent.web import main as web_main
    console.print(f"\n[bold green]tr-agent dashboard[/bold green] → http://{host}:{port}\n")
    web_main(host=host, port=port)


if __name__ == "__main__":
    app()
