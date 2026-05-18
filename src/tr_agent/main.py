import typer
from rich.console import Console
from rich.table import Table

from tr_agent import agent
from tr_agent.broker.paper import PaperBroker
from tr_agent.config import settings

app = typer.Typer(help="tr-agent: agente de trading local con Ollama")
console = Console()


@app.command()
def trade(
    tickers: str = typer.Option(..., "--tickers", "-t", help="Tickers separados por coma: AAPL,MSFT"),
    capital: float = typer.Option(None, "--capital", "-c", help="Capital inicial (solo si no hay portfolio guardado)"),
    mode: str = typer.Option("paper", "--mode", "-m", help="Modo: paper (por defecto) | live (iter 2)"),
    task: str = typer.Option(None, "--task", help="Tarea personalizada para el agente"),
):
    """Lanza el agente para analizar los tickers indicados y operar si hay señal."""
    if mode != "paper":
        console.print("[red]El modo live estará disponible en iteración 2.[/red]")
        raise typer.Exit(1)

    initial_capital = capital or settings.paper_initial_capital
    broker = PaperBroker(initial_capital=initial_capital, slippage=settings.paper_slippage)

    ticker_list = [t.strip().upper() for t in tickers.split(",")]
    default_task = (
        f"Analiza los siguientes activos y opera si hay señal clara: {', '.join(ticker_list)}. "
        "Recuerda revisar el portfolio antes de cada operación y respetar los límites de riesgo."
    )
    user_task = task or default_task

    console.print(f"\n[bold green]tr-agent[/bold green] — modo: {mode} | modelo: {settings.ollama_model}")
    console.print(f"Tickers: {', '.join(ticker_list)}\n")

    result = agent.core.run(broker, user_task)

    console.print("\n[bold]--- Resultado del agente ---[/bold]")
    console.print(result)

    _print_portfolio(broker, ticker_list)


@app.command()
def portfolio(
    tickers: str = typer.Option("", "--tickers", "-t", help="Tickers para calcular P&L no realizado"),
    capital: float = typer.Option(None, "--capital", "-c"),
):
    """Muestra el estado actual del portfolio paper."""
    initial_capital = capital or settings.paper_initial_capital
    broker = PaperBroker(initial_capital=initial_capital)
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()] if tickers else None
    _print_portfolio(broker, ticker_list)


def _print_portfolio(broker: PaperBroker, tickers: list[str] | None) -> None:
    p = broker.get_portfolio()
    metrics = broker.get_metrics(tickers)

    table = Table(title="Portfolio (paper trading)")
    table.add_column("Métrica")
    table.add_column("Valor", justify="right")

    table.add_row("Efectivo", f"${metrics['cash']:,.2f}")
    table.add_row("Valor de mercado", f"${metrics['market_value']:,.2f}")
    table.add_row("Valor total", f"${metrics['total_value']:,.2f}")
    table.add_row("P&L no realizado", f"${metrics['unrealized_pnl']:,.2f}")
    table.add_row("P&L realizado", f"${metrics['realized_pnl']:,.2f}")
    table.add_row("Retorno total", f"{metrics['total_return_pct']:.2f}%")
    table.add_row("Nº operaciones", str(metrics["num_trades"]))
    console.print(table)

    if p.positions:
        pos_table = Table(title="Posiciones abiertas")
        pos_table.add_column("Ticker")
        pos_table.add_column("Cantidad", justify="right")
        pos_table.add_column("Precio medio", justify="right")
        pos_table.add_column("Coste total", justify="right")
        for ticker, pos in p.positions.items():
            pos_table.add_row(ticker, str(pos.quantity), f"${pos.avg_price:.4f}", f"${pos.cost_basis:.2f}")
        console.print(pos_table)


if __name__ == "__main__":
    app()
