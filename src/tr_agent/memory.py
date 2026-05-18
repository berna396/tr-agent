"""
Builds the historical memory context injected into the LLM prompt.
Queries the journal for past performance on a given ticker and formats it
as a concise text block the LLM can reason about.
"""

from pathlib import Path

from tr_agent.journal import TickerStats, get_ticker_stats


def build_context(ticker: str, signal: str, path: Path | None = None) -> str:
    """
    Returns a formatted memory block for the LLM prompt.
    Empty string if no history exists yet.
    """
    kwargs = {"path": path} if path else {}
    stats = get_ticker_stats(ticker, **kwargs)

    if stats is None or stats.total_trades == 0:
        return ""

    lines = [
        f"Historical performance for {ticker} ({stats.total_trades} completed trades):",
        f"  Win rate: {stats.win_rate_pct:.0f}% ({stats.wins}W / {stats.losses}L)",
        f"  Avg return: {stats.avg_pnl_pct:+.1f}% | Avg win: {stats.avg_win_pct:+.1f}% | Avg loss: {stats.avg_loss_pct:+.1f}%",
    ]

    if stats.recent_outcomes:
        lines.append("  Last trades:")
        for o in stats.recent_outcomes:
            result = "WIN" if o["pnl"] > 0 else "LOSS"
            lines.append(
                f"    {o['buy_ts']} → {o['sell_ts']}: {result} {o['pnl_pct']:+.1f}% "
                f"(${o['buy_price']:.2f} → ${o['sell_price']:.2f})"
            )

    return "\n".join(lines)
