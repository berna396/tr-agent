import httpx
from tr_agent.config import settings


def send(message: str) -> bool:
    """Send a Telegram message. Returns True on success."""
    if not settings.telegram_token or not settings.telegram_chat_id:
        return False
    try:
        resp = httpx.post(
            f"https://api.telegram.org/bot{settings.telegram_token}/sendMessage",
            json={
                "chat_id": settings.telegram_chat_id,
                "text": message,
                "parse_mode": "Markdown",
            },
            timeout=10,
        )
        return resp.status_code == 200
    except Exception:
        return False


def send_run_summary(
    tickers: list[str],
    agent_result: str,
    metrics: dict,
    orders_this_run: list[dict],
) -> bool:
    lines = ["📊 *tr-agent run*", f"Watchlist: `{', '.join(tickers)}`", ""]

    if orders_this_run:
        lines.append("*Trades executed:*")
        for o in orders_this_run:
            emoji = "🟢" if o["side"] == "buy" else "🔴"
            lines.append(
                f"{emoji} {o['side'].upper()} {o['quantity']} {o['ticker']} @ ${o['fill_price']:.2f}"
            )
    else:
        lines.append("⏸ No trades — no clear signal")

    lines += [
        "",
        "*Portfolio*",
        f"💵 Cash: ${metrics['cash']:,.2f}",
        f"📈 Total value: ${metrics['total_value']:,.2f}",
        f"{'🟢' if metrics['total_return_pct'] >= 0 else '🔴'} Return: {metrics['total_return_pct']:+.2f}%",
        f"🔁 Total trades: {metrics['num_trades']}",
    ]

    return send("\n".join(lines))
