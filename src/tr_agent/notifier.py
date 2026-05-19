import logging

import httpx
from tr_agent.config import settings

log = logging.getLogger(__name__)


def send(message: str) -> bool:
    """Send a Telegram message. Returns True on success."""
    if not settings.telegram_token or not settings.telegram_chat_id:
        log.debug("Telegram not configured — skipping notification")
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
        if resp.status_code != 200:
            log.error(
                "Telegram send failed: HTTP %d — %s",
                resp.status_code,
                resp.text[:200],
            )
            # Retry without Markdown in case of a parse error
            resp2 = httpx.post(
                f"https://api.telegram.org/bot{settings.telegram_token}/sendMessage",
                json={"chat_id": settings.telegram_chat_id, "text": message},
                timeout=10,
            )
            if resp2.status_code != 200:
                log.error("Telegram send failed (plain text retry): HTTP %d", resp2.status_code)
                return False
        return True
    except Exception as exc:
        log.error("Telegram send exception: %s", exc)
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


def send_trade_alert(
    analysis,
    decision,
    trade_pct: float,
    news_ctx=None,
    stop_price: float | None = None,
) -> bool:
    """
    Rich signal alert for manual execution (or paper-mode review).
    Called when a trade passes all gates (technical + news + risk + LLM).
    """
    side = decision.side.value.upper()
    emoji = "🟢" if side == "BUY" else "🔴"
    sep = "━" * 23

    kelly_pct = f"{trade_pct:.0%}" if trade_pct else "—"
    trade_value = analysis.close * decision.quantity if analysis.close and decision.quantity else 0
    stop_line = (
        f"Stop: ${stop_price:.2f} ({(stop_price - analysis.close) / analysis.close:+.1%})"
        if stop_price else "Stop: ATR-based (pending)"
    )

    news_line = "—"
    if news_ctx is not None:
        sign = "+" if news_ctx.sentiment_score >= 0 else ""
        news_line = f"{news_ctx.risk_level.upper()} ({sign}{news_ctx.sentiment_score:.2f})"
        if news_ctx.flags:
            news_line += f" [{', '.join(news_ctx.flags)}]"

    ml_line = (
        f"{analysis.ml_confidence:.0%}" if analysis.ml_confidence is not None else "no model"
    )

    lines = [
        f"{emoji} *TRADE SIGNAL: {side} {analysis.ticker}*",
        f"`{sep}`",
        f"Entry: ${analysis.close:.2f}   {stop_line}",
        f"Kelly size: {kelly_pct} → ~${trade_value:,.0f} ({decision.quantity:.2f} shares)",
        f"RSI: {analysis.rsi:.0f} | MACD: {analysis.macd_hist:+.2f} | ADX: {analysis.ml_features.get('adx', 0):.0f}",
        f"News: {news_line}",
        f"ML confidence: {ml_line}",
        f"_{decision.reasoning}_",
    ]
    return send("\n".join(lines))
