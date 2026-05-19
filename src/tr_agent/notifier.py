import logging

import httpx
from tr_agent.config import settings

log = logging.getLogger(__name__)


# ── Telegram (kept for backward compatibility) ────────────────────────────────

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


# ── Slack ─────────────────────────────────────────────────────────────────────

def send_slack(text: str) -> bool:
    """Post a plain-text message to the configured Slack webhook. Returns True on success."""
    if not settings.slack_webhook_url:
        log.debug("Slack webhook not configured — skipping notification")
        return False
    try:
        resp = httpx.post(
            settings.slack_webhook_url,
            json={"text": text},
            timeout=10,
        )
        if resp.status_code != 200:
            log.error("Slack send failed: HTTP %d — %s", resp.status_code, resp.text[:200])
            return False
        return True
    except Exception as exc:
        log.error("Slack send exception: %s", exc)
        return False


def send_trade_slack(
    ticker: str,
    side: str,            # "buy" | "sell"
    quantity: float,
    price: float,
    cash_pct: float | None = None,   # fraction of cash, e.g. 0.08 = 8%
    stop_price: float | None = None,
    pnl_pct: float | None = None,    # realised P&L % (sell only)
    reason: str = "",                # e.g. "stop-loss"
) -> bool:
    """
    Send a concise buy/sell trade notification to Slack.
    This is the only notification the agent sends to Slack — no noise.
    """
    is_buy = side.lower() == "buy"
    emoji  = "🟢" if is_buy else "🔴"
    label  = "BUY" if is_buy else "SELL"

    lines = [f"{emoji} *{label} {ticker}*"]

    if is_buy:
        size_line = f"${price:.2f}"
        if cash_pct is not None:
            dollar_val = price * quantity
            size_line += f"  ·  *{cash_pct:.0%} of cash*  (~${dollar_val:,.0f} · {quantity:.2f} sh)"
        else:
            size_line += f"  ·  {quantity:.2f} shares"
        lines.append(size_line)

        if stop_price:
            stop_pct = (stop_price - price) / price
            lines.append(f"Stop: ${stop_price:.2f}  ({stop_pct:+.1%})")
    else:
        sell_line = f"${price:.2f}  ·  {quantity:.2f} shares"
        if reason:
            sell_line += f"  ·  _{reason}_"
        lines.append(sell_line)

        if pnl_pct is not None:
            sign = "+" if pnl_pct >= 0 else ""
            lines.append(f"P&L: {sign}{pnl_pct:.2f}%")

    return send_slack("\n".join(lines))
