import json
import logging
from dataclasses import dataclass

import ollama

from tr_agent.agent.prompts import SYSTEM_PROMPT
from tr_agent.broker.base import OrderSide, Portfolio
from tr_agent.config import settings
from tr_agent.risk import RiskCheck
from tr_agent.signals.technical import TechnicalAnalysis

log = logging.getLogger(__name__)


@dataclass
class TradeDecision:
    confirmed: bool
    quantity: float
    reasoning: str
    side: OrderSide


def confirm_trade(
    analysis: TechnicalAnalysis,
    risk_check: RiskCheck,
    portfolio: Portfolio,
) -> TradeDecision:
    """Ask the LLM to review the signal and risk check. Returns a final trade decision."""
    positions_summary = {
        t: {"qty": p.quantity, "avg_price": p.avg_price}
        for t, p in portfolio.positions.items()
    }

    prompt = f"""Signal: {analysis.signal.upper()} on {analysis.ticker}

Technical indicators:
- RSI(14): {analysis.rsi:.1f}
- MACD histogram: {analysis.macd_hist:.4f}
- SMA20: {analysis.sma_20:.2f} | SMA50: {analysis.sma_50:.2f}
- Current price: ${analysis.close:.2f}
- Analysis: {analysis.reasoning}

Risk check:
- Max quantity allowed: {risk_check.max_quantity:.4f} shares
- Reason: {risk_check.reason}

Portfolio:
- Cash available: ${portfolio.cash:,.2f}
- Open positions: {positions_summary if positions_summary else "none"}

Should we execute this {analysis.signal} trade?"""

    log.info(f"[LLM] Asking for confirmation on {analysis.ticker} {analysis.signal}...")

    try:
        response = ollama.chat(
            model=settings.ollama_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            format="json",
            options={"temperature": 0.1},
        )
        raw = response.message.content
        log.info(f"[LLM] Response: {raw}")
        data = json.loads(raw)
        return TradeDecision(
            confirmed=bool(data.get("confirmed", False)),
            quantity=float(data.get("quantity", 0)),
            reasoning=data.get("reasoning", ""),
            side=OrderSide(analysis.signal.lower()),
        )
    except Exception as e:
        log.error(f"[LLM] Failed to get confirmation: {e}")
        return TradeDecision(confirmed=False, quantity=0, reasoning=f"LLM error: {e}", side=OrderSide.BUY)
