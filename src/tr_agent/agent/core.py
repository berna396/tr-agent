import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import ollama

from tr_agent.agent.prompts import SYSTEM_PROMPT, ml_confidence_line, news_section, regime_line, rules_section
from tr_agent.broker.base import OrderSide, Portfolio
from tr_agent.config import settings
from tr_agent import memory, news as news_mod
from tr_agent.risk import RiskCheck
from tr_agent.signals.technical import TechnicalAnalysis

_RULES_PATH = Path(__file__).parents[3] / "data" / "llm_rules.md"

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
    journal_path: Optional[Path] = None,
    regime=None,
) -> TradeDecision:
    """Ask the LLM to review the signal, risk check and past performance. Returns a final trade decision."""
    positions_summary = {
        t: {"qty": p.quantity, "avg_price": p.avg_price}
        for t, p in portfolio.positions.items()
    }

    memory_context = memory.build_context(
        analysis.ticker, analysis.signal.value,
        **({"path": journal_path} if journal_path else {}),
    )

    ticker_news = news_mod.fetch_news(analysis.ticker)
    learned_rules = rules_section(_RULES_PATH)

    sma200_str = f" | SMA200: {analysis.sma_200:.2f}" if analysis.sma_200 is not None else ""
    if analysis.sma_200 is not None:
        lt_trend = "above" if analysis.close > analysis.sma_200 else "below"
        sma200_context = f"\n- Long-term trend: price is {lt_trend} SMA200 (${analysis.sma_200:.2f})"
    else:
        sma200_context = ""

    news_block = news_section(ticker_news)
    rules_block = learned_rules

    prompt = f"""Signal: {analysis.signal.upper()} on {analysis.ticker}

Technical indicators:
- RSI(14): {analysis.rsi:.1f}
- MACD histogram: {analysis.macd_hist:.4f}
- SMA20: {analysis.sma_20:.2f} | SMA50: {analysis.sma_50:.2f}{sma200_str}
- Current price: ${analysis.close:.2f}{sma200_context}
- Analysis: {analysis.reasoning}
- {ml_confidence_line(analysis.ml_confidence, analysis.ml_available)}
- {regime_line(regime)}
{f"{chr(10)}{news_block}" if news_block else ""}
Risk check:
- Max quantity allowed: {risk_check.max_quantity:.4f} shares
- Reason: {risk_check.reason}

Portfolio:
- Cash available: ${portfolio.cash:,.2f}
- Open positions: {positions_summary if positions_summary else "none"}
{f"{chr(10)}{memory_context}" if memory_context else ""}{f"{chr(10)}{rules_block}" if rules_block else ""}
Should we execute this {analysis.signal} trade?"""

    if ticker_news:
        log.info(f"[LLM] {analysis.ticker}: {len(ticker_news)} news headline(s) injected")
    if learned_rules:
        log.info(f"[LLM] {analysis.ticker}: learned rules injected ({len(learned_rules)} chars)")
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
