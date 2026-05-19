"""
LLM-powered news and earnings analyst.
Produces a structured NewsContext from raw headlines instead of passing them
raw to the trade-confirmation prompt.
"""

import json
import logging
from dataclasses import dataclass, field

import ollama

from tr_agent.config import settings

log = logging.getLogger(__name__)


@dataclass
class NewsContext:
    sentiment_score: float = 0.0        # -1.0 (bearish) to +1.0 (bullish)
    risk_level: str = "low"             # "low" | "medium" | "high"
    flags: list[str] = field(default_factory=list)   # e.g. ["earnings_risk", "analyst_downgrade"]
    summary: str = ""                   # 1-2 sentence synthesis

    @property
    def is_high_risk(self) -> bool:
        return self.risk_level == "high"

    def prompt_block(self) -> str:
        """Format for injection into the trade-confirmation prompt."""
        flag_str = ", ".join(self.flags) if self.flags else "none"
        sign = "+" if self.sentiment_score >= 0 else ""
        return (
            f"News analysis (LLM):\n"
            f"- Sentiment: {sign}{self.sentiment_score:.2f}  Risk: {self.risk_level}  Flags: {flag_str}\n"
            f"- Summary: {self.summary}"
        )


_SYSTEM = (
    "You are a financial news analyst. Summarise the provided headlines for a stock "
    "and output ONLY valid JSON with exactly these keys: "
    '{"sentiment_score": float, "risk_level": str, "flags": list[str], "summary": str}. '
    "sentiment_score: -1.0 (very bearish) to +1.0 (very bullish). "
    "risk_level: one of low|medium|high. "
    "flags: list of short labels such as earnings_risk, analyst_upgrade, analyst_downgrade, "
    "regulatory_risk, macro_risk, product_launch, legal_risk. Empty list if none apply. "
    "summary: 1-2 sentences max."
)

_NEUTRAL = NewsContext(sentiment_score=0.0, risk_level="low", flags=[], summary="No recent news.")


def analyze_news(
    ticker: str,
    headlines: list[dict],
    earnings_days_away: int | None = None,
) -> NewsContext:
    """
    Call the local LLM to produce a structured NewsContext from raw headlines.
    Never raises — returns a neutral context on any error.
    """
    if not headlines:
        if earnings_days_away is not None and earnings_days_away <= 3:
            return NewsContext(
                sentiment_score=0.0,
                risk_level="high",
                flags=["earnings_risk"],
                summary=f"Earnings in {earnings_days_away} day(s). No news available.",
            )
        return _NEUTRAL

    headline_text = "\n".join(
        f"- {h.get('title', '')} ({h.get('publisher', '')} {h.get('age_str', '')})"
        for h in headlines
    )
    earnings_line = (
        f"\nEarnings report in {earnings_days_away} day(s) — elevated uncertainty."
        if earnings_days_away is not None and earnings_days_away <= 5
        else ""
    )
    prompt = (
        f"Ticker: {ticker}{earnings_line}\n\nHeadlines:\n{headline_text}\n\n"
        "Provide your JSON analysis."
    )

    try:
        response = ollama.chat(
            model=settings.ollama_model,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": prompt},
            ],
            format="json",
            options={"temperature": 0.0},
        )
        data = json.loads(response.message.content)
        ctx = NewsContext(
            sentiment_score=float(data.get("sentiment_score", 0.0)),
            risk_level=str(data.get("risk_level", "low")).lower(),
            flags=[str(f) for f in data.get("flags", [])],
            summary=str(data.get("summary", "")),
        )
        log.info(
            f"[NewsAnalyst] {ticker}: sentiment={ctx.sentiment_score:+.2f} "
            f"risk={ctx.risk_level} flags={ctx.flags}"
        )
        return ctx
    except Exception as e:
        log.warning(f"[NewsAnalyst] {ticker}: LLM call failed — {e}")
        return _NEUTRAL
