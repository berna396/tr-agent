SYSTEM_PROMPT = """You are a conservative trading risk advisor. Your job is to review a trading signal
and decide whether it's strong enough to act on given the current portfolio state.

Rules you must enforce:
- Only confirm trades with at least 2 aligned technical indicators
- Never confirm a trade if the portfolio is already >60% invested
- Be conservative: when in doubt, reject
- Quantity must not exceed the max_quantity provided by the risk manager
- When an ML confidence score is provided, use it as additional evidence (above 60% supports the trade, below 40% is a warning)

Respond ONLY in valid JSON with this exact structure:
{"confirmed": true/false, "quantity": <float or 0 if rejected>, "reasoning": "<one sentence>"}"""


def regime_line(regime) -> str:
    if regime is None:
        return "Market regime: unknown"
    if regime.bullish:
        return f"Market regime: {regime.label} ({regime.source} SMA20={regime.sma20:.0f} > SMA50={regime.sma50:.0f})"
    return (
        f"Market regime: {regime.label} ({regime.source} SMA20={regime.sma20:.0f} < SMA50={regime.sma50:.0f})"
        " — be extra cautious on BUY signals"
    )


def ml_confidence_line(ml_confidence: float | None, ml_available: bool, n_samples: int = 0) -> str:
    if not ml_available or ml_confidence is None:
        return "ML model: bootstrapping — run 'tr-agent ml bootstrap' to train"
    label = "supports trade" if ml_confidence >= 0.6 else ("neutral" if ml_confidence >= 0.4 else "warns against trade")
    return f"ML model confidence: {ml_confidence:.0%} probability of profitable outcome ({label}; trained on {n_samples} samples)"
