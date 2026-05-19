import logging
from dataclasses import dataclass
from pathlib import Path

from tr_agent.broker.base import OrderSide, Portfolio, Quote
from tr_agent.signals.technical import Signal, TechnicalAnalysis

log = logging.getLogger(__name__)

MAX_TRADE_PCT = 0.20      # max 20% of cash per trade (hard cap even with Kelly)
MAX_INVESTED_PCT = 0.60   # max 60% of total portfolio invested
MIN_QUANTITY = 0.01        # minimum order size

_DB_PATH = Path(__file__).parents[2] / "data" / "journal.db"


def _kelly_fraction(analysis: TechnicalAnalysis) -> float:
    """Return trade size as fraction of cash. Uses half-Kelly when possible, else MAX_TRADE_PCT."""
    from tr_agent.config import settings
    if not settings.kelly_sizing or analysis.ml_confidence is None:
        return MAX_TRADE_PCT
    try:
        from tr_agent.ml.kelly import compute_kelly_fraction, get_historical_stats
        stats = get_historical_stats(_DB_PATH)
        if stats is None:
            return MAX_TRADE_PCT
        avg_win, avg_loss = stats
        fraction = compute_kelly_fraction(analysis.ml_confidence, avg_win, avg_loss)
        return min(fraction, MAX_TRADE_PCT)
    except Exception:
        return MAX_TRADE_PCT


@dataclass
class RiskCheck:
    approved: bool
    reason: str
    max_quantity: float = 0.0
    side: OrderSide = OrderSide.BUY


def evaluate(
    analysis: TechnicalAnalysis,
    portfolio: Portfolio,
    quote: Quote,
) -> RiskCheck:
    """
    Validates whether a signal is safe to trade given the current portfolio state.
    Returns a RiskCheck with max_quantity if approved.
    """
    signal = analysis.signal
    ticker = analysis.ticker

    if signal == Signal.NEUTRAL:
        return RiskCheck(approved=False, reason="Signal is NEUTRAL — nothing to do")

    total_value = portfolio.cash + sum(p.cost_basis for p in portfolio.positions.values())

    if signal == Signal.BUY:
        # Rule: portfolio must not be over-invested
        invested = sum(p.cost_basis for p in portfolio.positions.values())
        invested_pct = invested / total_value if total_value > 0 else 0
        if invested_pct >= MAX_INVESTED_PCT:
            return RiskCheck(
                approved=False,
                reason=f"Portfolio {invested_pct:.0%} invested — limit is {MAX_INVESTED_PCT:.0%}",
            )

        # Rule: max trade size — Kelly-adjusted if enabled, else fixed pct
        trade_pct = _kelly_fraction(analysis)
        max_trade_value = portfolio.cash * trade_pct
        max_quantity = max_trade_value / quote.price

        if max_quantity < MIN_QUANTITY:
            return RiskCheck(approved=False, reason=f"Insufficient cash for minimum order (${max_trade_value:.2f})")

        log.info(
            f"[Risk] BUY {ticker}: max {max_quantity:.4f} shares @ ${quote.price:.2f} "
            f"(${max_trade_value:.2f}, {trade_pct:.0%} of cash)"
        )
        return RiskCheck(
            approved=True,
            reason=f"Max trade: {max_quantity:.4f} shares ({trade_pct:.0%} of cash = ${max_trade_value:.2f})",
            max_quantity=max_quantity,
            side=OrderSide.BUY,
        )

    else:  # SELL
        # Rule: must have a position to sell
        if ticker not in portfolio.positions:
            return RiskCheck(approved=False, reason=f"No open position in {ticker} to sell")

        position = portfolio.positions[ticker]
        log.info(f"[Risk] SELL {ticker}: {position.quantity:.4f} shares @ avg ${position.avg_price:.2f}")
        return RiskCheck(
            approved=True,
            reason=f"Selling full position: {position.quantity:.4f} shares",
            max_quantity=position.quantity,
            side=OrderSide.SELL,
        )
