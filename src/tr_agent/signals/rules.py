"""
Motor de reglas de negocio independiente del LLM.
Útil para backtesting y para validar que el agente actúa de forma coherente.
"""
from dataclasses import dataclass
from typing import Callable

from tr_agent.signals.technical import Signal, TechnicalAnalysis


@dataclass
class Rule:
    name: str
    condition: Callable[[TechnicalAnalysis], bool]
    action: Signal
    description: str


DEFAULT_RULES: list[Rule] = [
    Rule(
        name="oversold_buy",
        condition=lambda t: t.rsi is not None and t.rsi < 30,
        action=Signal.BUY,
        description="Compra cuando RSI < 30 (sobrevendido extremo)",
    ),
    Rule(
        name="overbought_sell",
        condition=lambda t: t.rsi is not None and t.rsi > 70,
        action=Signal.SELL,
        description="Vende cuando RSI > 70 (sobrecomprado extremo)",
    ),
    Rule(
        name="golden_cross_buy",
        condition=lambda t: (
            t.sma_20 is not None and t.sma_50 is not None and t.sma_20 > t.sma_50
        ),
        action=Signal.BUY,
        description="Compra en golden cross (SMA20 > SMA50)",
    ),
    Rule(
        name="death_cross_sell",
        condition=lambda t: (
            t.sma_20 is not None and t.sma_50 is not None and t.sma_20 < t.sma_50
        ),
        action=Signal.SELL,
        description="Vende en death cross (SMA20 < SMA50)",
    ),
]


def evaluate(analysis: TechnicalAnalysis, rules: list[Rule] = DEFAULT_RULES) -> list[Rule]:
    """Devuelve las reglas que se activan para el análisis dado."""
    return [rule for rule in rules if rule.condition(analysis)]
