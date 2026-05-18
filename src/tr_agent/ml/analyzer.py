import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from tr_agent.ml.features import FEATURE_NAMES
from tr_agent.ml.signal_model import SignalModel

log = logging.getLogger(__name__)


def compute_shap_importances(model: SignalModel, X: pd.DataFrame) -> dict[str, float]:
    """Return mean |SHAP| per feature, sorted descending."""
    if model.model is None or X.empty:
        return {}
    try:
        import shap
        explainer = shap.TreeExplainer(model.model)
        shap_values = explainer.shap_values(X[FEATURE_NAMES])
        # LightGBM binary: shap_values is list[class0, class1] or 2D array
        if isinstance(shap_values, list):
            arr = np.array(shap_values[1])
        else:
            arr = np.array(shap_values)
        if arr.ndim == 3:
            arr = arr[:, :, 1]
        mean_abs = np.abs(arr).mean(axis=0)
        return dict(sorted(zip(FEATURE_NAMES, mean_abs.tolist()), key=lambda x: -x[1]))
    except Exception as e:
        log.warning(f"[ML] SHAP computation failed: {e}")
        return {}


def build_performance_report(db_path: Path, days: int = 30) -> dict:
    """Aggregate trade outcomes from the last N days."""
    db_path = Path(db_path)
    if not db_path.exists():
        return {"error": "no journal data yet", "days": days}

    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT * FROM trade_outcomes WHERE sell_ts >= ?", (cutoff,)
        ).fetchall()

    if not rows:
        return {"total_trades": 0, "days": days, "message": "no completed trades in window"}

    df = pd.DataFrame([dict(r) for r in rows])
    total = len(df)
    wins = int((df["pnl"] > 0).sum())
    total_pnl = float(df["pnl"].sum())
    avg_pnl_pct = float(df["pnl_pct"].mean())
    best = float(df["pnl_pct"].max())
    worst = float(df["pnl_pct"].min())

    by_ticker = {}
    for ticker, group in df.groupby("ticker"):
        t_wins = int((group["pnl"] > 0).sum())
        by_ticker[ticker] = {
            "trades": len(group),
            "wins": t_wins,
            "losses": len(group) - t_wins,
            "win_rate": round(t_wins / len(group) * 100, 1),
            "avg_pnl_pct": round(float(group["pnl_pct"].mean()), 2),
            "total_pnl": round(float(group["pnl"].sum()), 2),
        }

    return {
        "days": days,
        "total_trades": total,
        "wins": wins,
        "losses": total - wins,
        "win_rate": round(wins / total * 100, 1),
        "total_pnl": round(total_pnl, 2),
        "avg_pnl_pct": round(avg_pnl_pct, 2),
        "best_trade_pct": round(best, 2),
        "worst_trade_pct": round(worst, 2),
        "by_ticker": by_ticker,
    }


def generate_ollama_insights(
    ollama_model: str,
    report: dict,
    shap_importances: dict[str, float],
    training_report: Optional[dict] = None,
) -> str:
    """Call Ollama to generate a human-readable insight narrative."""
    import ollama

    shap_lines = "\n".join(
        f"  {i+1}. {feat}: {imp:.4f}"
        for i, (feat, imp) in enumerate(list(shap_importances.items())[:6])
    ) or "  Not available"

    ticker_lines = "\n".join(
        f"  {t}: {d['trades']} trades, {d['win_rate']}% win rate, avg {d['avg_pnl_pct']:+.2f}% P&L"
        for t, d in report.get("by_ticker", {}).items()
    ) or "  No data"

    model_section = ""
    if training_report and training_report.get("deployed"):
        model_section = (
            f"\nML MODEL UPDATE:\n"
            f"  New version: v{training_report['version']}\n"
            f"  CV AUC: {training_report['cv_auc']:.3f}\n"
            f"  Training samples: {training_report['n_samples']}\n"
        )

    prompt = f"""You are analyzing performance data for an AI paper trading agent. Provide a concise analysis (4-6 sentences).

LAST {report.get('days', 30)} DAYS PERFORMANCE:
  Total trades: {report.get('total_trades', 0)}
  Win rate: {report.get('win_rate', 0)}%
  Total P&L: ${report.get('total_pnl', 0):+.2f}
  Avg return per trade: {report.get('avg_pnl_pct', 0):+.2f}%
  Best trade: {report.get('best_trade_pct', 0):+.2f}%
  Worst trade: {report.get('worst_trade_pct', 0):+.2f}%

BY TICKER:
{ticker_lines}

TOP PREDICTIVE FEATURES (SHAP):
{shap_lines}
{model_section}
Analyze what patterns are working, what signals are most predictive, and suggest one concrete adjustment."""

    try:
        response = ollama.chat(
            model=ollama_model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.4},
        )
        return response["message"]["content"].strip()
    except Exception as e:
        log.error(f"[ML] Ollama insight generation failed: {e}")
        return f"Insight generation unavailable: {e}"


def format_telegram_message(
    report: dict,
    insights: str,
    training_report: Optional[dict] = None,
) -> str:
    lines = [f"*Weekly ML Analysis*\n"]

    if "error" in report or report.get("total_trades", 0) == 0:
        lines.append("No completed trades in the last 30 days.")
    else:
        lines.append(
            f"*{report['days']}d Performance*\n"
            f"Trades: {report['total_trades']} | Win rate: {report['win_rate']}% | "
            f"P&L: ${report['total_pnl']:+.2f} | Avg: {report['avg_pnl_pct']:+.2f}%"
        )

    if training_report and training_report.get("deployed"):
        lines.append(
            f"\n*Model Updated* → v{training_report['version']}\n"
            f"CV AUC: {training_report['cv_auc']:.3f} | "
            f"Samples: {training_report['n_samples']}"
        )

    if insights:
        lines.append(f"\n*Insights*\n{insights}")

    return "\n".join(lines)
