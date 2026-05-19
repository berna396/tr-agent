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
        signal_rows = con.execute(
            "SELECT ts, ticker, data FROM cycle_events WHERE event_type='signal'"
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
        "rsi_entry_quality": _rsi_entry_quality(df, signal_rows),
        "stop_loss_analysis": _stop_loss_analysis(df),
    }


def _rsi_entry_quality(outcomes: pd.DataFrame, signal_rows) -> dict:
    """
    Bucket BUY entries by RSI at signal time and compare win rate + avg P&L.
    Answers: did entries at RSI<30 outperform those at RSI 30-35?
    """
    # Index signals by ticker → sorted list of (ts, rsi)
    sig_by_ticker: dict[str, list[tuple[str, float]]] = {}
    for row in signal_rows:
        try:
            data = json.loads(row["data"])
            rsi = data.get("rsi")
            if rsi is not None:
                sig_by_ticker.setdefault(row["ticker"], []).append((row["ts"], float(rsi)))
        except Exception:
            continue

    buckets: dict[str, list[float]] = {"<30": [], "30-35": [], ">=35": []}

    for _, trade in outcomes.iterrows():
        ticker = trade["ticker"]
        buy_ts = trade["buy_ts"]
        sigs = sig_by_ticker.get(ticker, [])
        pre = [(ts, rsi) for ts, rsi in sigs if ts <= buy_ts]
        if not pre:
            continue
        _, entry_rsi = max(pre, key=lambda x: x[0])

        if entry_rsi < 30:
            key = "<30"
        elif entry_rsi < 35:
            key = "30-35"
        else:
            key = ">=35"
        buckets[key].append(float(trade["pnl_pct"]))

    result = {}
    for label, pnl_list in buckets.items():
        if not pnl_list:
            result[label] = {"trades": 0}
            continue
        wins = sum(1 for p in pnl_list if p > 0)
        result[label] = {
            "trades": len(pnl_list),
            "win_rate": round(wins / len(pnl_list) * 100, 1),
            "avg_pnl_pct": round(float(np.mean(pnl_list)), 2),
        }
    return result


def _stop_loss_analysis(outcomes: pd.DataFrame) -> dict:
    """
    Measure how often the 5% stop-loss floor is triggering vs organic sell signals.
    Stop-loss exits are identified by sell_reasoning starting with 'stop-loss'.
    """
    total = len(outcomes)
    sl_mask = outcomes["sell_reasoning"].fillna("").str.startswith("stop-loss")
    sl_count = int(sl_mask.sum())
    organic_count = total - sl_count

    result: dict = {
        "total_sells": total,
        "stop_loss_exits": sl_count,
        "organic_exits": organic_count,
        "stop_loss_rate_pct": round(sl_count / total * 100, 1) if total else 0.0,
    }
    if sl_count:
        sl_losses = outcomes.loc[sl_mask, "pnl_pct"]
        result["avg_stop_loss_pct"] = round(float(sl_losses.mean()), 2)
    return result


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

    rsi_quality = report.get("rsi_entry_quality", {})
    rsi_lines = "\n".join(
        f"  RSI {bucket}: {d['trades']} trades, {d.get('win_rate', 0)}% win rate, avg {d.get('avg_pnl_pct', 0):+.2f}%"
        for bucket, d in rsi_quality.items()
        if d.get("trades", 0) > 0
    ) or "  Not enough data"

    sl = report.get("stop_loss_analysis", {})
    sl_line = (
        f"  Stop-loss exits: {sl.get('stop_loss_exits', 0)}/{sl.get('total_sells', 0)} "
        f"({sl.get('stop_loss_rate_pct', 0)}%), avg loss {sl.get('avg_stop_loss_pct', 0):+.2f}%"
        if sl
        else "  Not available"
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

RSI ENTRY QUALITY:
{rsi_lines}

STOP-LOSS ANALYSIS:
{sl_line}

TOP PREDICTIVE FEATURES (SHAP):
{shap_lines}
{model_section}
Analyze what patterns are working, whether RSI entry thresholds are well-calibrated, whether the stop-loss is triggering too aggressively, and suggest one concrete parameter adjustment."""

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


def generate_rules_md(
    ollama_model: str,
    report: dict,
    shap_importances: dict[str, float],
) -> str:
    """
    Ask Ollama to synthesize performance stats into structured, actionable rules
    for the LLM confirmation agent. Returns a markdown string to be saved as
    data/llm_rules.md and injected into future trade confirmation prompts.
    """
    import ollama

    if report.get("total_trades", 0) == 0:
        from datetime import date
        return f"## Learned Rules (generated {date.today()} · no trades yet)\n\nNo completed trades. Rules will be generated after the first closed positions."

    ticker_lines = "\n".join(
        f"  {t}: {d['trades']} trades, {d['win_rate']}% win rate, avg {d['avg_pnl_pct']:+.2f}%"
        for t, d in report.get("by_ticker", {}).items()
    ) or "  No per-ticker data"

    rsi_quality = report.get("rsi_entry_quality", {})
    rsi_lines = "\n".join(
        f"  RSI {b}: {d['trades']} trades, {d.get('win_rate', 0)}% win rate, avg {d.get('avg_pnl_pct', 0):+.2f}%"
        for b, d in rsi_quality.items() if d.get("trades", 0) > 0
    ) or "  Not enough data"

    sl = report.get("stop_loss_analysis", {})
    sl_line = (
        f"  {sl.get('stop_loss_exits', 0)}/{sl.get('total_sells', 0)} exits were stop-losses "
        f"({sl.get('stop_loss_rate_pct', 0)}%), avg loss {sl.get('avg_stop_loss_pct', 0):+.2f}%"
        if sl else "  Not available"
    )

    shap_lines = "\n".join(
        f"  {i+1}. {feat}: {imp:.4f}"
        for i, (feat, imp) in enumerate(list(shap_importances.items())[:5])
    ) or "  Not available"

    from datetime import date
    prompt = f"""You are analyzing a paper trading agent's 30-day performance to generate concise, actionable rules.
These rules will be injected into the agent's LLM trade confirmation prompt each time it evaluates a new signal.
Write rules the LLM can directly apply to improve future decisions.

PERFORMANCE DATA ({report.get('days', 30)} days, {report.get('total_trades', 0)} trades):
  Overall win rate: {report.get('win_rate', 0)}%
  Total P&L: ${report.get('total_pnl', 0):+.2f}
  Avg return: {report.get('avg_pnl_pct', 0):+.2f}%
  Best: {report.get('best_trade_pct', 0):+.2f}% | Worst: {report.get('worst_trade_pct', 0):+.2f}%

BY TICKER:
{ticker_lines}

RSI ENTRY QUALITY:
{rsi_lines}

STOP-LOSS ANALYSIS:
{sl_line}

TOP SHAP FEATURES:
{shap_lines}

Generate a rules file with EXACTLY this markdown structure — keep each rule to one line, max 10 rules total:

## Learned Rules (generated {date.today()} · {report.get('total_trades', 0)} trades · {report.get('days', 30)}-day window)

### What's working:
- [rule based on data]

### What to avoid:
- [rule based on data]

### Suggested constraints:
- [specific, actionable constraint for the LLM to apply]

Be specific and data-driven. Reference actual percentages from the data above."""

    try:
        response = ollama.chat(
            model=ollama_model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3},
        )
        return response["message"]["content"].strip()
    except Exception as e:
        log.error(f"[ML] Rules generation failed: {e}")
        return f"## Learned Rules\n\nRules generation unavailable: {e}"


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

    rsi_quality = report.get("rsi_entry_quality", {})
    if any(d.get("trades", 0) > 0 for d in rsi_quality.values()):
        rsi_parts = []
        for bucket, d in rsi_quality.items():
            if d.get("trades", 0) > 0:
                rsi_parts.append(
                    f"RSI {bucket}: {d['trades']}t, {d.get('win_rate', 0)}% WR, {d.get('avg_pnl_pct', 0):+.2f}%"
                )
        lines.append(f"\n*RSI Entry Quality*\n" + " | ".join(rsi_parts))

    sl = report.get("stop_loss_analysis", {})
    if sl and sl.get("total_sells", 0) > 0:
        avg_str = f", avg {sl['avg_stop_loss_pct']:+.2f}%" if "avg_stop_loss_pct" in sl else ""
        lines.append(
            f"\n*Stop-Loss*: {sl['stop_loss_exits']}/{sl['total_sells']} exits "
            f"({sl['stop_loss_rate_pct']}%{avg_str})"
        )

    if insights:
        lines.append(f"\n*Insights*\n{insights}")

    return "\n".join(lines)
