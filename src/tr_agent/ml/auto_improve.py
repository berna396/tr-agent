import logging
from pathlib import Path

from tr_agent.config import DEFAULT_WATCHLIST, settings
from tr_agent.ml import analyzer, trainer
from tr_agent.ml.signal_model import SignalModel

log = logging.getLogger(__name__)

_DB_PATH = Path(__file__).parents[3] / "data" / "journal.db"
_MODEL_PATH = Path(__file__).parents[3] / "data" / "models" / "signal_model.pkl"
_HISTORY_PATH = Path(__file__).parents[3] / "data" / "models" / "training_history.json"
_RULES_PATH = Path(__file__).parents[3] / "data" / "llm_rules.md"

_CRYPTO_MODEL_PATH = Path(__file__).parents[3] / "data" / "models" / "crypto_signal_model.pkl"
_CRYPTO_HISTORY_PATH = Path(__file__).parents[3] / "data" / "models" / "crypto_training_history.json"
_CRYPTO_RULES_PATH = Path(__file__).parents[3] / "data" / "crypto_llm_rules.md"


def daily_ml_check(tickers: list[str] | None = None) -> None:
    """Check if enough new live data has arrived; retrain and deploy if so."""
    tickers = tickers or DEFAULT_WATCHLIST

    if not trainer.should_retrain(_DB_PATH, _HISTORY_PATH, settings.ml_min_new_samples):
        log.info("[ML] Daily check — not enough new samples to retrain")
        return

    log.info("[ML] Retraining model on new live data...")
    report = trainer.train_and_deploy(
        tickers, _DB_PATH, _MODEL_PATH, _HISTORY_PATH, settings.ml_backtest_period
    )

    if report.get("deployed"):
        model = SignalModel.load(_MODEL_PATH)
        shap = {}
        if model:
            from tr_agent.ml.dataset import build_full_dataset
            X, _ = build_full_dataset(tickers, _DB_PATH, settings.ml_backtest_period)
            if not X.empty:
                shap = analyzer.compute_shap_importances(model, X)

        top_features = list(shap.keys())[:3]
        msg = (
            f"*ML Model Retrained* → v{report['version']}\n"
            f"CV AUC: {report['cv_auc']:.3f} | Samples: {report['n_samples']}\n"
            f"Top features: {', '.join(top_features) or 'n/a'}"
        )
        from tr_agent import notifier
        notifier.send(msg)
    else:
        log.info(f"[ML] Retrain skipped: {report.get('reason', 'unknown')}")


def daily_ml_check_crypto() -> None:
    """Check if enough new crypto live data has arrived; retrain and deploy if so."""
    tickers = settings.crypto_watchlist
    if not tickers:
        return

    if not trainer.should_retrain(_DB_PATH, _CRYPTO_HISTORY_PATH, settings.ml_min_new_samples):
        log.info("[ML] Crypto daily check — not enough new samples to retrain")
        return

    log.info("[ML] Retraining crypto model on new live data...")
    report = trainer.train_and_deploy(
        tickers, _DB_PATH, _CRYPTO_MODEL_PATH, _CRYPTO_HISTORY_PATH, settings.ml_backtest_period
    )

    if report.get("deployed"):
        model = SignalModel.load(_CRYPTO_MODEL_PATH)
        shap = {}
        if model:
            from tr_agent.ml.dataset import build_full_dataset
            X, _ = build_full_dataset(tickers, _DB_PATH, settings.ml_backtest_period)
            if not X.empty:
                shap = analyzer.compute_shap_importances(model, X)

        top_features = list(shap.keys())[:3]
        msg = (
            f"*Crypto ML Model Retrained* → v{report['version']}\n"
            f"CV AUC: {report['cv_auc']:.3f} | Samples: {report['n_samples']}\n"
            f"Top features: {', '.join(top_features) or 'n/a'}"
        )
        from tr_agent import notifier
        notifier.send(msg)
    else:
        log.info(f"[ML] Crypto retrain skipped: {report.get('reason', 'unknown')}")


def weekly_analysis_crypto() -> None:
    """Build a 30-day crypto performance report and send Ollama insights."""
    tickers = settings.crypto_watchlist
    if not tickers:
        return
    log.info("[ML] Running crypto weekly analysis...")

    report = analyzer.build_performance_report(_DB_PATH, days=30, tickers=tickers)

    shap = {}
    model = SignalModel.load(_CRYPTO_MODEL_PATH)
    if model:
        from tr_agent.ml.dataset import build_full_dataset
        X, _ = build_full_dataset(tickers, _DB_PATH, settings.ml_backtest_period)
        if not X.empty:
            shap = analyzer.compute_shap_importances(model, X)

    insights = analyzer.generate_ollama_insights(settings.ollama_model, report, shap)

    rules_md = analyzer.generate_rules_md(settings.ollama_model, report, shap)
    _CRYPTO_RULES_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CRYPTO_RULES_PATH.write_text(rules_md)
    log.info(f"[ML] Crypto LLM rules updated → {_CRYPTO_RULES_PATH}")

    message = analyzer.format_telegram_message(report, insights)
    from tr_agent import notifier
    notifier.send(message)
    log.info("[ML] Crypto weekly analysis sent")


def weekly_analysis(tickers: list[str] | None = None) -> None:
    """Build a 30-day performance report and send Ollama insights via Telegram."""
    tickers = tickers or DEFAULT_WATCHLIST
    log.info("[ML] Running weekly analysis...")

    report = analyzer.build_performance_report(_DB_PATH, days=30)

    shap = {}
    model = SignalModel.load(_MODEL_PATH)
    if model:
        from tr_agent.ml.dataset import build_full_dataset
        X, _ = build_full_dataset(tickers, _DB_PATH, settings.ml_backtest_period)
        if not X.empty:
            shap = analyzer.compute_shap_importances(model, X)

    insights = analyzer.generate_ollama_insights(
        settings.ollama_model, report, shap
    )

    rules_md = analyzer.generate_rules_md(settings.ollama_model, report, shap)
    _RULES_PATH.parent.mkdir(parents=True, exist_ok=True)
    _RULES_PATH.write_text(rules_md)
    log.info(f"[ML] LLM rules updated → {_RULES_PATH}")

    message = analyzer.format_telegram_message(report, insights)

    from tr_agent import notifier
    notifier.send(message)
    log.info("[ML] Weekly analysis sent")
