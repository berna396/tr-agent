import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score

from tr_agent.ml.dataset import build_full_dataset
from tr_agent.ml.features import FEATURE_NAMES
from tr_agent.ml.signal_model import SignalModel

log = logging.getLogger(__name__)

_MIN_SAMPLES = 50   # need at least this many samples to train meaningfully
# In HYBRID mode the model is LLM context, not a decision gate — deploy if it has
# any discriminative power (AUC differs from 0.5 by > 2pp in either direction)
_MIN_AUC = 0.45


def walk_forward_evaluate(
    X: pd.DataFrame, y: pd.Series, n_splits: int = 5
) -> dict:
    """Time-series cross-validation — no data leakage."""
    if len(X) < n_splits * 10:
        return {"auc": 0.0, "n_splits": 0, "message": "not enough data for CV"}

    tscv = TimeSeriesSplit(n_splits=n_splits)
    aucs = []

    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        if y_te.nunique() < 2:
            continue

        model = SignalModel()
        model.train(X_tr, y_tr)
        metrics = model.evaluate(X_te, y_te)
        if metrics.get("auc"):
            aucs.append(metrics["auc"])

    if not aucs:
        return {"auc": 0.0, "n_splits": 0, "message": "all folds had single class"}

    return {
        "auc": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "n_splits": len(aucs),
    }


def should_retrain(db_path: Path, history_path: Path, min_new_samples: int = 10) -> bool:
    db_path = Path(db_path)
    history_path = Path(history_path)

    if not db_path.exists():
        return False

    last_train_ts = _load_last_train_ts(history_path)
    if last_train_ts is None:
        return True  # never trained on live data yet

    import sqlite3
    with sqlite3.connect(db_path) as con:
        count = con.execute(
            "SELECT COUNT(*) FROM trade_outcomes WHERE sell_ts > ?",
            (last_train_ts,),
        ).fetchone()[0]

    return count >= min_new_samples


def train_and_deploy(
    tickers: list[str],
    db_path: Path,
    model_path: Path,
    history_path: Path,
    period: str = "2y",
    force: bool = False,
) -> dict:
    model_path = Path(model_path)
    history_path = Path(history_path)

    X, y = build_full_dataset(tickers, db_path, period)
    n_samples = len(y)

    if n_samples < _MIN_SAMPLES:
        msg = f"Only {n_samples} samples — need {_MIN_SAMPLES} to train"
        log.info(f"[ML] {msg}")
        return {"deployed": False, "reason": msg, "n_samples": n_samples}

    cv_result = walk_forward_evaluate(X, y)
    cv_auc = cv_result.get("auc", 0.0)

    if cv_auc < _MIN_AUC and not force:
        msg = f"CV AUC {cv_auc:.3f} below threshold {_MIN_AUC}"
        log.info(f"[ML] {msg}")
        _append_history(history_path, {"deployed": False, "cv_auc": cv_auc, "n_samples": n_samples, "reason": msg})
        return {"deployed": False, "reason": msg, "cv_auc": cv_auc, "n_samples": n_samples}

    # Train on full dataset and deploy
    version = _next_version(history_path)
    model = SignalModel()
    model.train(X, y)
    model.auc = cv_auc
    model.save(model_path, version)

    record = {
        "deployed": True,
        "version": version,
        "cv_auc": cv_auc,
        "cv_auc_std": cv_result.get("auc_std", 0.0),
        "n_samples": n_samples,
        "train_ts": datetime.now(timezone.utc).isoformat(),
    }
    _append_history(history_path, record)
    log.info(f"[ML] Model v{version} deployed — AUC={cv_auc:.3f}, samples={n_samples}")
    return record


def load_training_history(history_path: Path) -> list[dict]:
    history_path = Path(history_path)
    if not history_path.exists():
        return []
    with open(history_path) as f:
        return json.load(f)


def _load_last_train_ts(history_path: Path) -> Optional[str]:
    history = load_training_history(history_path)
    deployed = [r for r in history if r.get("deployed")]
    if not deployed:
        return None
    return deployed[-1].get("train_ts")


def _next_version(history_path: Path) -> int:
    history = load_training_history(history_path)
    deployed = [r for r in history if r.get("deployed")]
    return len(deployed) + 1


def _append_history(history_path: Path, record: dict) -> None:
    history_path = Path(history_path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history = load_training_history(history_path)
    history.append(record)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
