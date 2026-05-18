import logging
import pickle
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from tr_agent.ml.features import FEATURE_NAMES

log = logging.getLogger(__name__)

_DEFAULT_PARAMS = {"n_estimators": 200, "num_leaves": 31, "learning_rate": 0.05}
_TUNE_MIN_SAMPLES = 200


def tune_hyperparams(X: pd.DataFrame, y: pd.Series) -> dict:
    """Grid-search best LightGBM params using time-series CV. Returns defaults if too few samples."""
    if len(X) < _TUNE_MIN_SAMPLES:
        log.info(f"[ML] Skipping hyperparameter tuning — {len(X)} samples (need {_TUNE_MIN_SAMPLES}+)")
        return _DEFAULT_PARAMS.copy()

    param_grid = {
        "n_estimators": [100, 200, 300],
        "num_leaves": [15, 31, 63],
        "learning_rate": [0.01, 0.05, 0.1],
    }
    base = LGBMClassifier(class_weight="balanced", random_state=42, verbose=-1)
    cv = TimeSeriesSplit(n_splits=3)
    gs = GridSearchCV(base, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1)
    gs.fit(X[FEATURE_NAMES], y)
    best = {k: gs.best_params_[k] for k in param_grid}
    log.info(f"[ML] Tuned params: {best} (CV AUC={gs.best_score_:.3f})")
    return best


class SignalModel:
    def __init__(self) -> None:
        self.model: Optional[LGBMClassifier] = None
        self.version: int = 0
        self.auc: Optional[float] = None
        self.train_date: Optional[str] = None
        self.n_samples: int = 0

    def train(self, X: pd.DataFrame, y: pd.Series, params: Optional[dict] = None) -> None:
        p = params or {}
        self.model = LGBMClassifier(
            n_estimators=p.get("n_estimators", 200),
            learning_rate=p.get("learning_rate", 0.05),
            num_leaves=p.get("num_leaves", 31),
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        )
        self.model.fit(X[FEATURE_NAMES], y)
        self.n_samples = len(y)
        self.train_date = datetime.now(timezone.utc).isoformat()

    def predict_proba(self, features: dict) -> Optional[float]:
        if self.model is None:
            return None
        x = pd.DataFrame([[features.get(f, 0.0) for f in FEATURE_NAMES]], columns=FEATURE_NAMES)
        try:
            return float(self.model.predict_proba(x)[0, 1])
        except Exception:
            return None

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        from sklearn.metrics import precision_score, recall_score, roc_auc_score

        if self.model is None or X.empty:
            return {}
        proba = self.model.predict_proba(X[FEATURE_NAMES])[:, 1]
        preds = (proba >= 0.5).astype(int)
        return {
            "auc": float(roc_auc_score(y, proba)),
            "precision": float(precision_score(y, preds, zero_division=0)),
            "recall": float(recall_score(y, preds, zero_division=0)),
            "n_samples": int(len(y)),
        }

    def save(self, path: Path, version: int) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.version = version
        payload = {
            "model": self.model,
            "version": version,
            "auc": self.auc,
            "train_date": self.train_date,
            "n_samples": self.n_samples,
            "feature_names": FEATURE_NAMES,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        versioned = path.parent / f"signal_model_v{version}.pkl"
        shutil.copy(path, versioned)
        self._prune_old_versions(path.parent, version)
        log.info(f"[ML] Model v{version} saved → {path}")

    @classmethod
    def load(cls, path: Path) -> Optional["SignalModel"]:
        path = Path(path)
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)
            m = cls()
            m.model = payload["model"]
            m.version = payload.get("version", 1)
            m.auc = payload.get("auc")
            m.train_date = payload.get("train_date")
            m.n_samples = payload.get("n_samples", 0)
            return m
        except Exception as e:
            log.warning(f"[ML] Failed to load model from {path}: {e}")
            return None

    @staticmethod
    def _prune_old_versions(model_dir: Path, current_version: int, keep: int = 3) -> None:
        for v in range(1, current_version - keep):
            old = model_dir / f"signal_model_v{v}.pkl"
            if old.exists():
                old.unlink()
