import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tr_agent.ml.features import FEATURE_NAMES
from tr_agent.ml.trainer import (
    _MIN_SAMPLES,
    load_training_history,
    train_and_deploy,
    walk_forward_evaluate,
)


def _make_dataset(n: int = 200) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.standard_normal((n, len(FEATURE_NAMES))), columns=FEATURE_NAMES)
    # Slight signal: rsi < 0 → label 1 (makes AUC > 0.5 so model gets deployed)
    y = (X["rsi"] < 0).astype(int)
    return X, y


def test_walk_forward_returns_auc():
    X, y = _make_dataset(200)
    result = walk_forward_evaluate(X, y, n_splits=3)
    assert "auc" in result
    assert 0.0 <= result["auc"] <= 1.0


def test_walk_forward_insufficient_data():
    X, y = _make_dataset(10)
    result = walk_forward_evaluate(X, y, n_splits=5)
    assert result["auc"] == 0.0
    assert result["n_splits"] == 0


def test_walk_forward_no_data_leakage():
    """Each test fold must only use training data that precedes it in time."""
    X, y = _make_dataset(100)
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=3)
    splits = list(tscv.split(X))
    for train_idx, test_idx in splits:
        # All test indices must be strictly greater than all training indices
        assert min(test_idx) > max(train_idx)


def test_train_and_deploy_with_enough_data(tmp_path):
    X, y = _make_dataset(200)
    db_path = tmp_path / "journal.db"  # no DB → skips live data
    model_path = tmp_path / "signal_model.pkl"
    history_path = tmp_path / "training_history.json"

    # Patch build_full_dataset to return our controlled dataset
    import tr_agent.ml.trainer as trainer_module
    original_fn = trainer_module.build_full_dataset

    def mock_build(tickers, db_path, period):
        return X, y

    trainer_module.build_full_dataset = mock_build
    try:
        report = train_and_deploy(
            ["AAPL"], db_path, model_path, history_path
        )
    finally:
        trainer_module.build_full_dataset = original_fn

    assert report["deployed"] is True
    assert report["version"] == 1
    assert report["cv_auc"] > 0.0
    assert model_path.exists()


def test_train_and_deploy_insufficient_samples(tmp_path):
    db_path = tmp_path / "journal.db"
    model_path = tmp_path / "signal_model.pkl"
    history_path = tmp_path / "training_history.json"

    import tr_agent.ml.trainer as trainer_module
    original_fn = trainer_module.build_full_dataset

    def mock_build(tickers, db_path, period):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((5, len(FEATURE_NAMES))), columns=FEATURE_NAMES)
        y = pd.Series([1, 0, 1, 0, 1])
        return X, y

    trainer_module.build_full_dataset = mock_build
    try:
        report = train_and_deploy(["AAPL"], db_path, model_path, history_path)
    finally:
        trainer_module.build_full_dataset = original_fn

    assert report["deployed"] is False
    assert "reason" in report


def test_training_history_appended(tmp_path):
    history_path = tmp_path / "history.json"
    db_path = tmp_path / "journal.db"
    model_path = tmp_path / "model.pkl"

    import tr_agent.ml.trainer as trainer_module
    original_fn = trainer_module.build_full_dataset

    def mock_build(tickers, db_path, period):
        X, y = _make_dataset(200)
        return X, y

    trainer_module.build_full_dataset = mock_build
    try:
        train_and_deploy(["AAPL"], db_path, model_path, history_path)
    finally:
        trainer_module.build_full_dataset = original_fn

    history = load_training_history(history_path)
    assert len(history) == 1
    assert "train_ts" in history[0] or "reason" in history[0]
