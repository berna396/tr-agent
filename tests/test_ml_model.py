import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tr_agent.ml.features import FEATURE_NAMES
from tr_agent.ml.signal_model import SignalModel, tune_hyperparams, _TUNE_MIN_SAMPLES, _DEFAULT_PARAMS


def _make_dataset(n: int = 100) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((n, len(FEATURE_NAMES))), columns=FEATURE_NAMES)
    y = pd.Series(rng.integers(0, 2, n))
    return X, y


def test_train_and_predict():
    X, y = _make_dataset()
    model = SignalModel()
    model.train(X, y)
    features = {f: float(X.iloc[0][f]) for f in FEATURE_NAMES}
    prob = model.predict_proba(features)
    assert prob is not None
    assert 0.0 <= prob <= 1.0


def test_predict_proba_on_untrained_returns_none():
    model = SignalModel()
    features = {f: 0.0 for f in FEATURE_NAMES}
    assert model.predict_proba(features) is None


def test_evaluate_returns_metrics():
    X, y = _make_dataset(200)
    model = SignalModel()
    model.train(X[:150], y[:150])
    metrics = model.evaluate(X[150:], y[150:])
    assert "auc" in metrics
    assert 0.0 <= metrics["auc"] <= 1.0
    assert "n_samples" in metrics


def test_save_and_load_roundtrip():
    X, y = _make_dataset()
    model = SignalModel()
    model.train(X, y)
    model.auc = 0.65

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "signal_model.pkl"
        model.save(path, version=1)

        loaded = SignalModel.load(path)
        assert loaded is not None
        assert loaded.version == 1
        assert loaded.auc == 0.65
        assert loaded.n_samples == len(y)

        # Predictions should be identical
        features = {f: float(X.iloc[0][f]) for f in FEATURE_NAMES}
        assert model.predict_proba(features) == loaded.predict_proba(features)


def test_load_nonexistent_returns_none():
    result = SignalModel.load(Path("/nonexistent/model.pkl"))
    assert result is None


def test_load_corrupted_file_returns_none(tmp_path):
    bad_file = tmp_path / "bad.pkl"
    bad_file.write_bytes(b"not a pickle")
    result = SignalModel.load(bad_file)
    assert result is None


def test_versioned_copies_created(tmp_path):
    X, y = _make_dataset()
    model = SignalModel()
    model.train(X, y)
    path = tmp_path / "signal_model.pkl"
    model.save(path, version=3)
    assert (tmp_path / "signal_model_v3.pkl").exists()


def test_tune_hyperparams_returns_defaults_on_small_dataset():
    X, y = _make_dataset(n=_TUNE_MIN_SAMPLES - 1)
    result = tune_hyperparams(X, y)
    assert result == _DEFAULT_PARAMS


def test_tune_hyperparams_returns_dict_on_large_dataset():
    X, y = _make_dataset(n=_TUNE_MIN_SAMPLES + 50)
    result = tune_hyperparams(X, y)
    assert isinstance(result, dict)
    assert set(result.keys()) == {"n_estimators", "num_leaves", "learning_rate"}


def test_train_accepts_params():
    X, y = _make_dataset()
    model = SignalModel()
    model.train(X, y, params={"n_estimators": 50, "num_leaves": 15, "learning_rate": 0.1})
    assert model.model is not None
    assert model.model.n_estimators == 50


def test_missing_features_default_to_zero():
    X, y = _make_dataset()
    model = SignalModel()
    model.train(X, y)
    # Pass only partial features — missing ones should default to 0.0
    partial = {"rsi": 45.0}
    prob = model.predict_proba(partial)
    assert prob is not None
    assert 0.0 <= prob <= 1.0
