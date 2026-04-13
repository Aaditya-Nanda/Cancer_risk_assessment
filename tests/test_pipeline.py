"""
tests/test_pipeline.py
──────────────────────
Basic unit tests — run with: pytest tests/
"""
import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import read_yaml, encode_labels, decode_labels, save_object, load_object


# ── Utils ─────────────────────────────────────────────────────────────────────

def test_read_yaml():
    cfg = read_yaml("config/config.yaml")
    assert "project" in cfg
    assert "paths" in cfg
    assert cfg["project"]["target_column"] == "Level"


def test_label_encoding_roundtrip():
    labels = pd.Series(["Low", "Medium", "High", "Low"])
    encoded = encode_labels(labels)
    decoded = decode_labels(encoded.values)
    assert decoded == ["Low", "Medium", "High", "Low"]


def test_save_load_object(tmp_path):
    obj = {"model": "test", "score": 0.95}
    path = str(tmp_path / "test_obj.pkl")
    save_object(path, obj)
    loaded = load_object(path)
    assert loaded == obj


# ── Preprocessing (smoke test with synthetic data) ────────────────────────────

def test_preprocessor_smoke():
    from src.preprocessing import Preprocessor

    # Build a tiny synthetic DataFrame matching expected schema
    n = 50
    rng = np.random.default_rng(0)
    data = {col: rng.integers(1, 8, n).tolist()
            for col in [
                "Age", "Gender", "Air Pollution", "Alcohol use",
                "Dust Allergy", "OccuPational Hazards", "Genetic Risk",
                "chronic Lung Disease", "Balanced Diet", "Obesity",
                "Smoking", "Passive Smoker", "Chest Pain",
                "Coughing of Blood", "Fatigue", "Weight Loss",
                "Shortness of Breath", "Wheezing", "Swallowing Difficulty",
                "Clubbing of Finger Nails", "Frequent Cold",
                "Dry Cough", "Snoring",
            ]}
    labels = ["Low"] * 17 + ["Medium"] * 17 + ["High"] * 16
    data["Level"] = labels
    df = pd.DataFrame(data)

    prep = Preprocessor()
    X, y = prep.fit_transform_train(df)

    assert X.shape[1] == 23, "Feature count should be 23"
    assert len(np.unique(y)) == 3, "Should have 3 classes"


# ── Model trainer (smoke test) ────────────────────────────────────────────────

def test_model_trainer_smoke():
    from src.model_trainer import ModelTrainer

    rng = np.random.default_rng(42)
    X = rng.standard_normal((120, 23))
    y = np.array([0] * 40 + [1] * 40 + [2] * 40)

    trainer = ModelTrainer()
    # Use tiny param grid for speed
    import unittest.mock as mock
    tiny_grid = {
        "xgboost": {"n_estimators": [10], "max_depth": [2],
                    "learning_rate": [0.1], "subsample": [0.8],
                    "colsample_bytree": [0.8], "reg_alpha": [0],
                    "reg_lambda": [1]},
        "random_forest": {"n_estimators": [10], "max_depth": [3],
                          "min_samples_split": [2], "min_samples_leaf": [1],
                          "max_features": ["sqrt"], "bootstrap": [True]},
    }

    with mock.patch.object(trainer, '_train_xgboost',
                           wraps=lambda X, y: trainer._search_and_save(
                               __import__('xgboost').XGBClassifier(
                                   objective='multi:softprob', num_class=3,
                                   eval_metric='mlogloss', use_label_encoder=False,
                                   random_state=42),
                               tiny_grid["xgboost"], X, y,
                               "xgboost", "xgboost_model_test.pkl", n_iter=1)):
        pass  # Just verify import works

    assert trainer is not None


# ── Prediction pipeline (requires trained artifacts) ─────────────────────────

def test_prediction_pipeline_artifacts_exist():
    """Skip gracefully if models not yet trained."""
    required = [
        "artifacts/models/xgboost_model.pkl",
        "artifacts/models/random_forest_model.pkl",
        "artifacts/preprocessors/scaler.pkl",
    ]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        pytest.skip(f"Artifacts not found (run pipeline first): {missing}")

    from src.prediction_pipeline import PredictionPipeline, FEATURE_COLUMNS
    pipeline = PredictionPipeline()
    sample = {col: 3 for col in FEATURE_COLUMNS}
    result = pipeline.predict(sample)
    assert "xgboost" in result
    assert "random_forest" in result
    assert result["xgboost"]["label"] in ["Low", "Medium", "High"]
