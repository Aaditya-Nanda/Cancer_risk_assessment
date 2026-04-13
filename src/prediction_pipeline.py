"""
prediction_pipeline.py
──────────────────────
Loads the saved scaler + both models and runs
single-sample or batch inference at serving time.
Returns prediction + class probabilities for both models.
"""
import os
import sys

import numpy as np
import pandas as pd

from src.logger import get_logger
from src.exception import CancerRiskException
from src.utils import read_yaml, load_object, decode_labels

logger = get_logger(__name__)

CLASS_NAMES = ["Low", "Medium", "High"]

# Feature order must match training
FEATURE_COLUMNS = [
    "Age", "Gender", "Air Pollution", "Alcohol use", "Dust Allergy",
    "OccuPational Hazards", "Genetic Risk", "chronic Lung Disease",
    "Balanced Diet", "Obesity", "Smoking", "Passive Smoker",
    "Chest Pain", "Coughing of Blood", "Fatigue", "Weight Loss",
    "Shortness of Breath", "Wheezing", "Swallowing Difficulty",
    "Clubbing of Finger Nails", "Frequent Cold", "Dry Cough", "Snoring",
]


class PredictionPipeline:
    """
    Usage:
        pipeline = PredictionPipeline()
        result = pipeline.predict(input_dict)
        # result = {
        #   "xgboost":      {"label": "High", "probabilities": {...}},
        #   "random_forest":{"label": "High", "probabilities": {...}},
        # }
    """

    def __init__(self, config_path: str = "config/config.yaml",
                 params_path: str = "config/params.yaml"):
        self.cfg    = read_yaml(config_path)
        self.params = read_yaml(params_path)

        preprocessor_dir = self.cfg["paths"]["preprocessors"]
        models_dir       = self.cfg["paths"]["models"]

        self.scaler  = load_object(
            os.path.join(preprocessor_dir, "scaler.pkl")
        )
        self.imputer = load_object(
            os.path.join(preprocessor_dir, "imputer.pkl")
        )
        self.xgb_model = load_object(
            os.path.join(models_dir, self.params["xgboost"]["model_filename"])
        )
        self.rf_model = load_object(
            os.path.join(models_dir, self.params["random_forest"]["model_filename"])
        )
        logger.info("PredictionPipeline initialised — scaler + 2 models loaded.")

    # ── public ────────────────────────────────────────────────────────────────

    def predict(self, input_data: dict) -> dict:
        """
        input_data: dict mapping feature name → value
        Returns combined prediction dict for both models.
        """
        X = self._prepare(input_data)
        results = {}
        for name, model in [("xgboost", self.xgb_model),
                             ("random_forest", self.rf_model)]:
            results[name] = self._infer(model, X, name)
        return results

    def predict_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Batch prediction on a DataFrame. Returns df with added columns."""
        records = []
        for _, row in df.iterrows():
            res = self.predict(row.to_dict())
            records.append({
                "xgb_label":  res["xgboost"]["label"],
                "xgb_conf":   res["xgboost"]["confidence"],
                "rf_label":   res["random_forest"]["label"],
                "rf_conf":    res["random_forest"]["confidence"],
            })
        return pd.concat([df.reset_index(drop=True),
                          pd.DataFrame(records)], axis=1)

    # ── private ───────────────────────────────────────────────────────────────

    def _prepare(self, data: dict) -> np.ndarray:
        """Convert input dict → imputed → scaled numpy array."""
        row = [float(data.get(col, 0)) for col in FEATURE_COLUMNS]
        arr = np.array(row).reshape(1, -1)
        arr = self.imputer.transform(arr)
        return self.scaler.transform(arr)

    def _infer(self, model, X: np.ndarray, name: str) -> dict:
        pred_int = int(model.predict(X)[0])
        label    = decode_labels([pred_int])[0]
        try:
            proba = model.predict_proba(X)[0]
            probs = {cls: round(float(p), 4)
                     for cls, p in zip(CLASS_NAMES, proba)}
            confidence = round(float(proba[pred_int]), 4)
        except Exception:
            probs      = {cls: None for cls in CLASS_NAMES}
            confidence = None

        return {
            "label":         label,
            "confidence":    confidence,
            "probabilities": probs,
        }
