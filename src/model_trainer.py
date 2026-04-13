"""
model_trainer.py
────────────────
Trains XGBoost and Random Forest classifiers using
RandomizedSearchCV for efficient hyperparameter tuning.
Both models are saved to artifacts/models/.
"""
import os
import sys
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

from src.logger import get_logger
from src.exception import CancerRiskException
from src.utils import read_yaml, save_object

logger = get_logger(__name__)


class ModelTrainer:
    def __init__(
        self,
        config_path: str = "config/config.yaml",
        params_path: str = "config/params.yaml",
    ):
        self.cfg    = read_yaml(config_path)
        self.params = read_yaml(params_path)

        self.models_dir   = self.cfg["paths"]["models"]
        self.random_state = self.cfg["project"]["random_state"]
        self.cv_folds     = self.params["training"]["cv_folds"]
        self.scoring      = self.params["training"]["scoring"]
        self.n_jobs       = self.params["training"]["n_jobs"]

        os.makedirs(self.models_dir, exist_ok=True)

    # ── public ────────────────────────────────────────────────────────────────

    def train_both(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> dict:
        """
        Train XGBoost and Random Forest with RandomizedSearchCV.
        Returns dict with both fitted best estimators and their CV scores.
        """
        results = {}

        logger.info("=" * 60)
        logger.info("Training XGBoost …")
        xgb_result = self._train_xgboost(X_train, y_train)
        results["xgboost"] = xgb_result

        logger.info("=" * 60)
        logger.info("Training Random Forest …")
        rf_result = self._train_random_forest(X_train, y_train)
        results["random_forest"] = rf_result

        logger.info("=" * 60)
        logger.info("✅ Both models trained and saved.")
        return results

    # ── private ───────────────────────────────────────────────────────────────

    def _train_xgboost(
        self, X: np.ndarray, y: np.ndarray
    ) -> dict:
        num_classes = len(np.unique(y))
        base = XGBClassifier(
            objective="multi:softprob",
            num_class=num_classes,
            eval_metric="mlogloss",
            use_label_encoder=False,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        param_grid = self.params["xgboost"]["param_grid"]
        return self._search_and_save(
            base,
            param_grid,
            X,
            y,
            model_name="xgboost",
            filename=self.params["xgboost"]["model_filename"],
            n_iter=30,
        )

    def _train_random_forest(
        self, X: np.ndarray, y: np.ndarray
    ) -> dict:
        base = RandomForestClassifier(
            class_weight="balanced",
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        # Convert null → None for max_depth
        param_grid = self.params["random_forest"]["param_grid"].copy()
        param_grid["max_depth"] = [
            None if v is None else v
            for v in param_grid["max_depth"]
        ]
        return self._search_and_save(
            base,
            param_grid,
            X,
            y,
            model_name="random_forest",
            filename=self.params["random_forest"]["model_filename"],
            n_iter=30,
        )

    def _search_and_save(
        self,
        estimator,
        param_grid: dict,
        X: np.ndarray,
        y: np.ndarray,
        model_name: str,
        filename: str,
        n_iter: int = 30,
    ) -> dict:
        cv = StratifiedKFold(
            n_splits=self.cv_folds, shuffle=True,
            random_state=self.random_state
        )
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring=self.scoring,
            cv=cv,
            refit=True,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=1,
        )
        t0 = time.time()
        search.fit(X, y)
        elapsed = time.time() - t0

        best_score = search.best_score_
        best_params = search.best_params_
        best_model = search.best_estimator_

        logger.info(f"[{model_name}] Best CV {self.scoring}: {best_score:.4f}")
        logger.info(f"[{model_name}] Best params: {best_params}")
        logger.info(f"[{model_name}] Training time: {elapsed:.1f}s")

        save_path = os.path.join(self.models_dir, filename)
        save_object(save_path, best_model)

        return {
            "model":       best_model,
            "best_score":  best_score,
            "best_params": best_params,
            "cv_results":  search.cv_results_,
            "model_name":  model_name,
            "saved_path":  save_path,
        }


# ── CLI quick-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pandas as pd
    from src.preprocessing import Preprocessor

    train_df = pd.read_csv("data/processed/train.csv")
    prep = Preprocessor()
    X_train, y_train = prep.fit_transform_train(train_df)

    trainer = ModelTrainer()
    results = trainer.train_both(X_train, y_train)

    for name, r in results.items():
        print(f"\n{name} → Best CV score: {r['best_score']:.4f}")
