"""
preprocessing.py
────────────────
Handles:
  • Feature / target separation
  • Label encoding of target
  • Feature scaling (StandardScaler / MinMaxScaler / RobustScaler)
  • Class-imbalance correction via SMOTE or SMOTEENN
  • Saves fitted preprocessor to artifacts/preprocessors/
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN

from src.logger import get_logger
from src.exception import CancerRiskException
from src.utils import read_yaml, save_object, encode_labels

logger = get_logger(__name__)

SCALERS = {
    "standard": StandardScaler,
    "minmax":   MinMaxScaler,
    "robust":   RobustScaler,
}


class Preprocessor:
    """
    Fit on training data, transform train/val/test.
    Persists the fitted scaler for inference time.
    """

    def __init__(
        self,
        config_path: str = "config/config.yaml",
        params_path: str = "config/params.yaml",
    ):
        self.cfg    = read_yaml(config_path)
        self.params = read_yaml(params_path)

        self.target        = self.cfg["project"]["target_column"]
        self.scaler_name   = self.cfg["preprocessing"]["scaler"]
        self.strategy      = self.cfg["imbalance"]["strategy"]
        self.k_neighbors   = self.cfg["imbalance"]["smote_k_neighbors"]
        self.random_state  = self.cfg["project"]["random_state"]
        self.preprocessor_dir = self.cfg["paths"]["preprocessors"]

        self.scaler  = SCALERS.get(self.scaler_name, StandardScaler)()
        self.imputer = SimpleImputer(strategy="median")
        self.le      = LabelEncoder()

    # ── public ────────────────────────────────────────────────────────────────

    def fit_transform_train(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Fit scaler on training features, encode target,
        apply imbalance strategy.  Returns (X_resampled, y_resampled).
        """
        X, y = self._split_xy(df)
        logger.info(f"Class distribution before resampling:\n{pd.Series(y).value_counts()}")

        X_imputed = self.imputer.fit_transform(X)
        X_scaled  = self.scaler.fit_transform(X_imputed)

        X_res, y_res = self._resample(X_scaled, y)
        logger.info(f"Class distribution after resampling:\n{pd.Series(y_res).value_counts()}")

        self._persist()
        return X_res, y_res

    def transform(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Transform val / test data with the already-fitted scaler."""
        X, y = self._split_xy(df)
        X_imputed = self.imputer.transform(X)
        X_scaled  = self.scaler.transform(X_imputed)
        return X_scaled, y

    def get_feature_names(self, df: pd.DataFrame) -> list[str]:
        return [c for c in df.columns if c != self.target]

    # ── private ───────────────────────────────────────────────────────────────

    def _split_xy(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        y_raw = df[self.target].values
        y = encode_labels(pd.Series(y_raw)).values
        X = df.drop(columns=[self.target]).values.astype(float)
        return X, y

    def _resample(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        strategy = self.strategy.lower()
        if strategy == "smote":
            sampler = SMOTE(
                k_neighbors=self.k_neighbors,
                random_state=self.random_state,
            )
            logger.info("Applying SMOTE …")
        elif strategy == "smoteenn":
            sampler = SMOTEENN(
                smote=SMOTE(
                    k_neighbors=self.k_neighbors,
                    random_state=self.random_state,
                ),
                random_state=self.random_state,
            )
            logger.info("Applying SMOTEENN …")
        else:
            logger.info("No resampling applied (strategy=none).")
            return X, y

        X_res, y_res = sampler.fit_resample(X, y)
        logger.info(f"Resampled shape: {X_res.shape}")
        return X_res, y_res

    def _persist(self) -> None:
        os.makedirs(self.preprocessor_dir, exist_ok=True)
        save_object(
            os.path.join(self.preprocessor_dir, "scaler.pkl"),
            self.scaler,
        )
        save_object(
            os.path.join(self.preprocessor_dir, "imputer.pkl"),
            self.imputer,
        )
        logger.info("Fitted scaler + imputer saved ✓")


# ── CLI quick-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    train_df = pd.read_csv("data/processed/train.csv")
    val_df   = pd.read_csv("data/processed/val.csv")
    test_df  = pd.read_csv("data/processed/test.csv")

    prep = Preprocessor()
    X_train, y_train = prep.fit_transform_train(train_df)
    X_val,   y_val   = prep.transform(val_df)
    X_test,  y_test  = prep.transform(test_df)

    logger.info(f"Train X: {X_train.shape}, y: {y_train.shape}")
    logger.info(f"Val   X: {X_val.shape},   y: {y_val.shape}")
    logger.info(f"Test  X: {X_test.shape},  y: {y_test.shape}")
