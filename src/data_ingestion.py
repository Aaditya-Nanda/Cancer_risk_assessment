"""
data_ingestion.py
─────────────────
Loads raw CSV, validates schema, performs stratified
train / val / test splits, and saves to data/processed/.
"""
import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from src.logger import get_logger
from src.exception import CancerRiskException
from src.utils import read_yaml, ensure_dir

logger = get_logger(__name__)

# ── Expected feature columns (after dropping id cols) ─────────────────────────
EXPECTED_COLUMNS = [
    "Age", "Gender", "Air Pollution", "Alcohol use", "Dust Allergy",
    "OccuPational Hazards", "Genetic Risk", "chronic Lung Disease",
    "Balanced Diet", "Obesity", "Smoking", "Passive Smoker",
    "Chest Pain", "Coughing of Blood", "Fatigue", "Weight Loss",
    "Shortness of Breath", "Wheezing", "Swallowing Difficulty",
    "Clubbing of Finger Nails", "Frequent Cold", "Dry Cough",
    "Snoring", "Level",
]


class DataIngestion:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.cfg = read_yaml(config_path)
        self.raw_path    = self.cfg["paths"]["raw_data"]
        self.out_dir     = self.cfg["paths"]["processed_data"]
        self.target      = self.cfg["project"]["target_column"]
        self.test_size   = self.cfg["data"]["test_size"]
        self.val_size    = self.cfg["data"]["val_size"]
        self.random_state = self.cfg["project"]["random_state"]
        self.drop_cols   = self.cfg["preprocessing"]["drop_columns"]

    # ── public ────────────────────────────────────────────────────────────────

    def run(self) -> tuple[str, str, str]:
        """
        Full ingestion pipeline.
        Returns (train_path, val_path, test_path).
        """
        df = self._load()
        df = self._clean(df)
        self._validate(df)
        train, val, test = self._split(df)
        paths = self._save(train, val, test)
        logger.info("✅ Data ingestion complete.")
        return paths

    # ── private ───────────────────────────────────────────────────────────────

    def _load(self) -> pd.DataFrame:
        logger.info(f"Loading raw data from: {self.raw_path}")
        if not os.path.exists(self.raw_path):
            raise CancerRiskException(
                f"Raw data file not found: {self.raw_path}", sys
            )
        df = pd.read_csv(self.raw_path)
        logger.info(f"Loaded shape: {df.shape}")
        return df

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        # Drop id / index columns (case-insensitive match)
        existing_drop = [
            c for c in df.columns
            if c.lower() in [d.lower() for d in self.drop_cols]
        ]
        if existing_drop:
            df = df.drop(columns=existing_drop)
            logger.info(f"Dropped columns: {existing_drop}")
        df.columns = df.columns.str.strip()
        df = df.dropna()
        logger.info(f"Shape after cleaning: {df.shape}")
        return df

    def _validate(self, df: pd.DataFrame) -> None:
        missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
        if missing:
            raise CancerRiskException(
                f"Missing expected columns: {missing}", sys
            )
        if df[self.target].nunique() != 3:
            raise CancerRiskException(
                f"Expected 3 target classes, found "
                f"{df[self.target].nunique()}", sys
            )
        logger.info("Schema validation passed ✓")

    def _split(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_val, test = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=df[self.target],
        )
        # val_size is fraction of original → adjust for remaining set
        adjusted_val = self.val_size / (1 - self.test_size)
        train, val = train_test_split(
            train_val,
            test_size=adjusted_val,
            random_state=self.random_state,
            stratify=train_val[self.target],
        )
        logger.info(
            f"Split sizes — train: {len(train)}, "
            f"val: {len(val)}, test: {len(test)}"
        )
        return train, val, test

    def _save(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
    ) -> tuple[str, str, str]:
        ensure_dir(self.out_dir)
        paths = (
            os.path.join(self.out_dir, "train.csv"),
            os.path.join(self.out_dir, "val.csv"),
            os.path.join(self.out_dir, "test.csv"),
        )
        train.to_csv(paths[0], index=False)
        val.to_csv(paths[1],   index=False)
        test.to_csv(paths[2],  index=False)
        logger.info(f"Saved → {paths}")
        return paths


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    ingestion = DataIngestion()
    ingestion.run()
