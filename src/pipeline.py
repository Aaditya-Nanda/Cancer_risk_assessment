"""
pipeline.py
───────────
End-to-end orchestrator.
Run this to execute the full training pipeline in one shot:
  python -m src.pipeline
"""
import sys
import time

import pandas as pd

from src.logger import get_logger
from src.exception import CancerRiskException
from src.data_ingestion  import DataIngestion
from src.preprocessing   import Preprocessor
from src.model_trainer   import ModelTrainer
from src.model_evaluation import ModelEvaluator

logger = get_logger(__name__)


def run_pipeline(
    config_path: str = "config/config.yaml",
    params_path: str = "config/params.yaml",
) -> pd.DataFrame:
    """
    Execute the complete training pipeline.
    Returns the model comparison DataFrame.
    """
    t_start = time.time()
    logger.info("=" * 70)
    logger.info("CANCER RISK ASSESSMENT — TRAINING PIPELINE START")
    logger.info("=" * 70)

    # ── Step 1: Data ingestion ────────────────────────────────────────────────
    logger.info("\n[1/4] Data Ingestion")
    ingestion = DataIngestion(config_path)
    train_path, val_path, test_path = ingestion.run()

    # ── Step 2: Preprocessing ─────────────────────────────────────────────────
    logger.info("\n[2/4] Preprocessing + Imbalance Handling")
    train_df = pd.read_csv(train_path)
    val_df   = pd.read_csv(val_path)
    test_df  = pd.read_csv(test_path)

    prep = Preprocessor(config_path, params_path)
    X_train, y_train = prep.fit_transform_train(train_df)
    X_val,   y_val   = prep.transform(val_df)
    X_test,  y_test  = prep.transform(test_df)

    # ── Step 3: Model training ────────────────────────────────────────────────
    logger.info("\n[3/4] Model Training (XGBoost + Random Forest)")
    trainer = ModelTrainer(config_path, params_path)
    model_results = trainer.train_both(X_train, y_train)

    # ── Step 4: Evaluation ────────────────────────────────────────────────────
    logger.info("\n[4/4] Model Evaluation on Test Set")
    evaluator = ModelEvaluator(config_path, params_path)
    comparison = evaluator.evaluate_both(model_results, X_test, y_test)

    elapsed = time.time() - t_start
    logger.info("=" * 70)
    logger.info(f"PIPELINE COMPLETE  —  Total time: {elapsed:.1f}s")
    logger.info("=" * 70)
    logger.info(f"\nModel Comparison:\n{comparison.to_string()}")

    return comparison


if __name__ == "__main__":
    run_pipeline()
