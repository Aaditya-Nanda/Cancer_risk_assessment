"""
model_evaluation.py
───────────────────
Full evaluation suite for both models:
  • Confusion matrix (saved as PNG)
  • Classification report (saved as JSON + printed)
  • ROC-AUC curves — one-vs-rest, all classes (saved as PNG)
  • Side-by-side comparison DataFrame returned for Streamlit display
"""
import os
import sys
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

from src.logger import get_logger
from src.exception import CancerRiskException
from src.utils import read_yaml, INT_TO_LABEL

logger = get_logger(__name__)

CLASS_NAMES = ["Low", "Medium", "High"]
COLORS      = ["#2ecc71", "#f39c12", "#e74c3c"]


class ModelEvaluator:
    def __init__(
        self,
        config_path: str = "config/config.yaml",
        params_path: str = "config/params.yaml",
    ):
        self.cfg        = read_yaml(config_path)
        self.params     = read_yaml(params_path)
        self.reports_dir = self.cfg["paths"]["reports"]
        self.plots_dir   = self.cfg["paths"]["plots"]
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.plots_dir,   exist_ok=True)

    # ── public ────────────────────────────────────────────────────────────────

    def evaluate_both(
        self,
        models: dict,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> pd.DataFrame:
        """
        Evaluate both models on the test set.
        Returns a side-by-side comparison DataFrame.
        """
        records = []
        for name, result in models.items():
            model = result["model"]
            metrics = self._evaluate_single(model, X_test, y_test, name)
            records.append(metrics)

        comparison = pd.DataFrame(records).set_index("model")
        comparison.to_csv(
            os.path.join(self.reports_dir, "model_comparison.csv")
        )
        logger.info(f"\n{comparison.to_string()}")
        logger.info("✅ Evaluation complete.")
        return comparison

    def evaluate_single(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str,
    ) -> dict:
        return self._evaluate_single(model, X_test, y_test, model_name)

    # ── private ───────────────────────────────────────────────────────────────

    def _evaluate_single(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str,
    ) -> dict:
        y_pred = model.predict(X_test)
        y_prob = self._get_proba(model, X_test)

        acc  = accuracy_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred, average="weighted")
        try:
            roc = roc_auc_score(
                y_test, y_prob, multi_class="ovr", average="weighted"
            )
        except Exception:
            roc = float("nan")

        # Text report
        report = classification_report(
            y_test, y_pred,
            target_names=CLASS_NAMES,
            output_dict=True,
        )
        self._save_report(report, model_name)
        self._plot_confusion_matrix(y_test, y_pred, model_name)
        if y_prob is not None:
            self._plot_roc_curves(y_test, y_prob, model_name)

        logger.info(
            f"[{model_name}] Accuracy: {acc:.4f} | "
            f"F1(w): {f1:.4f} | ROC-AUC: {roc:.4f}"
        )
        return {
            "model":    model_name,
            "accuracy": round(acc, 4),
            "f1_weighted": round(f1, 4),
            "roc_auc_weighted": round(roc, 4),
        }

    def _get_proba(self, model, X: np.ndarray):
        try:
            return model.predict_proba(X)
        except Exception:
            return None

    def _save_report(self, report: dict, model_name: str) -> None:
        path = os.path.join(
            self.reports_dir, f"{model_name}_classification_report.json"
        )
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Classification report saved → {path}")

    def _plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
    ) -> None:
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            ax=ax,
        )
        ax.set_title(f"Confusion Matrix — {model_name.replace('_', ' ').title()}")
        ax.set_ylabel("True label")
        ax.set_xlabel("Predicted label")
        plt.tight_layout()
        path = os.path.join(self.plots_dir, f"{model_name}_confusion_matrix.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Confusion matrix saved → {path}")

    def _plot_roc_curves(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        model_name: str,
    ) -> None:
        n_classes = y_prob.shape[1]
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))

        fig, ax = plt.subplots(figsize=(7, 5))
        for i, (cls, color) in enumerate(zip(CLASS_NAMES, COLORS)):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2,
                    label=f"{cls} (AUC = {roc_auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curves — {model_name.replace('_', ' ').title()}")
        ax.legend(loc="lower right")
        plt.tight_layout()
        path = os.path.join(self.plots_dir, f"{model_name}_roc_curves.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"ROC curves saved → {path}")
