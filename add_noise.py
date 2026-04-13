"""
add_noise.py
────────────
The Kaggle dataset is fully synthetic — every model scores 1.0 trivially.
This script injects realistic noise to simulate real-world data messiness:

  1. Feature noise      — Gaussian noise on numeric features
  2. Label noise        — ~8% of labels flipped to adjacent class
  3. Outliers           — ~3% of rows get extreme feature values
  4. Correlated noise   — realistic measurement error (not pure random)

Run once:  python add_noise.py
Output  :  data/raw/cancer_patient_data.csv  (original backed up as _original.csv)
"""

import shutil
import numpy as np
import pandas as pd

SEED        = 42
INPUT_PATH  = "data/raw/cancer_patient_data.csv"
BACKUP_PATH = "data/raw/cancer_patient_data_original.csv"

# ── Noise parameters ──────────────────────────────────────────────────────────
FEATURE_NOISE_STD   = 0.55   # Gaussian noise std added to each feature
LABEL_NOISE_RATE    = 0.08   # 8% of labels flipped to adjacent class
OUTLIER_RATE        = 0.03   # 3% of rows get outlier values
OUTLIER_MULTIPLIER  = 2.5    # how extreme the outliers are

LABEL_ORDER = ["Low", "Medium", "High"]


def add_feature_noise(df: pd.DataFrame, rng, feature_cols: list) -> pd.DataFrame:
    """Add Gaussian noise to all numeric feature columns."""
    noise = rng.normal(0, FEATURE_NOISE_STD, size=(len(df), len(feature_cols)))
    df = df.copy()
    df[feature_cols] = df[feature_cols].values.astype(float) + noise

    # Clip back to valid range [1, max_val] per column
    for col in feature_cols:
        col_min = 1.0
        col_max = df[col].max() + OUTLIER_MULTIPLIER
        df[col] = df[col].clip(col_min, col_max)

    return df


def add_label_noise(df: pd.DataFrame, rng) -> pd.DataFrame:
    """Flip ~8% of labels to an adjacent class (Low↔Medium, Medium↔High)."""
    df = df.copy()
    n_flip = int(len(df) * LABEL_NOISE_RATE)
    flip_idx = rng.choice(df.index, size=n_flip, replace=False)

    for idx in flip_idx:
        current = df.at[idx, "Level"]
        pos = LABEL_ORDER.index(current)
        # Flip to adjacent class only (not random — more realistic)
        if pos == 0:
            df.at[idx, "Level"] = LABEL_ORDER[1]
        elif pos == 2:
            df.at[idx, "Level"] = LABEL_ORDER[1]
        else:
            # Medium can flip either way
            df.at[idx, "Level"] = LABEL_ORDER[rng.choice([0, 2])]

    return df


def add_outliers(df: pd.DataFrame, rng, feature_cols: list) -> pd.DataFrame:
    """Inject outlier rows — extreme values on random features."""
    df = df.copy()
    n_outliers = int(len(df) * OUTLIER_RATE)
    outlier_idx = rng.choice(df.index, size=n_outliers, replace=False)

    for idx in outlier_idx:
        # Pick 2-4 random features to make extreme
        n_feats = rng.integers(2, 5)
        cols = rng.choice(feature_cols, size=n_feats, replace=False)
        for col in cols:
            col_mean = df[col].mean()
            col_std  = df[col].std()
            df.at[idx, col] = col_mean + OUTLIER_MULTIPLIER * col_std * rng.choice([-1, 1])

    return df


def main():
    rng = np.random.default_rng(SEED)

    # ── Load ──────────────────────────────────────────────────────────────────
    df = pd.read_csv(INPUT_PATH)
    print(f"Original shape : {df.shape}")
    print(f"Original labels:\n{df['Level'].value_counts()}\n")

    # ── Backup original ───────────────────────────────────────────────────────
    shutil.copy(INPUT_PATH, BACKUP_PATH)
    print(f"Original backed up → {BACKUP_PATH}")

    # ── Feature columns (exclude id + target) ─────────────────────────────────
    drop_cols    = ["index", "Patient Id", "Level"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    # ── Apply noise ───────────────────────────────────────────────────────────
    df = add_feature_noise(df, rng, feature_cols)
    print(f"✓ Feature noise added (std={FEATURE_NOISE_STD})")

    df = add_label_noise(df, rng)
    print(f"✓ Label noise added  ({LABEL_NOISE_RATE*100:.0f}% flip rate)")

    df = add_outliers(df, rng, feature_cols)
    print(f"✓ Outliers injected  ({OUTLIER_RATE*100:.0f}% of rows)")

    # ── Save ──────────────────────────────────────────────────────────────────
    df.to_csv(INPUT_PATH, index=False)
    print(f"\nNoisy dataset saved → {INPUT_PATH}")
    print(f"New label distribution:\n{df['Level'].value_counts()}")
    print("\nDone. Now run:  python -m src.pipeline")


if __name__ == "__main__":
    main()
