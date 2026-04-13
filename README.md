# 🩺 Cancer Risk Assessment System

Multi-class cancer risk prediction (Low / Medium / High) using XGBoost and Random Forest with SMOTE/SMOTEENN class-imbalance handling.

---

## Setup

```bash
git clone <your-repo>
cd cancer_risk_assessment

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
pip install -e .                # install src as a package
```

## Dataset

Download from Kaggle:
> https://www.kaggle.com/datasets/avi1023/cancer-patients-and-air-pollution

Save the CSV as:
```
data/raw/cancer_patient_data.csv
```

## Run the training pipeline

```bash
python -m src.pipeline
```

This runs all 4 steps in sequence:
1. Data ingestion + train/val/test split
2. Preprocessing + SMOTEENN resampling
3. XGBoost + Random Forest training with RandomizedSearchCV
4. Evaluation — confusion matrix, ROC-AUC, classification report

Artifacts saved to:
- `artifacts/models/`        ← trained models (.pkl)
- `artifacts/preprocessors/` ← fitted scaler (.pkl)
- `artifacts/reports/`       ← metrics JSON + comparison CSV
- `artifacts/plots/`         ← confusion matrices + ROC curves

## Run the Streamlit app

```bash
streamlit run app.py
```

> **Demo mode**: The app works as a UI skeleton even before training — it shows mock predictions so you can test the interface immediately.

## Project Structure

```
cancer_risk_assessment/
├── config/
│   ├── config.yaml          # paths, split sizes, preprocessing settings
│   └── params.yaml          # model hyperparameter search spaces
├── data/
│   ├── raw/                 # place downloaded CSV here
│   ├── processed/           # train / val / test splits
│   └── features/            # optional feature-engineered outputs
├── notebooks/
│   └── 01_EDA.ipynb         # exploratory data analysis
├── src/
│   ├── logger.py
│   ├── exception.py
│   ├── utils.py
│   ├── data_ingestion.py
│   ├── preprocessing.py
│   ├── model_trainer.py
│   ├── model_evaluation.py
│   ├── prediction_pipeline.py
│   └── pipeline.py          # ← orchestrator
├── artifacts/               # auto-created by pipeline
├── logs/                    # auto-created run logs
├── tests/
├── app.py                   # Streamlit frontend
├── requirements.txt
└── setup.py
```

## Run individual modules

```bash
python -m src.data_ingestion   # Step 1 only
python -m src.preprocessing    # Step 2 only
python -m src.model_trainer    # Step 3 only
```
