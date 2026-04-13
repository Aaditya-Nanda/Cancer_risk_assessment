from setuptools import setup, find_packages

setup(
    name="cancer_risk_assessment",
    version="1.0.0",
    author="Your Name",
    description="Cancer Risk Assessment — ML Pipeline",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "pandas>=2.0",
        "numpy>=1.24",
        "scikit-learn>=1.4",
        "xgboost>=2.0",
        "imbalanced-learn>=0.12",
        "matplotlib>=3.7",
        "seaborn>=0.13",
        "streamlit>=1.35",
        "pyyaml>=6.0",
    ],
)
