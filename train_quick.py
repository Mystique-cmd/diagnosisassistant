#!/usr/bin/env python
"""Quick training script for ML classifier"""
from modules.ml_classifier import MLDiagnosticClassifier

clf = MLDiagnosticClassifier()
try:
    clf.train_models_from_csv("data/final_training_data.csv", verbose=True)
    print("\n✅ Training complete and models saved!")
except Exception as e:
    print(f"Training failed: {e}")
    print("App will use synthetic data instead.")
