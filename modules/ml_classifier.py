# ============================================================
# MODULE 7: Machine Learning — Symptom Classifier
# Covers: Week 13 (Machine Learning in AI)
# ============================================================

from typing import Dict, List, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix


class MLDiagnosticClassifier:
    """
    ML-based medical diagnostic classifier.

    Trains a small ensemble of classifiers, selects the best model via
    stratified cross-validation, and predicts the top diagnoses with
    confidence scores from a binary symptom vector.
    """

    SYMPTOM_FEATURES = [
        'fever', 'cough', 'fatigue', 'shortness_of_breath', 'sore_throat',
        'headache', 'body_aches', 'loss_of_taste', 'loss_of_smell', 'runny_nose',
        'sneezing', 'chest_pain', 'nausea', 'diarrhea', 'rash',
        'joint_pain', 'chills', 'sweating'
    ]

    # Backwards-compatible alias
    SYMPTOMS = SYMPTOM_FEATURES

    DISEASES = [
        'covid19', 'common_cold', 'flu', 'tuberculosis',
        'cardiac_event', 'gastroenteritis', 'pneumonia'
    ]

    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'Extra Trees': ExtraTreesClassifier(
                n_estimators=100, random_state=42),
        }
        self.best_model = None
        self.best_model_name = None
        self.is_trained = False
        self._X_test = None
        self._y_test = None

    def _generate_synthetic_data(self, n_samples: int = 2000) -> pd.DataFrame:
        """
        Generate a synthetic labelled dataset from clinical symptom profiles.

        Each disease profile maps symptom -> P(symptom | disease). A small
        amount of noise is added so the classifier learns robust patterns.
        """
        np.random.seed(42)
        records = []

        profiles = {
            'covid19': {
                'fever': 0.88, 'cough': 0.80, 'fatigue': 0.90,
                'loss_of_taste': 0.80, 'loss_of_smell': 0.85,
                'shortness_of_breath': 0.65,
            },
            'common_cold': {
                'runny_nose': 0.90, 'sneezing': 0.80,
                'sore_throat': 0.75, 'cough': 0.80,
            },
            'flu': {
                'fever': 0.90, 'body_aches': 0.80, 'chills': 0.75,
                'fatigue': 0.85, 'headache': 0.70, 'cough': 0.80,
            },
            'tuberculosis': {
                'cough': 0.95, 'fever': 0.70, 'sweating': 0.80,
                'fatigue': 0.85, 'shortness_of_breath': 0.60,
            },
            'cardiac_event': {
                'chest_pain': 0.92, 'shortness_of_breath': 0.85,
                'sweating': 0.75, 'nausea': 0.50,
            },
            'gastroenteritis': {
                'nausea': 0.90, 'diarrhea': 0.90,
                'body_aches': 0.50, 'fever': 0.60,
            },
            'pneumonia': {
                'cough': 0.90, 'fever': 0.90, 'shortness_of_breath': 0.85,
                'chest_pain': 0.70, 'chills': 0.70,
            },
        }

        n_per_class = n_samples // len(profiles)
        for disease, symptom_probs in profiles.items():
            for _ in range(n_per_class):
                record = {f: 0 for f in self.SYMPTOM_FEATURES}
                for symptom, prob in symptom_probs.items():
                    if symptom in record:
                        record[symptom] = int(np.random.random() < prob)
                # Add some noise
                for feat in self.SYMPTOM_FEATURES:
                    if record[feat] == 0 and np.random.random() < 0.05:
                        record[feat] = 1
                record['disease'] = disease
                records.append(record)

        df = pd.DataFrame(records).sample(frac=1, random_state=42)
        return df

    def train(self, verbose: bool = True) -> Dict:
        """Train all candidate models and select the best one via CV."""
        df = self._generate_synthetic_data(2000)
        X = df[self.SYMPTOM_FEATURES].values
        y = self.label_encoder.fit_transform(df['disease'])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        self._X_test = X_test
        self._y_test = y_test

        results = {}
        best_score = -np.inf

        if verbose:
            print("=" * 55)
            print("  ML Diagnostic Classifier — Training")
            print("=" * 55)

        for name, model in self.models.items():
            model.fit(X_train, y_train)

            cv_scores = cross_val_score(
                model, X_train, y_train, cv=5, scoring='accuracy')
            test_acc = model.score(X_test, y_test)

            results[name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_acc': test_acc,
            }
            if verbose:
                print(f"\n  {name}")
                print(f"     CV Accuracy  : {cv_scores.mean():.4f} "
                      f"(+/- {cv_scores.std():.4f})")
                print(f"     Test Accuracy: {test_acc:.4f}")

            if cv_scores.mean() > best_score:
                best_score = cv_scores.mean()
                self.best_model = model
                self.best_model_name = name

        self.is_trained = True
        if verbose:
            print(f"\n  Best model: {self.best_model_name} "
                  f"(CV accuracy {best_score:.4f})")
            print("=" * 55)

        return results

    def predict(self, symptoms: List[str]) -> Dict[str, Any]:
        if not self.is_trained:
            self.train(verbose=False)

        features = np.array([
            [1 if s in symptoms else 0
             for s in self.SYMPTOM_FEATURES]
        ])
        pred_encoded = self.best_model.predict(features)[0]
        pred_proba = self.best_model.predict_proba(features)[0]

        classes = self.label_encoder.classes_
        results = sorted(zip(classes, pred_proba),
                         key=lambda x: x[1], reverse=True)

        top_dx, top_conf = results[0]

        return {
            'diagnosis': top_dx,
            'confidence': float(top_conf),
            'top5': results[:5],
            'all_probs': dict(zip(classes, pred_proba.tolist())),
            'model_used': self.best_model_name,
            'symptom_vector': features[0].tolist(),
        }

    def analyze(self, percept) -> Dict:
        """Module interface for the agent (safely handles dict and PatientPercept objects)"""
        if isinstance(percept, dict):
            symptoms = percept.get('symptoms', [])
        else:
            symptoms = getattr(percept, 'symptoms', [])

        if not isinstance(symptoms, list):
            symptoms = []

        result = self.predict(symptoms)
        result['summary'] = (f"{result['model_used']}: "
                             f"{result['diagnosis']} "
                             f"({result['confidence']:.2%})")
        return result

    def plot_evaluation(self, save_path: str = "ml_evaluation.png"):
        """Visualize model performance: confusion matrix + feature importances"""
        if not self.is_trained:
            self.train(verbose=False)

        y_pred = self.best_model.predict(self._X_test)
        cm = confusion_matrix(self._y_test, y_pred)
        labels = self.label_encoder.classes_

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Confusion Matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels, ax=axes[0])
        axes[0].set_title(f"Confusion Matrix\n({self.best_model_name})",
                          fontweight='bold')
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("True")
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Feature Importance
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            sorted_idx = np.argsort(importances)[::-1][:12]
            top_features = [self.SYMPTOM_FEATURES[i] for i in sorted_idx]
            top_values = importances[sorted_idx]
            colors = plt.cm.RdYlGn(top_values / top_values.max())
            axes[1].barh(range(len(top_features)), top_values[::-1],
                         color=colors[::-1])
            axes[1].set_yticks(range(len(top_features)))
            axes[1].set_yticklabels(top_features[::-1])
            axes[1].set_title("Feature Importances (Top 12)",
                              fontweight='bold')
            axes[1].set_xlabel("Importance Score")
        else:
            axes[1].text(0.5, 0.5,
                         f"{self.best_model_name} has no\n"
                         f"feature_importances_ attribute",
                         ha='center', va='center')
            axes[1].axis('off')

        plt.suptitle(f"ML Diagnostic Model Evaluation — {self.best_model_name}",
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {save_path}")
        return save_path


if __name__ == "__main__":
    clf = MLDiagnosticClassifier()
    clf.train(verbose=True)

