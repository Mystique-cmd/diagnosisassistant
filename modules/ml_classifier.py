from typing import Dict, List
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
import joblib  # Used to save and load trained models
import matplotlib

matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')


class MLDiagnosticClassifier:
    """
    Ensemble ML-based diagnostic classifier.
    Uses Decision Trees, Random Forest, and
    Gradient Boosting for robust diagnosis.
    """

    SYMPTOM_FEATURES = [
        'fever', 'cough', 'fatigue', 'difficulty_breathing',
        'muscle_ache', 'headache', 'sore_throat', 'nausea',
        'chills', 'vomiting', 'diarrhea', 'rash', 'loss_of_smell',
        'chest_pain', 'joint_pain', 'sweating', 'abdominal_pain', 'runny_nose'
    ]

    DISEASE_LABELS = [
        'flu', 'covid19', 'dengue', 'cardiac_event',
        'diabetes', 'common_cold', 'tuberculosis', 'meningitis'
    ]

    def __init__(self):
        self.models = {
            'Decision Tree':     DecisionTreeClassifier(
                max_depth=8, criterion='entropy', random_state=42),
            'Random Forest':     RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, random_state=42),
        }
        self.best_model      = None
        self.best_model_name = None
        self.label_encoder   = LabelEncoder()
        self.is_trained      = False

    def load_trained_models(self):
        """Loads models from disk if they exist."""
        if os.path.exists('data/trained_models.pkl'):
            models_data = joblib.load('data/trained_models.pkl')
            self.models = models_data.get('models', self.models)
            # Restore the label encoder's fitted state
            if 'label_encoder_classes' in models_data:
                self.label_encoder.classes_ = models_data['label_encoder_classes']
            # Set Random Forest as the best model (most robust)
            self.best_model = self.models.get('Random Forest')
            self.best_model_name = 'Random Forest'
            self.is_trained = True
            print("Models loaded from disk successfully.")
            return True
        else:
            print("No saved models found at data/trained_models.pkl")
            return False

    def save_trained_models(self):
        """Save the trained models to disk for faster loading."""
        os.makedirs('data', exist_ok=True)
        models_data = {
            'models': self.models,
            'label_encoder_classes': self.label_encoder.classes_
        }
        joblib.dump(models_data, 'data/trained_models.pkl')
        print("Models saved to data/trained_models.pkl")

    def train_models_from_csv(self, csv_path="data/final_training_data.csv", verbose: bool = True) -> Dict:
        """Train all models from actual CSV dataset instead of synthetic data."""
        print(f"Loading training data from {csv_path}...")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Could not find {csv_path}. Did you run the data generator?")
            
        df = pd.read_csv(csv_path)
        
        # X is the input (the 18 symptoms)
        # y is the output (the disease label)
        X = df[self.SYMPTOM_FEATURES]
        y = df['disease']
        
        # Encode the disease labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split the data: 80% for training, 20% for testing
        print("Splitting data into 80% training and 20% testing...")
        X_train, X_test, y_train, y_test = train_test_split(
            X.values, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
        
        # Train each model
        print("Training models (this may take a moment)...")
        results = {}
        best_score = -np.inf
        
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            
            # Test accuracy on the 20% holdout set
            predictions = model.predict(X_test)
            acc = accuracy_score(y_test, predictions)
            
            # 5-fold cross validation on training set
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            
            results[name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_acc': acc
            }
            
            if verbose:
                print(f"  - {name}")
                print(f"     CV Accuracy  : {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
                print(f"     Test Accuracy: {acc:.2%}")
            
            # Select best model based on CV accuracy
            if cv_scores.mean() > best_score:
                best_score = cv_scores.mean()
                self.best_model = model
                self.best_model_name = name
            
        self.is_trained = True
        self._X_test = X_test
        self._y_test = y_test
        
        if verbose:
            print(f"\n  Best Model: {self.best_model_name} (CV accuracy {best_score:.4f})")
        
        # Save the trained models so you don't have to re-train every time you run app.py
        self.save_trained_models()
        return results
   
    def _generate_synthetic_data(self, n_samples: int = 2000) -> pd.DataFrame:
        """Generate realistic synthetic medical dataset"""
        np.random.seed(42)
        records = []

        # Disease profiles: P(symptom | disease)
        profiles = {
            'flu':           {'fever': 0.90, 'cough': 0.85, 'fatigue': 0.88,
                              'headache': 0.70, 'body_aches': 0.80, 'loss_of_smell': 0.20},
            'covid19':       {'fever': 0.88, 'cough': 0.80, 'fatigue': 0.90,
                              'loss_of_smell': 0.85, 'headache': 0.65, 'body_aches': 0.60},
            'dengue':        {'fever': 0.98, 'rash': 0.75, 'joint_pain': 0.85,
                              'headache': 0.90, 'fatigue': 0.80, 'body_aches': 0.88},
            'cardiac_event': {'chest_pain': 0.92, 'shortness_of_breath': 0.88,
                              'fatigue': 0.70, 'sweating': 0.75, 'headache': 0.30},
            'diabetes':      {'fatigue': 0.82, 'frequent_urination': 0.95,
                              'excessive_thirst': 0.92, 'blurred_vision': 0.70,
                              'weight_loss': 0.50},
            'common_cold':   {'cough': 0.90, 'fever': 0.50, 'headache': 0.60,
                              'fatigue': 0.55, 'body_aches': 0.50},
            'tuberculosis':  {'cough': 0.95, 'weight_loss': 0.85, 'night_sweats': 0.80,
                              'fatigue': 0.88, 'fever': 0.70},
            'meningitis':    {'headache': 0.95, 'stiff_neck': 0.90, 'fever': 0.92,
                              'light_sensitivity': 0.85, 'fatigue': 0.80},
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

    
    def train(self, csv_path="data/final_training_data.csv", use_synthetic=False, verbose: bool = True) -> Dict:
        """
        Train all models. Attempts to load from CSV file first, falls back to synthetic data if needed.
        
        Args:
            csv_path: Path to the training CSV file
            use_synthetic: If True, force use of synthetic data generation
            verbose: Print training progress
        """
        # Try to load pre-trained models from disk first
        if self.load_trained_models():
            if verbose:
                print(f"Using pre-trained models from disk. Best model: {self.best_model_name}")
            return {}
        
        # Try to train from CSV file if it exists
        if os.path.exists(csv_path) and not use_synthetic:
            try:
                return self.train_models_from_csv(csv_path, verbose)
            except Exception as e:
                print(f"Warning: Could not train from CSV ({e}). Falling back to synthetic data.")
        
        # Fall back to synthetic data generation
        df = self._generate_synthetic_data(2000)
        X = df[self.SYMPTOM_FEATURES].values
        y = self.label_encoder.fit_transform(df['disease'])

        # Test data is held out here and never touches training or
        # model-selection below — only used for the final scoreboard
        # and for the confusion-matrix / feature-importance plots.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        results = {}
        best_score = -np.inf

        if verbose:
            print("=" * 55)
            print("  ML Diagnostic Classifier — Training (Synthetic Data)")
            print("=" * 55)

        for name, model in self.models.items():
            model.fit(X_train, y_train)

            # 5-fold cross validation, computed on the training split
            # only, to check the model generalizes rather than just
            # memorizing the training data.
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            test_acc = model.score(X_test, y_test)

            results[name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_acc': test_acc
            }
            if verbose:
                print(f"\n  {name}")
                print(f"     CV Accuracy  : {cv_scores.mean():.4f} "
                      f"(+/- {cv_scores.std():.4f})")
                print(f"     Test Accuracy: {test_acc:.4f}")

            # NOTE: model selection uses 5-fold CV accuracy (the metric
            # the spec calls out for choosing the best model, since it's
            # a better estimate of generalization than a single test
            # split). We still report test_acc above for visibility.
            if cv_scores.mean() > best_score:
                best_score = cv_scores.mean()
                self.best_model = model
                self.best_model_name = name

        self.is_trained = True
        self._X_test = X_test
        self._y_test = y_test

        if verbose:
            print(f"\n  Best Model: {self.best_model_name} "
                  f"(CV accuracy {best_score:.4f})")
        return results

    
    def predict(self, symptoms: List[str]) -> Dict:
        """Predict disease from symptom list"""
        if not self.is_trained:
            if not self.load_trained_models():
                self.train(verbose=False)

        # Symptom strings are converted into the fixed 18-dim binary
        # vector before ever touching the model.
        features = np.array([
            [1 if s in symptoms else 0
             for s in self.SYMPTOM_FEATURES]
        ])
        pred_encoded = self.best_model.predict(features)[0]
        pred_proba = self.best_model.predict_proba(features)[0]

        disease = self.label_encoder.inverse_transform([pred_encoded])[0]
        classes = self.label_encoder.inverse_transform(range(len(pred_proba)))
        prob_map = dict(zip(classes, pred_proba))
        top5 = sorted(prob_map.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            'diagnosis': disease,
            'confidence': round(float(pred_proba[pred_encoded]), 4),
            'top5': top5,
            'all_probs': prob_map,
            'model_used': self.best_model_name,
            'symptom_vector': features[0].tolist()
        }

    def analyze(self, percept) -> Dict:
        """Module interface for the agent"""
        result = self.predict(percept.symptoms)
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
                         f"{self.best_model_name} has no\nfeature_importances_ attribute",
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
    
    # Try to train from CSV file, with fallback to synthetic data
    print("Training from CSV file (data/final_training_data.csv)...\n")
    try:
        clf.train_models_from_csv("data/final_training_data.csv", verbose=True)
    except Exception as e:
        print(f"CSV training failed ({e}). Using synthetic data instead.\n")
        clf.train(use_synthetic=True, verbose=True)

    result = clf.predict(['fever', 'cough', 'fatigue', 'loss_of_smell'])
    print(f"\nDiagnosis : {result['diagnosis']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Model Used: {result['model_used']}")
    print(f"Top 5     : {result['top5']}")

    path = clf.plot_evaluation()
    print(f"\nEvaluation plots saved to: {path}")