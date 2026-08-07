# ============================================================
# MODULE 7: Machine Learning — Symptom Classifier
# Covers: Week 13 (Machine Learning in AI)
# ============================================================

from typing import Dict, List, Any, Union
import numpy as np
from sklearn.ensemble import RandomForestClassifier

class MLDiagnosticClassifier:
    """
    ML-based medical diagnostic classifier using Random Forest.
    Inputs: Binary symptom vector
    Output: Top predicted diagnoses with confidence scores
    """

    SYMPTOMS = [
        'fever', 'cough', 'fatigue', 'shortness_of_breath', 'sore_throat',
        'headache', 'body_aches', 'loss_of_taste', 'loss_of_smell', 'runny_nose',
        'sneezing', 'chest_pain', 'nausea', 'diarrhea', 'rash',
        'joint_pain', 'chills', 'sweating'
    ]

    DISEASES = [
        'covid19', 'common_cold', 'flu', 'tuberculosis',
        'cardiac_event', 'gastroenteritis', 'pneumonia'
    ]

    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
        self._train_default_model()

    def _symptoms_to_vector(self, symptoms: List[str]) -> List[int]:
        symptom_set = {s.lower().strip() for s in symptoms}
        return [1 if sym in symptom_set else 0 for sym in self.SYMPTOMS]

    def _train_default_model(self):
        # Synthetic dataset generation based on clinical patterns
        np.random.seed(42)
        X, y = [], []
        
        disease_profiles = {
            'covid19': ['fever', 'cough', 'fatigue', 'loss_of_taste', 'loss_of_smell', 'shortness_of_breath'],
            'common_cold': ['runny_nose', 'sneezing', 'sore_throat', 'cough'],
            'flu': ['fever', 'body_aches', 'chills', 'fatigue', 'headache', 'cough'],
            'tuberculosis': ['cough', 'fever', 'sweating', 'fatigue', 'shortness_of_breath'],
            'cardiac_event': ['chest_pain', 'shortness_of_breath', 'sweating', 'nausea'],
            'gastroenteritis': ['nausea', 'diarrhea', 'body_aches', 'fever'],
            'pneumonia': ['cough', 'fever', 'shortness_of_breath', 'chest_pain', 'chills']
        }

<<<<<<< Updated upstream
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
        """Train all models and select the best one"""
        df = self._generate_synthetic_data(2000)
        X = df[self.SYMPTOM_FEATURES].values
        y = self.label_encoder.fit_transform(df['disease'])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        results = {}
        best_score = -np.inf

        if verbose:
            print("=" * 55)
            print("  ML Diagnostic Classifier — Training")
            print("=" * 55)

        for name, model in self.models.items():
            model.fit(X_train, y_train)

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

            if cv_scores.mean() > best_score:
                best_score = cv_scores.mean()
                self.best_model = model
                self.best_model_name = name
=======
        for disease, profile in disease_profiles.items():
            for _ in range(50):
                vec = self._symptoms_to_vector(profile)
                # Introduce slight noise / variation
                noise = np.random.choice([0, 1], size=len(self.SYMPTOMS), p=[0.9, 0.1])
                vec = [min(1, max(0, v + n)) for v, n in zip(vec, noise)]
                X.append(vec)
                y.append(disease)
>>>>>>> Stashed changes

        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, symptoms: List[str]) -> Dict[str, Any]:
        if not self.is_trained:
            self._train_default_model()

<<<<<<< Updated upstream
        features = np.array([
            [1 if s in symptoms else 0
             for s in self.SYMPTOM_FEATURES]
        ])
        pred_encoded = self.best_model.predict(features)[0]
        pred_proba = self.best_model.predict_proba(features)[0]
=======
        vector = self._symptoms_to_vector(symptoms)
        probs = self.model.predict_proba([vector])[0]
        
        classes = self.model.classes_
        results = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
>>>>>>> Stashed changes

        top_dx, top_conf = results[0]

        return {
            'diagnosis': top_dx,
            'confidence': float(top_conf),
            'top5': results[:5],
            'model_used': 'Random Forest',
            'symptom_vector': vector
        }

<<<<<<< Updated upstream
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
    clf.train(verbose=True)
=======
    def analyze(self, percept: Union[Dict, Any]) -> Dict:
        """Module interface for the agent (safely handles dict and PatientPercept objects)"""
        if isinstance(percept, dict):
            symptoms = percept.get('symptoms', [])
        else:
            symptoms = getattr(percept, 'symptoms', [])

        if not isinstance(symptoms, list):
            symptoms = []
>>>>>>> Stashed changes

        result = self.predict(symptoms)
        result['summary'] = (f"{result['model_used']}: "
                             f"{result['diagnosis']} "
                             f"({result['confidence']*100:.1f}% confidence)")
        return result
