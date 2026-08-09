# ============================================================
# EVALUATION MODULE: Performance Metrics
# Measures how well the entire AI system performs.
# Compares diagnoses from all modules against ground-truth
# labels and generates comprehensive performance reports.
# ============================================================

import os
import sys
import warnings
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report
)
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

# Import all diagnostic modules
from modules.agent import HealthcareDiagnosticAgent, PatientPercept
from modules.knowledge_base import MedicalKnowledgeBase
from modules.bayesian_net import SimpleBayesianDiagnostics
from modules.ml_classifier import MLDiagnosticClassifier
from modules.fuzzy_controller import FuzzySeverityAssessor

# NeuralNetwork has a pre-existing missing import (Dict from typing)
# in the module itself; we wrap the import so evaluation still works
# when it is unavailable.
try:
    from modules.neural_network import NeuralDiagnosticModel
    _NEURAL_AVAILABLE = True
except (ImportError, NameError, Exception):
    NeuralDiagnosticModel = None
    _NEURAL_AVAILABLE = False
    print("  WARNING: NeuralNetwork module could not be imported; "
          "it will be excluded from evaluation.")


# Standardised label set (must match modules' disease space)
DISEASE_LABELS = [
    'flu', 'covid19', 'dengue', 'cardiac_event',
    'diabetes', 'common_cold', 'tuberculosis', 'meningitis'
]

# Symptom column names expected in the CSV
SYMPTOM_COLUMNS = [
    'fever', 'cough', 'fatigue', 'headache',
    'body_aches', 'loss_of_smell', 'chest_pain',
    'rash', 'joint_pain', 'shortness_of_breath',
    'sweating', 'frequent_urination', 'excessive_thirst',
    'blurred_vision', 'night_sweats', 'weight_loss',
    'stiff_neck', 'light_sensitivity'
]

GROUND_TRUTH_COL = 'disease'


class ModelEvaluator:
    """
    Quality assurance module for the entire diagnostic system.

    Loads a CSV of patient cases with known ground-truth labels,
    runs every registered AI module on each case, and computes
    standard classification metrics for each module.

    Metrics reported
    ----------------
    Accuracy   : Correct / Total
    Precision  : TP / (TP + FP)   (macro-averaged)
    Recall     : TP / (TP + FN)   (macro-averaged)
    F1-Score   : 2 x (P x R) / (P + R)   (macro-averaged)
    Confusion  : Grid of predictions vs. actual labels
    ROC-AUC    : Area under ROC curve (One-vs-Rest, macro)
    """

    def __init__(self, data_path: str = "data/"):
        """
        Parameters
        ----------
        data_path : str
            Directory where CSV evaluation files are stored.
        """
        self.data_path = data_path
        self._label_encoder = LabelEncoder()
        self._label_encoder.fit(DISEASE_LABELS)
        self._modules = {}           # name -> module instance
        self._agent = None           # HealthcareDiagnosticAgent

    # ------------------------------------------------------------------
    #  Module initialisation
    # ------------------------------------------------------------------
    def _build_modules(self) -> HealthcareDiagnosticAgent:
        """Instantiate and register all AI diagnostic modules."""
        agent = HealthcareDiagnosticAgent()

        modules_list = [
            ('KnowledgeBase', MedicalKnowledgeBase()),
            ('BayesianNet', SimpleBayesianDiagnostics()),
            ('MLClassifier', MLDiagnosticClassifier()),
            ('FuzzyController', FuzzySeverityAssessor()),
        ]

        # Only add NeuralNetwork if it was imported successfully
        if _NEURAL_AVAILABLE:
            modules_list.append(('NeuralNetwork', NeuralDiagnosticModel()))

        for name, mod in modules_list:
            try:
                agent.register_module(name, mod)
                self._modules[name] = mod
            except Exception as e:
                print(f"  WARNING: Failed to register {name}: {e}")

        self._agent = agent
        return agent

    # ------------------------------------------------------------------
    #  Data loading
    # ------------------------------------------------------------------
    def load_data(self, filename: str) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Load a CSV file from the data directory.

        The CSV must contain:
        - 18 symptom columns (binary 0/1) matching SYMPTOM_COLUMNS
        - A 'disease' column with ground-truth labels

        Returns
        -------
        X : pd.DataFrame   symptom columns only
        y : np.ndarray     encoded ground-truth labels (int)
        """
        path = os.path.join(self.data_path, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Evaluation data not found: {path}\n"
                f"Please place your CSV file at {path} with columns:\n"
                f"  {SYMPTOM_COLUMNS} + '{GROUND_TRUTH_COL}'"
            )

        df = pd.read_csv(path)

        # Validate required columns
        missing_symptoms = [c for c in SYMPTOM_COLUMNS if c not in df.columns]
        if missing_symptoms:
            raise ValueError(
                f"Missing symptom column(s) in {filename}: {missing_symptoms}"
            )
        if GROUND_TRUTH_COL not in df.columns:
            raise ValueError(
                f"Missing ground-truth column '{GROUND_TRUTH_COL}' in {filename}"
            )

        X = df[SYMPTOM_COLUMNS]
        y_raw = df[GROUND_TRUTH_COL].str.lower().str.replace(' ', '_')

        # Encode labels; warn about unseen classes
        unseen = set(y_raw) - set(self._label_encoder.classes_)
        if unseen:
            print(f"  WARNING: labels not in standard set: {unseen}")
            print(f"     These will be mapped to 'common_cold' as fallback.")

        y = y_raw.map(
            lambda lbl: lbl if lbl in self._label_encoder.classes_
            else 'common_cold'
        )
        y_encoded = self._label_encoder.transform(y)

        print(f"  Loaded {len(df)} cases from {filename}")
        print(f"     Symptom columns : {len(SYMPTOM_COLUMNS)}")
        print(f"     Disease classes : {len(np.unique(y_encoded))}")

        return X, y_encoded

    # ------------------------------------------------------------------
    #  Patient-level prediction helpers
    # ------------------------------------------------------------------
    def _row_to_symptom_list(self, row: pd.Series) -> List[str]:
        """Convert a binary symptom row into a list of present symptom names."""
        return [col for col in SYMPTOM_COLUMNS if row.get(col, 0) == 1]

    def _normalise_diagnosis(self, module_name: str,
                             diagnosis: str) -> str:
        """
        Map a module's raw diagnosis string to the standard DISEASE_LABELS set.

        Handles common variations:
        - 'cardiac' -> 'cardiac_event'
        - 'flu_suspected', 'flu_confirmed' -> 'flu'
        - severity labels from FuzzyController -> 'common_cold' (fallback)
        """
        diag = diagnosis.lower().replace(' ', '_')

        # Direct match
        if diag in DISEASE_LABELS:
            return diag

        # Module-specific normalisation
        # KnowledgeBase / BayesianNet suffixes
        for label in DISEASE_LABELS:
            if label in diag:
                return label

        # FuzzyController severity labels have no disease mapping
        if module_name == 'FuzzyController':
            return 'common_cold'

        return 'common_cold'  # safe fallback

    def _predict_row(self, row: pd.Series,
                     agent: HealthcareDiagnosticAgent) -> Dict[str, str]:
        """
        Run a single patient row through the agent and collect
        each module's predicted diagnosis.
        """
        symptoms = self._row_to_symptom_list(row)

        # Build a minimal PatientPercept (vitals are dummy values;
        # the CSV contains only symptom data).
        percept = PatientPercept(
            patient_id='eval',
            symptoms=symptoms,
            age=40,
            temperature=37.0,
            heart_rate=80,
            blood_pressure='120/80'
        )

        # Full agent cycle
        agent.perceive(percept)
        results = agent.think()
        agent.act(results)

        # Collect per-module diagnoses
        predictions = {}
        for module_name, module_result in results.items():
            if isinstance(module_result, dict) and 'diagnosis' in module_result:
                raw = module_result['diagnosis']
                predictions[module_name] = self._normalise_diagnosis(
                    module_name, raw
                )
            else:
                predictions[module_name] = 'common_cold'

        return predictions

    # ------------------------------------------------------------------
    #  Core evaluation
    # ------------------------------------------------------------------
    def evaluate(self,
                 csv_file: str = "test_data.csv",
                 verbose: bool = True) -> Dict:
        """
        Evaluate all modules against ground-truth labels from a CSV file.

        Parameters
        ----------
        csv_file : str
            Name of the CSV file inside ``self.data_path``.
        verbose  : bool
            If True, print progress during evaluation.

        Returns
        -------
        report : dict
            Nested dictionary with structure::

                {
                    'ModuleName': {
                        'accuracy':         float,
                        'precision':        float,
                        'recall':           float,
                        'f1_score':         float,
                        'confusion_matrix': np.ndarray (n_classes x n_classes),
                        'roc_auc':          float or None,
                        'y_true':           np.ndarray (int encoded),
                        'y_pred':           np.ndarray (int encoded),
                        'y_score':          np.ndarray (proba matrix) or None,
                    },
                    ...
                    'metadata': {
                        'csv_file':     str,
                        'num_cases':    int,
                        'num_classes':  int,
                        'class_labels': list,
                    }
                }
        """
        if not self._modules:
            self._build_modules()

        # 1. Load data
        X, y_true_encoded = self.load_data(csv_file)
        n_cases = X.shape[0]

        # 2. Prepare storage
        module_names = list(self._modules.keys())
        predictions_raw: Dict[str, List[str]] = {
            name: [] for name in module_names
        }

        # 3. Run every patient through the agent
        if verbose:
            print(f"\n  Evaluating {n_cases} patients across "
                  f"{len(module_names)} modules...")

        agent = self._build_modules()  # fresh agent for evaluation

        for idx, (_, row) in enumerate(X.iterrows()):
            preds = self._predict_row(row, agent)
            for name in module_names:
                predictions_raw[name].append(preds.get(name, 'common_cold'))

            if verbose and (idx + 1) % max(1, n_cases // 10) == 0:
                print(f"     Processed {idx + 1}/{n_cases} patients...")

        # 4. Encode predictions and compute metrics
        report: Dict = {}
        all_labels = self._label_encoder.classes_

        for name in module_names:
            y_pred_raw = predictions_raw[name]

            # Encode; map unseen labels to 'common_cold'
            y_pred_clean = [
                lbl if lbl in self._label_encoder.classes_
                else 'common_cold'
                for lbl in y_pred_raw
            ]
            y_pred_encoded = self._label_encoder.transform(y_pred_clean)

            # --- Accuracy ---
            acc = accuracy_score(y_true_encoded, y_pred_encoded)

            # --- Precision, Recall, F1 (macro-averaged) ---
            prec = precision_score(
                y_true_encoded, y_pred_encoded, average='macro', zero_division=0
            )
            rec = recall_score(
                y_true_encoded, y_pred_encoded, average='macro', zero_division=0
            )
            f1 = f1_score(
                y_true_encoded, y_pred_encoded, average='macro', zero_division=0
            )

            # --- Confusion Matrix ---
            cm = confusion_matrix(
                y_true_encoded, y_pred_encoded,
                labels=range(len(all_labels))
            )

            # --- ROC-AUC (One-vs-Rest, macro) ---
            roc_auc = None
            y_score = None
            if name in self._modules:
                mod = self._modules[name]
                if name == 'MLClassifier' and hasattr(mod, 'predict'):
                    # Build probability scores from MLClassifier
                    try:
                        score_matrix = np.zeros((n_cases, len(all_labels)))
                        for i, (_, row) in enumerate(X.iterrows()):
                            symptoms = self._row_to_symptom_list(row)
                            result = mod.predict(symptoms)
                            if 'top5' in result:
                                probs = result.get('all_probs', None)
                                if probs is None:
                                    probs_dict = dict(result['top5'])
                                    for label in all_labels:
                                        idx_l = self._label_encoder.transform([label])[0]
                                        score_matrix[i, idx_l] = probs_dict.get(label, 0.0)
                                else:
                                    for label, prob in probs.items():
                                        lbl_norm = label.lower().replace(' ', '_')
                                        if lbl_norm in self._label_encoder.classes_:
                                            idx_l = self._label_encoder.transform([lbl_norm])[0]
                                            score_matrix[i, idx_l] = prob
                        # Re-normalise rows to sum exactly to 1.0. A module's
                        # own rounding (e.g. round(x, 4)) or a truncated
                        # top-N output can leave sums a few 1e-4 off, which
                        # fails sklearn's strict np.allclose(1, ...) check
                        # even though the values are "close enough" by eye.
                        row_sums = score_matrix.sum(axis=1, keepdims=True)
                        row_sums[row_sums == 0] = 1.0  # avoid div-by-zero
                        score_matrix = score_matrix / row_sums
                        if len(np.unique(y_true_encoded)) > 1:
                            roc_auc = roc_auc_score(
                                y_true_encoded, score_matrix,
                                multi_class='ovr', average='macro',
                                labels=range(len(all_labels))
                            )
                            y_score = score_matrix
                    except Exception:
                        roc_auc = None

                elif name == 'BayesianNet' and hasattr(mod, 'compute_posterior'):
                    try:
                        score_matrix = np.zeros((n_cases, len(all_labels)))
                        for i, (_, row) in enumerate(X.iterrows()):
                            symptoms = self._row_to_symptom_list(row)
                            posteriors = mod.compute_posterior(symptoms)
                            for disease, prob in posteriors.items():
                                if disease in self._label_encoder.classes_:
                                    idx_l = self._label_encoder.transform([disease])[0]
                                    score_matrix[i, idx_l] = prob
                        # Same row re-normalisation as above (see comment).
                        row_sums = score_matrix.sum(axis=1, keepdims=True)
                        row_sums[row_sums == 0] = 1.0
                        score_matrix = score_matrix / row_sums
                        if len(np.unique(y_true_encoded)) > 1:
                            roc_auc = roc_auc_score(
                                y_true_encoded, score_matrix,
                                multi_class='ovr', average='macro',
                                labels=range(len(all_labels))
                            )
                            y_score = score_matrix
                    except Exception:
                        roc_auc = None

                elif name == 'NeuralNetwork' and hasattr(mod, 'predict'):
                    # NeuralDiagnosticModel.predict() already returns a
                    # full 'all_probs' dict over all 8 classes, unlike
                    # MLClassifier's truncated 'top5' — no fallback needed.
                    try:
                        score_matrix = np.zeros((n_cases, len(all_labels)))
                        for i, (_, row) in enumerate(X.iterrows()):
                            symptoms = self._row_to_symptom_list(row)
                            result = mod.predict(symptoms)
                            probs = result.get('all_probs', {})
                            for label, prob in probs.items():
                                lbl_norm = label.lower().replace(' ', '_')
                                if lbl_norm in self._label_encoder.classes_:
                                    idx_l = self._label_encoder.transform([lbl_norm])[0]
                                    score_matrix[i, idx_l] = prob
                        # Same row re-normalisation as above (see comment).
                        row_sums = score_matrix.sum(axis=1, keepdims=True)
                        row_sums[row_sums == 0] = 1.0
                        score_matrix = score_matrix / row_sums
                        if len(np.unique(y_true_encoded)) > 1:
                            roc_auc = roc_auc_score(
                                y_true_encoded, score_matrix,
                                multi_class='ovr', average='macro',
                                labels=range(len(all_labels))
                            )
                            y_score = score_matrix
                    except Exception:
                        roc_auc = None

            module_report = {
                'accuracy':         round(acc, 4),
                'precision':        round(prec, 4),
                'recall':           round(rec, 4),
                'f1_score':         round(f1, 4),
                'confusion_matrix': cm,
                'roc_auc':          round(roc_auc, 4) if roc_auc is not None else None,
                'y_true':           y_true_encoded,
                'y_pred':           y_pred_encoded,
                'y_score':          y_score,
            }
            report[name] = module_report

        # 5. Add metadata
        report['metadata'] = {
            'csv_file':     csv_file,
            'num_cases':    n_cases,
            'num_classes':  len(all_labels),
            'class_labels': list(all_labels),
        }

        if verbose:
            self.print_report(report)

        return report

    # ------------------------------------------------------------------
    #  Reporting
    # ------------------------------------------------------------------
    @staticmethod
    def print_report(report: Dict):
        """Print a formatted performance report to the console."""
        metadata = report.get('metadata', {})
        print("\n" + "=" * 65)
        print("  SYSTEM PERFORMANCE REPORT")
        print("=" * 65)
        print(f"  Dataset       : {metadata.get('csv_file', 'N/A')}")
        print(f"  Cases         : {metadata.get('num_cases', 'N/A')}")
        print(f"  Disease classes: {metadata.get('num_classes', 'N/A')}")
        print("=" * 65)

        # Table header
        header = f"{'Module':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'ROC-AUC':>10}"
        print(f"\n  {header}")
        print("  " + "-" * len(header))

        for module_name in sorted(report.keys()):
            if module_name == 'metadata':
                continue
            m = report[module_name]
            acc = f"{m['accuracy']:.4f}"
            prec = f"{m['precision']:.4f}"
            rec = f"{m['recall']:.4f}"
            f1 = f"{m['f1_score']:.4f}"
            auc = f"{m['roc_auc']:.4f}" if m['roc_auc'] is not None else "  N/A  "
            print(f"  {module_name:<20} {acc:>10} {prec:>10} {rec:>10} {f1:>10} {auc:>10}")

        print("=" * 65)

        # Per-class breakdown summary
        print("\n  -- Per-Class Support --")
        if len(report) > 1:
            first_mod = [k for k in report if k != 'metadata'][0]
            y_true = report[first_mod]['y_true']
            classes = metadata.get('class_labels', [])
            for i, label in enumerate(classes):
                count = int(np.sum(y_true == i))
                print(f"    {label:<20} {count:>4} cases")

        print()

    # ------------------------------------------------------------------
    #  Convenience method: evaluate and visualise in one call
    # ------------------------------------------------------------------
    def evaluate_and_visualise(self,
                               csv_file: str = "test_data.csv",
                               save_dir: str = "reports/"):
        """
        Evaluate all modules and generate visualisations.

        Equivalent to calling ``evaluate()`` followed by
        ``PerformanceVisualizer.plot_all(report, save_dir)``.
        """
        report = self.evaluate(csv_file, verbose=True)

        from evaluation.visualizations import PerformanceVisualizer
        viz = PerformanceVisualizer()
        viz.plot_all(report, save_dir=save_dir)
        return report


# Stand-alone usage
if __name__ == "__main__":
    evaluator = ModelEvaluator(data_path="data/")

    # Place your CSV in data/test_data.csv with columns:
    #   fever, cough, fatigue, ..., disease
    report = evaluator.evaluate(csv_file="test_data.csv", verbose=True)

    # Optionally generate visualisations
    # evaluator.evaluate_and_visualise("test_data.csv")
