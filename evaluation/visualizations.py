# ============================================================
# EVALUATION MODULE: Performance Visualizations
# Generates visual reports for the system performance metrics.
# ============================================================

import os
import warnings
from typing import Dict, List, Optional
import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

warnings.filterwarnings('ignore')

# Styling
sns.set_style("whitegrid")
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor':   '#f8f9fa',
    'font.family':      'sans-serif',
    'font.size':        11,
})


class PerformanceVisualizer:
    """
    Generates publication-quality performance visualisations.

    Plot types
    ----------
    1. Confusion matrix heatmap  (per module)
    2. Multi-class ROC curves    (per module)
    3. Metrics comparison bar    (all modules side-by-side)
    4. Combined dashboard        (all of the above in one figure)
    """

    def __init__(self):
        self._colour_palette = sns.color_palette("husl", 8)

    # ─────────────────────────────────────────────────────────
    #  Plot 1: Confusion Matrix
    # ─────────────────────────────────────────────────────────
    def plot_confusion_matrix(self,
                              module_name: str,
                              cm: np.ndarray,
                              labels: List[str],
                              save_dir: str = "reports/") -> str:
        """
        Plot a normalised confusion matrix heatmap.

        Parameters
        ----------
        module_name : str
            Name of the module (used in the title and filename).
        cm          : np.ndarray  (n_classes x n_classes)
            Confusion matrix from sklearn.metrics.confusion_matrix.
        labels      : list of str
            Class label names in the same order as the confusion matrix.
        save_dir    : str
            Directory to save the figure.

        Returns
        -------
        save_path : str
            Full path to the saved PNG file.
        """
        os.makedirs(save_dir, exist_ok=True)

        # Normalise by row (true labels) for percentages
        cm_norm = cm.astype('float') / (
            cm.sum(axis=1, keepdims=True) + 1e-10
        )

        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(
            cm_norm,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            linewidths=0.5,
            linecolor='white',
            cbar_kws={'label': 'Proportion', 'shrink': 0.8},
        )

        ax.set_xlabel("Predicted Label", fontweight='bold', fontsize=12)
        ax.set_ylabel("True Label", fontweight='bold', fontsize=12)
        ax.set_title(
            f"Confusion Matrix — {module_name}",
            fontweight='bold', fontsize=14, pad=15
        )
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        plt.setp(ax.yaxis.get_majorticklabels(), rotation=0)
        plt.tight_layout()

        save_path = os.path.join(save_dir, f"cm_{module_name.lower()}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return save_path

    # ─────────────────────────────────────────────────────────
    #  Plot 2: Multi-class ROC Curves
    # ─────────────────────────────────────────────────────────
    def plot_roc_curves(self,
                        module_name: str,
                        y_true: np.ndarray,
                        y_score: np.ndarray,
                        labels: List[str],
                        save_dir: str = "reports/") -> Optional[str]:
        """
        Plot One-vs-Rest ROC curves for each class.

        Parameters
        ----------
        module_name : str
            Module name (title and filename).
        y_true      : np.ndarray  (n_samples,) int-encoded true labels.
        y_score     : np.ndarray  (n_samples, n_classes) probability scores.
        labels      : list of str
            Class label names.
        save_dir    : str
            Directory to save the figure.

        Returns
        -------
        save_path : str or None
            Path to saved figure, or None if y_score is not available.
        """
        if y_score is None or y_true is None:
            return None

        os.makedirs(save_dir, exist_ok=True)

        n_classes = len(labels)

        # Binarise the true labels for OvR
        y_true_bin = label_binarize(y_true, classes=range(n_classes))

        fig, ax = plt.subplots(figsize=(9, 7))

        # Store macro-average components
        all_fpr = []
        all_tpr = []
        roc_aucs = []

        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            roc_aucs.append(roc_auc)
            all_fpr.append(fpr)
            all_tpr.append(tpr)

            ax.plot(
                fpr, tpr,
                lw=2,
                label=f"{labels[i]} (AUC = {roc_auc:.3f})",
                color=self._colour_palette[i % len(self._colour_palette)],
            )

        # Macro-average ROC curve
        all_fpr_concat = np.unique(np.concatenate(all_fpr))
        mean_tpr = np.zeros_like(all_fpr_concat)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr_concat, all_fpr[i], all_tpr[i])
        mean_tpr /= n_classes
        macro_auc = auc(all_fpr_concat, mean_tpr)

        ax.plot(
            all_fpr_concat, mean_tpr,
            lw=3, linestyle='--', color='navy',
            label=f"Macro-average (AUC = {macro_auc:.3f})",
        )

        # Diagonal reference line
        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random')

        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel("False Positive Rate", fontweight='bold', fontsize=12)
        ax.set_ylabel("True Positive Rate", fontweight='bold', fontsize=12)
        ax.set_title(
            f"Multi-class ROC Curves — {module_name}",
            fontweight='bold', fontsize=14, pad=15
        )
        ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = os.path.join(save_dir, f"roc_{module_name.lower()}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return save_path

    # ─────────────────────────────────────────────────────────
    #  Plot 3: Metrics Comparison (Bar Chart)
    # ─────────────────────────────────────────────────────────
    def plot_metrics_comparison(self,
                                report: Dict,
                                save_dir: str = "reports/") -> str:
        """
        Grouped bar chart comparing Accuracy, Precision, Recall,
        and F1-Score across all evaluated modules.

        Parameters
        ----------
        report   : dict
            The full report dictionary from ``ModelEvaluator.evaluate()``.
        save_dir : str
            Directory to save the figure.

        Returns
        -------
        save_path : str
            Path to saved figure.
        """
        os.makedirs(save_dir, exist_ok=True)

        modules = [k for k in sorted(report.keys()) if k != 'metadata']
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

        n_modules = len(modules)
        n_metrics = len(metrics)
        bar_width = 0.18
        x = np.arange(n_modules)

        fig, ax = plt.subplots(figsize=(12, 6))

        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            values = [
                report[mod].get(metric, 0) or 0
                for mod in modules
            ]
            offset = (i - n_metrics / 2 + 0.5) * bar_width
            bars = ax.bar(
                x + offset,
                values,
                bar_width,
                label=label,
                color=self._colour_palette[i % len(self._colour_palette)],
                edgecolor='white',
                linewidth=0.5,
            )
            # Annotate bars with values
            for bar, val in zip(bars, values):
                if val > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{val:.2f}",
                        ha='center', va='bottom',
                        fontsize=8, fontweight='bold',
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(modules, fontsize=10)
        ax.set_ylabel("Score", fontweight='bold', fontsize=12)
        ax.set_title(
            "Model Performance Comparison Across Modules",
            fontweight='bold', fontsize=14, pad=15
        )
        ax.set_ylim([0, 1.15])
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        save_path = os.path.join(save_dir, "metrics_comparison.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return save_path

    # ─────────────────────────────────────────────────────────
    #  Plot 4: Full Dashboard
    # ─────────────────────────────────────────────────────────
    def plot_all(self,
                 report: Dict,
                 save_dir: str = "reports/") -> List[str]:
        """
        Generate all visualisations and return the list of saved file paths.

        Creates for each module:
        - Confusion matrix heatmap
        - ROC curves (if probability scores are available)
        - One combined metrics comparison chart

        Parameters
        ----------
        report   : dict
            Full report from ``ModelEvaluator.evaluate()``.
        save_dir : str
            Directory to save figures.

        Returns
        -------
        saved_files : list of str
            Paths to all generated PNG files.
        """
        os.makedirs(save_dir, exist_ok=True)
        metadata = report.get('metadata', {})
        labels = metadata.get('class_labels', [])

        saved_files = []

        # ── Per-module plots ──
        for module_name in sorted(report.keys()):
            if module_name == 'metadata':
                continue
            m = report[module_name]

            # Confusion matrix
            cm = m.get('confusion_matrix')
            if cm is not None:
                path = self.plot_confusion_matrix(
                    module_name, cm, labels, save_dir
                )
                saved_files.append(path)
                print(f"  📊 Saved: {path}")

            # ROC curves
            y_true = m.get('y_true')
            y_score = m.get('y_score')
            if y_true is not None and y_score is not None:
                path = self.plot_roc_curves(
                    module_name, y_true, y_score, labels, save_dir
                )
                if path:
                    saved_files.append(path)
                    print(f"  📈 Saved: {path}")

        # ── Comparison chart ──
        comp_path = self.plot_metrics_comparison(report, save_dir)
        saved_files.append(comp_path)
        print(f"  📊 Saved: {comp_path}")

        print(f"\n  ✅ All visualisations saved to: {save_dir}/")
        return saved_files


# ─────────────────────────────────────────────────────────────
#  Stand-alone usage example
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":

    from evaluation.metric import ModelEvaluator

    evaluator = ModelEvaluator(data_path="data/")
    report = evaluator.evaluate(csv_file="test_data.csv", verbose=True)

    viz = PerformanceVisualizer()
    saved = viz.plot_all(report, save_dir="reports/")

    print(f"\n  Generated {len(saved)} visualisation files.")
