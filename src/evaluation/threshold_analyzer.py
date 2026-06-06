import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve
import warnings

warnings.filterwarnings("ignore")


class ThresholdAnalyzer:
    """
    Analizador de umbral óptimo para modelos de detección de anomalías.
    Calcula el threshold que maximiza el F1-Score usando la curva
    Precision-Recall, fundamental para modelos basados en scores continuos
    (Autoencoder reconstruction error, Isolation Forest anomaly score).
    """

    def __init__(self):
        self.analysis_results = {}

    def find_optimal_threshold(self, model_name: str, y_true: np.ndarray,
                                scores: np.ndarray) -> dict:
        """
        Encuentra el threshold óptimo que maximiza el F1-Score.
        
        Args:
            model_name: Identificador del modelo.
            y_true: Etiquetas reales binarias.
            scores: Scores continuos del modelo (mayor = más anómalo).
        
        Returns:
            Diccionario con threshold óptimo y métricas asociadas.
        """
        print(f"  📐 Analizando threshold óptimo para {model_name}...")

        # Curva Precision-Recall
        precision, recall, thresholds_pr = precision_recall_curve(y_true, scores)

        # F1 para cada threshold
        f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
        best_idx = np.argmax(f1_scores)

        optimal_threshold = thresholds_pr[best_idx]
        best_f1 = f1_scores[best_idx]
        best_precision = precision[best_idx]
        best_recall = recall[best_idx]

        # Curva ROC
        fpr, tpr, thresholds_roc = roc_curve(y_true, scores)

        result = {
            'model': model_name,
            'optimal_threshold': optimal_threshold,
            'best_f1': best_f1,
            'best_precision': best_precision,
            'best_recall': best_recall,
            'pr_curve': (precision, recall, thresholds_pr),
            'roc_curve': (fpr, tpr, thresholds_roc),
        }

        self.analysis_results[model_name] = result

        print(f"  ✅ Threshold óptimo: {optimal_threshold:.6f}")
        print(f"     F1={best_f1:.4f} | Precision={best_precision:.4f} | Recall={best_recall:.4f}")

        return result

    def get_predictions_at_threshold(self, scores: np.ndarray,
                                     threshold: float) -> np.ndarray:
        """Genera predicciones binarias usando un threshold específico."""
        return (scores >= threshold).astype(int)

    def compare_thresholds(self) -> pd.DataFrame:
        """Compara los thresholds óptimos de todos los modelos analizados."""
        rows = []
        for name, result in self.analysis_results.items():
            rows.append({
                'model': result['model'],
                'optimal_threshold': result['optimal_threshold'],
                'f1_at_optimal': result['best_f1'],
                'precision_at_optimal': result['best_precision'],
                'recall_at_optimal': result['best_recall'],
            })
        return pd.DataFrame(rows).sort_values('f1_at_optimal', ascending=False)
