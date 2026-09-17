import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve
import warnings

warnings.filterwarnings("ignore")


class ThresholdAnalyzer:
    """
    Analizador de umbral óptimo para modelos de detección de fraude (v2.0 Enterprise).
    Calcula tanto el threshold que maximiza el F1-Score (Precision-Recall)
    como la calibración económica por costo financiero real bajo la Ley 21.234 de Fraudes
    (asimetría de restitución 40:1).
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

        print(f"  ✅ Threshold óptimo (F1): {optimal_threshold:.6f}")
        print(f"     F1={best_f1:.4f} | Precision={best_precision:.4f} | Recall={best_recall:.4f}")

        return result

    def find_optimal_cost_threshold(self, y_true: np.ndarray, y_probs: np.ndarray,
                                    c_fn: float = 1_000_000.0, c_fp: float = 25_000.0,
                                    n_points: int = 100) -> dict:
        """
        Encuentra el umbral de corte que minimiza la pérdida económica total
        bajo la Ley 21.234 de Fraudes en Chile.
        
        Parámetros Financieros Calibrados:
            - c_fn: Costo de Falso Negativo (Fraude no detectado):
              Restitución obligatoria de hasta 35 UF (~$850.000) + peritaje + provisión CMF = ~$1.000.000 CLP
            - c_fp: Costo de Falso Positivo (Fricción cliente legítimo):
              Contact center + pérdida comisiones + churn = ~$25.000 CLP
            Ratio de asimetría económica: 40:1.
        """
        thresholds = np.linspace(0.05, 0.95, n_points)
        costs = []
        fns = []
        fps = []

        for th in thresholds:
            y_pred = (y_probs >= th).astype(int)
            fn = int(np.sum((y_true == 1) & (y_pred == 0)))
            fp = int(np.sum((y_true == 0) & (y_pred == 1)))
            total_cost = (c_fn * fn) + (c_fp * fp)
            costs.append(total_cost)
            fns.append(fn)
            fps.append(fp)

        best_idx = int(np.argmin(costs))
        optimal_theta = float(thresholds[best_idx])
        min_cost = float(costs[best_idx])

        # Comparar contra umbral ingenuo (0.50)
        idx_50 = int(np.argmin(np.abs(thresholds - 0.50)))
        cost_at_50 = float(costs[idx_50])
        savings = cost_at_50 - min_cost

        print(f"  💰 Umbral Óptimo Financiero (Ley 21.234): theta* = {optimal_theta:.3f}")
        print(f"     Pérdida Mínima: CLP ${min_cost:,.0f} | Ahorro vs th=0.50: CLP ${savings:,.0f}")

        return {
            'optimal_threshold': round(optimal_theta, 4),
            'min_expected_loss': min_cost,
            'cost_at_default_50': cost_at_50,
            'net_savings': savings,
            'false_negatives_at_opt': fns[best_idx],
            'false_positives_at_opt': fps[best_idx],
            'cost_ratio_fn_fp': round(c_fn / c_fp, 2)
        }

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
