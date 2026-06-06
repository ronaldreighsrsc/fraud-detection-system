import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)
import warnings

warnings.filterwarnings("ignore")


class FraudMetricsEngine:
    """
    Motor de evaluación comparativa para modelos de detección de fraude.
    Calcula métricas estándar y financieras para cada modelo,
    generando un reporte consolidado tipo "torneo" similar al
    ablation study del IPSA Hybrid Forecasting Model.

    Métricas calculadas:
        - Accuracy, Precision, Recall, F1-Score
        - AUC-ROC y AUC-PR (Average Precision)
        - Confusion Matrix
        - Costo financiero estimado (falsos negativos vs falsos positivos)
    """

    def __init__(self, cost_false_negative: float = 1000.0,
                 cost_false_positive: float = 50.0):
        """
        Args:
            cost_false_negative: Costo promedio de no detectar un fraude (USD).
            cost_false_positive: Costo de investigar una transacción legítima (USD).
        """
        self.cost_fn = cost_false_negative
        self.cost_fp = cost_false_positive
        self.results = {}

    def evaluate_model(self, model_name: str, y_true: np.ndarray,
                       y_pred: np.ndarray, y_scores: np.ndarray = None) -> dict:
        """
        Evalúa un modelo individual y almacena los resultados.
        
        Args:
            model_name: Nombre identificador del modelo.
            y_true: Etiquetas reales (0 = legítimo, 1 = fraude).
            y_pred: Predicciones binarias del modelo.
            y_scores: Scores continuos (probabilidades o anomaly scores).
        """
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        metrics = {
            'model': model_name,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn),
            'total_cost': fn * self.cost_fn + fp * self.cost_fp,
        }

        # AUC métricas (requieren scores continuos)
        if y_scores is not None:
            metrics['auc_roc'] = roc_auc_score(y_true, y_scores)
            metrics['auc_pr'] = average_precision_score(y_true, y_scores)
        else:
            metrics['auc_roc'] = None
            metrics['auc_pr'] = None

        self.results[model_name] = metrics
        return metrics

    def print_model_report(self, model_name: str, y_true: np.ndarray,
                           y_pred: np.ndarray) -> None:
        """Imprime un reporte detallado para un modelo específico."""
        print(f"\n{'='*60}")
        print(f"  📊 Reporte de Evaluación: {model_name}")
        print(f"{'='*60}")

        metrics = self.results.get(model_name)
        if metrics is None:
            print("  ⚠️ Modelo no evaluado aún.")
            return

        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")

        if metrics['auc_roc'] is not None:
            print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
            print(f"  AUC-PR:    {metrics['auc_pr']:.4f}")

        print(f"\n  Confusion Matrix:")
        print(f"    TP={metrics['true_positives']:,}  FP={metrics['false_positives']:,}")
        print(f"    FN={metrics['false_negatives']:,}  TN={metrics['true_negatives']:,}")
        print(f"\n  💰 Costo financiero estimado: ${metrics['total_cost']:,.0f}")
        print(f"     (FN × ${self.cost_fn:,.0f} + FP × ${self.cost_fp:,.0f})")

    def generate_comparison_table(self) -> pd.DataFrame:
        """
        Genera una tabla comparativa tipo torneo entre todos los modelos evaluados.
        Ordena por F1-Score descendente (métrica más relevante para datos desbalanceados).
        """
        if not self.results:
            print("  ⚠️ No hay modelos evaluados aún.")
            return pd.DataFrame()

        df = pd.DataFrame(list(self.results.values()))

        # Ordenar por F1-Score (más relevante que accuracy en desbalanceo)
        df = df.sort_values('f1_score', ascending=False).reset_index(drop=True)

        # Formatear para display
        display_cols = ['model', 'accuracy', 'precision', 'recall',
                        'f1_score', 'auc_roc', 'auc_pr', 'total_cost']
        df_display = df[display_cols].copy()

        return df_display

    def print_tournament_results(self) -> None:
        """Imprime los resultados del torneo de modelos en formato ASCII profesional."""
        df = self.generate_comparison_table()
        if df.empty:
            return

        print("\n" + "=" * 80)
        print("  🏆 TORNEO DE MODELOS — DETECCIÓN DE FRAUDE")
        print("  " + "-" * 76)
        print(f"  {'Modelo':<25} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} "
              f"{'F1':>8} {'AUC-ROC':>9} {'Costo ($)':>12}")
        print("  " + "-" * 76)

        for _, row in df.iterrows():
            auc = f"{row['auc_roc']:.4f}" if row['auc_roc'] is not None else "N/A"
            print(f"  {row['model']:<25} {row['accuracy']:>9.4f} {row['precision']:>10.4f} "
                  f"{row['recall']:>8.4f} {row['f1_score']:>8.4f} {auc:>9} "
                  f"${row['total_cost']:>11,.0f}")

        print("=" * 80)
        print(f"  🥇 Ganador: {df.iloc[0]['model']} (F1-Score: {df.iloc[0]['f1_score']:.4f})")

    def save_results(self, output_path: str = "./src/evaluation/results/fraud_tournament.csv") -> None:
        """Guarda los resultados del torneo en un CSV."""
        df = self.generate_comparison_table()
        df.to_csv(output_path, index=False)
        print(f"  💾 Resultados guardados en: {output_path}")
