import os
import numpy as np
from evaluation.metrics_engine import FraudMetricsEngine
from evaluation.threshold_analyzer import ThresholdAnalyzer
import warnings

warnings.filterwarnings("ignore")


def run_evaluation_pipeline():
    """
    Bloque 3: Pipeline de Evaluación Comparativa.
    Carga las predicciones de los tres modelos y ejecuta el torneo
    de métricas con análisis de costo financiero.
    """
    print("🚀 Iniciando Pipeline de Evaluación (Bloque 3)...")

    results_dir = "./src/evaluation/results/"

    # --- PASO 1: Cargar Predicciones ---
    print("\n--- PASO 1: Cargando Predicciones ---")
    y_test = np.load(os.path.join(results_dir, "y_test.npy"))
    print(f"  📊 Muestras de test: {len(y_test):,} (Fraudes: {y_test.sum():,})")

    models_data = {
        'Deep Autoencoder': {
            'preds': np.load(os.path.join(results_dir, "ae_predictions.npy")),
            'scores': np.load(os.path.join(results_dir, "ae_scores.npy")),
        },
        'XGBoost Classifier': {
            'preds': np.load(os.path.join(results_dir, "xgb_predictions.npy")),
            'scores': np.load(os.path.join(results_dir, "xgb_probs.npy")),
        },
        'Isolation Forest': {
            'preds': np.load(os.path.join(results_dir, "iso_predictions.npy")),
            'scores': np.load(os.path.join(results_dir, "iso_scores.npy")),
        },
    }

    # --- PASO 2: Evaluación de Métricas ---
    print("\n--- PASO 2: Evaluación de Métricas ---")
    engine = FraudMetricsEngine(
        cost_false_negative=1000.0,   # Costo de un fraude no detectado
        cost_false_positive=50.0,     # Costo de investigar transacción legítima
    )

    for model_name, data in models_data.items():
        engine.evaluate_model(
            model_name=model_name,
            y_true=y_test,
            y_pred=data['preds'],
            y_scores=data['scores'],
        )
        engine.print_model_report(model_name, y_test, data['preds'])

    # --- PASO 3: Análisis de Threshold ---
    print("\n\n--- PASO 3: Análisis de Threshold Óptimo ---")
    analyzer = ThresholdAnalyzer()
    for model_name, data in models_data.items():
        analyzer.find_optimal_threshold(model_name, y_test, data['scores'])

    # --- PASO 4: Resultados del Torneo ---
    engine.print_tournament_results()
    engine.save_results(os.path.join(results_dir, "fraud_tournament.csv"))

    print(f"\n✅ ¡Pipeline de evaluación completado!")


if __name__ == "__main__":
    run_evaluation_pipeline()
