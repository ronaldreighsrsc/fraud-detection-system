"""
V2.0: Experimental GPU-Accelerated Pipeline using NVIDIA RAPIDS

This script demonstrates how the Financial Fraud Detection pipeline scales 
to massive transactional datasets (millions of rows) using NVIDIA GPUs.

Requires:
- Linux Environment (Ubuntu/WSL2)
- NVIDIA GPU (Pascal architecture or newer)
- RAPIDS environment (cudf, cuml) installed

DO NOT RUN THIS ON NATIVE WINDOWS CPU ENVIRONMENTS.
"""

import warnings
warnings.filterwarnings("ignore")

try:
    import cudf
    import cuml
    import xgboost as xgb
    import numpy as np
    from cuml.ensemble import IsolationForest as cuIsolationForest
except ImportError:
    print("❌ ERROR: NVIDIA RAPIDS (cudf, cuml) not detected.")
    print("Please run this script in a Linux GPU environment (e.g., Google Colab, AWS EC2) with RAPIDS installed.")
    exit(1)

def run_rapids_fraud_pipeline():
    print("🚀 Iniciando Pipeline Acelerado por GPU (NVIDIA RAPIDS) - Fraud Detection...")
    
    # 1. Carga ultra-rápida desde disco a VRAM usando cuDF (reemplazo de Pandas)
    print("\n📂 Cargando millones de transacciones en VRAM (cuDF)...")
    df = cudf.read_csv("./data/raw/transactions.csv")
    
    # 2. Feature Engineering masivo en GPU
    print("🔧 Feature Engineering Financiero en GPU (Cero transferencia a CPU)...")
    
    # Velocidad de transacciones por cliente
    df = df.sort_values(['customer_id', 'transaction_date'])
    df['time_diff'] = df.groupby('customer_id')['transaction_date'].diff().dt.total_seconds()
    df['amount_diff'] = df.groupby('customer_id')['transaction_amount'].diff()
    df['transaction_velocity'] = df['amount_diff'] / (df['time_diff'] + 1)
    
    df = df.fillna(0)
    
    # Separar variables target
    y = df['is_fraud'].astype('int32')
    
    # Eliminar metadatos para quedarnos con X
    drop_cols = ['transaction_id', 'customer_id', 'transaction_date', 'merchant_id', 'is_fraud']
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    
    # Label encoding para variables categóricas
    if 'merchant_category' in X.columns:
        X['merchant_category'] = X['merchant_category'].astype('category').cat.codes
        
    print(f"  📐 Dataset final en GPU: {X.shape[0]:,} transacciones × {X.shape[1]} features")

    # 3. XGBoost entrenado directamente en GPU (Zero-Copy desde cuDF vía Apache Arrow)
    print("\n🌲 Entrenando XGBoost Classifier en GPU...")
    
    # Configurar parámetros para NVIDIA GPU
    xgb_params = {
        'tree_method': 'gpu_hist',       # La magia de RAPIDS
        'predictor': 'gpu_predictor',
        'objective': 'binary:logistic',
        'learning_rate': 0.05,
        'max_depth': 6,
        'eval_metric': 'logloss',
        'scale_pos_weight': float((len(y) - y.sum()) / y.sum()),
        'random_state': 42
    }
    
    # DMatrix maneja nativamente la memoria VRAM de cuDF
    dtrain = xgb.DMatrix(X, label=y)
    
    model = xgb.train(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=200
    )
    
    print("✅ Entrenamiento XGBoost Completado en milisegundos!")

    # 4. Isolation Forest en GPU (cuML)
    print("\n🕵️‍♂️ Entrenando Isolation Forest en GPU (cuML)...")
    iso_forest = cuIsolationForest(contamination=0.02, n_estimators=200, random_state=42)
    iso_forest.fit(X)
    
    print("✅ Entrenamiento Isolation Forest Completado!")
    
    print("\n🏆 Pipeline V2.0 Financiero (RAPIDS) ejecutado exitosamente en la tarjeta gráfica.")

if __name__ == "__main__":
    run_rapids_fraud_pipeline()
