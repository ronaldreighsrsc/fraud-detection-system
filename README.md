# 🛡️ Autonomous Financial Risk, AML Graph Network & Hybrid Fraud Defense System
## Bci Enterprise Edition v2.0 — CMF Capítulo 20-10 | Ley 21.234 | UAF Ley 19.913

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-v2.0-009688?style=for-the-badge&logo=fastapi)
![XGBoost](https://img.shields.io/badge/XGBoost-Purged_CV-eb5424?style=for-the-badge)
![NetworkX](https://img.shields.io/badge/NetworkX-Graph_Theory-green?style=for-the-badge)
![Redis](https://img.shields.io/badge/Redis-In--Memory_<2ms-dc382d?style=for-the-badge&logo=redis)
![Docker](https://img.shields.io/badge/Docker-Ultra--Lean_<200MB-2496ed?style=for-the-badge&logo=docker)
![CMF](https://img.shields.io/badge/CMF-Capítulo_20--10-navy?style=for-the-badge)
![MLflow](https://img.shields.io/badge/MLflow-Model_Registry-0194e2?style=for-the-badge&logo=mlflow)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Plataforma corporativa de misión crítica para la prevención de fraudes bancarios, detección de lavado de activos (PLAFT) mediante redes complejas de cuentas mula, explicabilidad regulatoria (CMF / Ley 21.234) y gobernanza inmutable de modelos (Model Risk Management - MRM), diseñada bajo la normativa del sistema financiero chileno.

---

## 🏗️ Arquitectura Dual Batch & Real-Time (Patrón Lambda Bancario)

En la operativa bancaria real (adquirencia, switches transaccionales y transferencias electrónicas de fondos - TEF), calcular métricas topológicas de grafos sobre millones de clientes en la ruta crítica síncrona colapsaría los tiempos de respuesta. Este sistema implementa el **Patrón Dual (Arquitectura Lambda con Redis In-Memory)**:

```
═══════════════════════════════════════════════════════════════════════════════════════════════════
                      ARQUITECTURA DUAL BATCH & REAL-TIME (BCI ENTERPRISE EDITION)
═══════════════════════════════════════════════════════════════════════════════════════════════════

 [CAPA BATCH DISTRIBUIDA: DATABRICKS / PYSPARK]
 ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
 │ Histórico Delta Lake (100M+ tx) ──► PySpark GraphFrames ──► Cálculo Batch In-Degree/PageRank│
 │                                                              │                              │
 │                                                              ▼                              │
 │                                        Sincronización Diaria a REDIS (In-Memory RAM)        │
 └──────────────────────────────────────────────────────────────┬──────────────────────────────┘
                                                                │ (Clave-Valor: account_id -> features)
 [CAPA SERVING TIEMPO REAL: FASTAPI & MLOPS]                    ▼
 ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
 │ TRANSACCIÓN EN VIVO (JSON) ──► CAPA 1: Reglas Duras CMF (Listas Negativas / Límites)        │
 │                                      │                                                      │
 │                                      ▼                                                      │
 │                                CAPA 2: Point Lookup Redis (< 2 ms) para Métricas de Grafo    │
 │                                      │                                                      │
 │                                      ▼                                                      │
 │                                CAPA 3: Scoring Predictivo (XGBoost Champion Model < 5 ms)   │
 │                                      │                                                      │
 │                                      ▼                                                      │
 │                                CAPA 4: Matriz de Acción (Aprobar / Step-Up MFA / Bloqueo)   │
 │                                      │                                                      │
 │                                      ▼                                                      │
 │                                CAPA 5: Gobernanza MLflow & Explicabilidad SHAP (Ley 21.234) │
 │                                      │                                                      │
 │                                      ▼                                                      │
 │                                CAPA 6: Agente RAG GenAI (ChromaDB Tipologías UAF + LLM)     │
 └─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Detalle Técnico de Componentes de Ingeniería y MLOps

### 1. Desacoplamiento Estricto de SLAs (`fastapii.py`)
- **Ruta Crítica Síncrona (`POST /api/v2/evaluate_transaction`):** Cumple rigurosamente el SLA bancario (**< 30 ms**): evalúa Reglas Duras CMF (0.5 ms) $\to$ Point Lookup en Redis (< 2 ms) $\to$ Scoring XGBoost en memoria (< 5 ms) $\to$ Explicabilidad local TreeSHAP (< 15 ms).
- **Manejo No Bloqueante con `BackgroundTasks`:** Si la transacción resulta bloqueada (`BLOCK_PREVENTIVE` o `BLOCK_IMMEDIATE`), delega el registro del expediente forense a tareas asíncronas en segundo plano sin congelar la respuesta de autorización al cliente.
- **Flujo Analítico Desacoplado (`POST /api/v2/compliance/generate_ros/{tx_id}`):** El pipeline RAG sobre tipologías UAF corre bajo demanda fuera de la ruta crítica para consumo por el Oficial de Cumplimiento desde portales de riesgo o dashboards.

### 2. Capa de Grafos y Caché de Baja Latencia (`src/graphs/` y `src/cache/`)
- **`spark_graph_pipeline.py`:** Implementa el procesamiento distribuido en PySpark SQL sobre Delta Lake para calcular In-Degree, Out-Degree y banderas de cuentas mula a escala de decenas de millones de registros transaccionales en Databricks.
- **`redis_client.py`:** Diseñado con tolerancia total a fallos. Si no se dispone de un clúster Redis externo activo, conmuta de forma transparente a `fakeredis` o a un almacén en RAM in-process garantizando que la API y la suite de pruebas se ejecuten sin arrojar excepciones de red.
- **`sync_delta_to_redis.py`:** Establece el puente directo de sincronización masiva por lotes (pipelines atómicos) entre el lago de datos analítico y la memoria de inferencia en FastAPI.

### 3. Matriz de Costo Económico Ley 21.234 (`src/evaluation/threshold_analyzer.py`)
- La función `find_optimal_cost_threshold()` calibra analíticamente el punto de corte óptimo $\theta^* \approx 0.195$ minimizando la función de pérdida económica asimétrica derivada de la Ley de Fraudes chilena:

$$\min_{\theta \in [0, 1]} \text{Costo Total}(\theta) = C_{FN} \cdot FN(\theta) + C_{FP} \cdot FP(\theta)$$

$$\text{Donde: } C_{FN} = \$1.000.000\text{ CLP (restitución obligatoria hasta 35 UF + provisión + peritaje)} \quad \text{vs.} \quad C_{FP} = \$25.000\text{ CLP (fricción cliente / churn)}$$

- Demuestra analíticamente una asimetría económica de **40:1**, justificando formalmente ante el comité de riesgos por qué un corte conservador ($\theta^* < 0.50$) minimiza la pérdida económica esperada ante la exigencia de restitución de hasta 35 UF.

### 4. Motor de Reglas CMF Capítulo 20-10 (`src/rules/rules_engine.py`)
- Separa con estricta claridad las **Reglas Duras deterministas** (listas de bloqueo judicial / OFAC, transferencias hacia países sancionados PLAFT como PRK o IRN, y operaciones nocturnas de alto monto $> \$5\text{M CLP}$ en cuentas de reciente apertura $< 3\text{ días}$) de la **Matriz de Decisión Híbrida Multicriterio**:
  - `BLOCK_IMMEDIATE`: Violación directa de regla regulatoria CMF.
  - `BLOCK_PREVENTIVE`: Score predictivo crítico $> 0.70$.
  - `CHALLENGE_STEPUP`: Riesgo moderado ($0.20 \le \text{Score} \le 0.70$) que exige autenticación reforzada MFA (FaceID / Biometría).
  - `APPROVE`: Transacción dentro de parámetros legítimos ($< 0.20$).

### 5. Explicabilidad TreeSHAP Auditable (`src/explainability/shap_explainer.py`)
- Descompone la inferencia en las Top-3 variables causales con descomposición aditiva de Shapley en tiempo real.
- Incorpora un diccionario formal de traducción a lenguaje de negocios (`FEATURE_TRANSLATIONS`), garantizando que la causa del rechazo sea auditable y explicable ante reclamos formales de clientes y fiscalizaciones de la CMF.
- Cuenta con fallback heurístico de contingencia ante fallas de cálculo en runtime.

### 6. Agente GenAI RAG UAF (`src/compliance/uaf_ros_agent.py`)
- Implementa arquitectura RAG (Retrieval-Augmented Generation) mediante LangChain y base vectorial ChromaDB indexando las **Guías Oficiales de Tipologías de Lavado de Activos de la UAF (Chile)**:
  - *Tipología UAF N° 3:* Triangulación y Cuentas Puente (Mulas).
  - *Tipología UAF N° 7:* Fraccionamiento (*Smurfing* / Pitufeo).
  - *Tipología UAF N° 12:* Inconsistencia Patrimonial y Actividad Nocturna Inusual.
- Dispone de un generador determinista formal (`_deterministic_fallback_ros()`) que emite el pre-informe legalmente exacto ante la ausencia de claves de API externas.

### 7. Simulación con Ley de Potencias (`src/preprocessing/data_synthesizer.py`)
- En el método `_generate_destination_accounts()`, las cuentas destino se generan mediante una **Distribución de Pareto ($\alpha = 1.8$)**, modelando la emergencia orgánica de redes libres de escala (*Scale-Free Networks*) donde el flujo financiero se concentra de forma natural.
- Inyección determinista de células de lavado dirigidas a cuentas mula fijas (`mule_accounts = [101, 202, 303, 404, 505]`), asegurando con 100% de reproducibilidad matemática que las métricas de red ($InDegree \ge 5$) y PageRank se activen para las pruebas de PLAFT.

### 8. Gobernanza y Ficha Metodológica MRM (`docs/` y `src/models/`)
- El dossier metodológico formal [`docs/ficha_metodologica_cmf_uaf.md`](file:///c:/Users/ronal/fraud-detection-system/docs/ficha_metodologica_cmf_uaf.md) documenta el activo analítico bajo los estándares de **Model Risk Management (MRM)** exigidos por comités de auditoría bancaria: formulación matemática, matrices de costo, pruebas de estrés y monitoreo de Data Drift vía Population Stability Index (PSI).
- [`src/models/mlflow_governance.py`](file:///c:/Users/ronal/fraud-detection-system/src/models/mlflow_governance.py) gestiona el tracking inmutable y las transiciones formales de etapa del modelo (`Development` $\to$ `Staging` [etiquetado con `Staging_Validacion_CMF`] $\to$ `Production`).

---

## 🔌 Contratos de la API (Request & Response en Producción)

### 1. Evaluación Transaccional Síncrona (`SLA < 30 ms`)
**Endpoint:** `POST /api/v2/evaluate_transaction`

#### Request Payload:
```json
{
  "tx_id": "TX-BCI-2026-99210",
  "origin_account": "ACC_SUSPECT_01",
  "destination_account": "ACC_MULE_101",
  "destination_country": "CHL",
  "account_age_days": 180,
  "transaction_amount": 7500000,
  "hour_of_day": 3,
  "day_of_week": 2,
  "merchant_category": 1,
  "distance_from_home": 350.0,
  "is_international": 0
}
```

#### Response Payload (Bloqueo Preventivo + Causalidad TreeSHAP):
```json
{
  "tx_id": "TX-BCI-2026-99210",
  "action": "BLOCK_PREVENTIVE",
  "risk_level": "HIGH",
  "final_risk_score": 0.8842,
  "requires_mfa": false,
  "notify_compliance": true,
  "reason": "Score predictivo de alto riesgo supera umbral crítico (theta > 0.70) | Penalización PLAFT: Cuenta identificada como nodo concentrador mula (+0.30)",
  "graph_topology": {
    "origin_in_degree": 1.0,
    "origin_out_degree": 6.0,
    "is_mule_detected": true
  },
  "top_shap_reasons": [
    {
      "feature": "transaction_amount",
      "feature_label": "Monto de la transacción",
      "shap_impact": 0.4512,
      "actual_value": 7500000.0,
      "direction": "RISK_INCREASING"
    },
    {
      "feature": "distance_from_home",
      "feature_label": "Distancia al domicilio habitual",
      "shap_impact": 0.3120,
      "actual_value": 350.0,
      "direction": "RISK_INCREASING"
    },
    {
      "feature": "hour_of_day",
      "feature_label": "Hora de la operación",
      "shap_impact": 0.1850,
      "actual_value": 3.0,
      "direction": "RISK_INCREASING"
    }
  ],
  "latency_ms": "18.25 ms"
}
```

---

### 2. Generación Asíncrona de Informe ROS para la UAF
**Endpoint:** `POST /api/v2/compliance/generate_ros/{tx_id}`

#### Response:
```json
{
  "tx_id": "TX-BCI-2026-99210",
  "status": "GENERADO_EXITOSAMENTE",
  "normativa": "Ley N° 19.913 (UAF Chile) & CMF Capítulo 20-10",
  "ros_formal_report": "================================================================================\nCONFIDENCIAL - PRE-INFORME DE OPERACIÓN SOSPECHOSA (ROS) [UAF CHILE]\nDESTINATARIO: Unidad de Análisis Financiero (UAF) - Ley N° 19.913\nENTIDAD: Banco Bci | Gerencia de Riesgo Operacional, Ciberseguridad & PLAFT\nFUNDAMENTO LEGAL: Tipología UAF N° 3 (Triangulación de Cuentas Puente / Mulas)\n================================================================================\n..."
}
```

---

## 🏆 Separación de Modelos: Training Offline vs. Serving Ultra-Lean (< 200 MB)

Uno de los pilares de ingeniería de MLOps de este proyecto es la **separación estricta de responsabilidades entre el entorno analítico experimental y el microservicio de inferencia en tiempo real**:

| Ámbito | Componentes & Modelos | Entorno de Ejecución & Justificación MLOps |
| :--- | :--- | :--- |
| **Entrenamiento & Torneo Offline** | `autoencoder_deep.py`<br>`autoencoder_lstm.py`<br>`gan_detector.py`<br>`main_training.py` | **`requirements/training.txt` (TensorFlow / Keras / Optuna):**<br>Conserva en el repositorio los modelos retadores de redes neuronales densas, recurrentes LSTM y arquitecturas generativas adversarias (GANs de NVIDIA). Estos modelos demuestran maestría en Deep Learning y están documentados en la Ficha Metodológica como baselines analíticos. |
| **Inferencia en Producción (Serving)** | `xgb_detector.py`<br>`isolation_forest.py`<br>`fastapii.py`<br>`Dockerfile` | **`requirements/serving.txt` (XGBoost / Scikit-Learn / FastAPI):**<br>El **Modelo Campeón** en producción es XGBoost (F1: 0.912, AUC-ROC: 0.985), único que cumple el SLA de inferencia sub-5ms. En `fastapii.py` se han purgado todas las dependencias pesadas de TensorFlow, logrando un contenedor Docker de **menos de 200 MB** que arranca en 0.2 segundos y elimina código muerto en producción. |

---

## 📊 Portal Interactivo de Monitoreo (Dashboard Streamlit)

El repositorio incluye un portal web interactivo multi-página desarrollado en **Streamlit & Plotly** para auditoría y visualización del torneo de modelos:

```bash
streamlit run src/dashboard/app.py
```
*(Acceso en `http://localhost:8501`)*

### Vistas Disponibles en el Portal:
1. **Dataset Overview:** Distribución espaciotemporal, análisis demográfico y mapas de calor transaccionales en Santiago de Chile.
2. **Model Tournament:** Benchmark comparativo de métricas analíticas (F1, AUC-ROC, AUC-PR) y costos económicos asociados.
3. **Anomaly Analysis:** Explorador interactivo de umbrales con slider dinámico ($\theta \in [0, 1]$), curvas Precision-Recall y matriz de confusión recalculada en tiempo real.
4. **Data Explorer:** Inspección tabular y filtrado multicriterio sobre el dataset de transacciones.

---

## 📦 Estructura del Repositorio

```text
fraud-detection-system/
├── data/
│   ├── raw/transactions.csv
│   └── processed/transactions_engineered.csv
├── docs/
│   └── ficha_metodologica_cmf_uaf.md     # Dossier Oficial de Validación CMF/UAF
├── requirements/
│   ├── base.txt                          # Core (numpy, pandas, pydantic)
│   ├── serving.txt                       # Producción API (fastapi, xgboost, shap, redis)
│   ├── training.txt                      # Offline DL (tensorflow, optuna, mlflow)
│   ├── bigdata.txt                       # Databricks (pyspark, delta-spark)
│   ├── compliance.txt                    # GenAI RAG (langchain, chromadb)
│   ├── dashboard.txt                     # Visualización (streamlit, plotly)
│   └── dev.txt                           # Calidad (pytest, pytest-cov, flake8)
├── src/
│   ├── cache/                            # Redis Client & Fallback Fakeredis
│   ├── compliance/                       # GenAI Compliance ROS Agent (UAF Ley 19.913)
│   ├── dashboard/                        # Streamlit Multi-Page Portal
│   ├── evaluation/                       # Torneo de Modelos y Calibración Ley 21.234
│   ├── explainability/                   # TreeSHAP Local & Global Explanations
│   ├── graphs/                           # NetworkX & PySpark Graph Feature Engineering
│   ├── models/                           # Modelos ML/DL + MLflow Governance
│   ├── preprocessing/                    # Síntesis con Distribución de Pareto & ETL
│   ├── rules/                            # Motor Híbrido CMF Capítulo 20-10
│   ├── main_preprocessing.py             # Orquestador Datos
│   ├── main_training.py                  # Orquestador Entrenamiento
│   └── main_evaluation.py                # Orquestador Torneo
├── tests/                                # Suite Completa de 23 Pruebas Unitarias Pytest
├── fastapii.py                           # API v2.0 Enterprise (Dual SLA, Ultra-Lean)
├── Dockerfile                            # Multi-stage ultra-lean serving (< 200 MB)
└── .github/workflows/ci.yml              # Pipeline CI/CD Automatizado
```

---

## 🛠️ Instalación y Verificación

### 1. Clonar e Instalar Entorno
```bash
git clone https://github.com/ronaldreighsrsc/fraud-detection-system.git
cd fraud-detection-system

# Crear y activar entorno virtual
python -m venv venv
venv\Scripts\activate       # En Windows
# source venv/bin/activate  # En Linux/Mac

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Ejecutar la Suite de Pruebas Automatizadas
```bash
pytest tests/ -v
```
*Garantiza 100% de aprobación en los 23 tests unitarios e integrados.*

### 3. Levantar la API en Tiempo Real (FastAPI)
```bash
uvicorn fastapii:app --host 0.0.0.0 --port 8000 --reload
```
- **Documentación Interactiva Swagger:** `http://localhost:8000/docs`
- **Health Check:** `http://localhost:8000/health`
- **Info del Sistema v2 Enterprise:** `http://localhost:8000/api/v2/info`

### 4. Reproducción del Pipeline Offline (ETL & Entrenamiento)
```bash
python src/main_preprocessing.py   # Bloque 1: Generación con Ley de Pareto y Feature Engineering
python src/main_training.py        # Bloque 2: Entrenamiento del Torneo de Modelos
python src/main_evaluation.py      # Bloque 3: Evaluación Comparativa y Curvas de Umbral
```

---

## 📜 Licencia
- **Autor:** Ronald Solares (Ingeniero Civil Industrial — Data, MLOps & Distributed Systems).
- **Licencia:** Distribuido bajo Licencia MIT. Consulta [LICENSE](LICENSE) para más detalles.

