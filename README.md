# 🛡️ Autonomous Financial Risk, AML Graph Network & Hybrid Fraud Defense System
## Bci Enterprise Edition v2.0 — CMF Capítulo 20-10 | Ley 21.234 | UAF Ley 19.913

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-v2.0-009688?style=for-the-badge&logo=fastapi)
![XGBoost](https://img.shields.io/badge/XGBoost-Purged_CV-eb5424?style=for-the-badge)
![NetworkX](https://img.shields.io/badge/NetworkX-Graph_Theory-green?style=for-the-badge)
![Redis](https://img.shields.io/badge/Redis-In--Memory_<2ms-dc382d?style=for-the-badge&logo=redis)
![Docker](https://img.shields.io/badge/Docker-Ultra--Lean-2496ed?style=for-the-badge&logo=docker)
![CMF](https://img.shields.io/badge/CMF-Capítulo_20--10-navy?style=for-the-badge)

Sistema corporativo integral de defensa contra el fraude financiero, lavado de activos (PLAFT) y gobernanza de modelos de riesgo (Model Risk Management - MRM), diseñado bajo los estándares de misión crítica de **Banco Bci** y la regulación financiera chilena.

---

## 🏗️ Arquitectura Dual Batch & Real-Time (Patrón Lambda Bancario)

En la operativa bancaria real, calcular métricas topológicas de grafos sobre millones de clientes en la ruta crítica colapsaría los tiempos de respuesta. Este sistema implementa una **Arquitectura Dual con Redis In-Memory**:

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
 │                                CAPA 3: Scoring Predictivo (XGBoost + Autoencoders)          │
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

## ⚡ Módulos Estratégicos v2.0 Enterprise

| Componente | Descripción de Grado Bancario | Ubicación en Repo |
| :--- | :--- | :--- |
| **Teoría de Grafos & PLAFT** | Grafo dirigido de transferencias con NetworkX y PySpark. Detección de cuentas concentradoras (*mulas*) y carruseles de lavado según distribución de Pareto ($\alpha=1.8$). | [`src/graphs/`](file:///c:/Users/ronal/fraud-detection-system/src/graphs) |
| **Caché en Memoria (Redis)** | Point-lookup atómico $O(1)$ en `< 2 ms` con tolerancia a fallos mediante `fakeredis` para desarrollo hermético y CI/CD. | [`src/cache/`](file:///c:/Users/ronal/fraud-detection-system/src/cache) |
| **Motor Híbrido de Reglas** | Evaluación de mandatos CMF Capítulo 20-10 (listas judiciales, países sancionados, transacciones nocturnas) previa al score ML. | [`src/rules/`](file:///c:/Users/ronal/fraud-detection-system/src/rules) |
| **Explicabilidad TreeSHAP** | Causalidad auditable local de bloqueos para cumplimiento estricto de la **Ley 21.234 de Fraudes** ante reclamos de clientes. | [`src/explainability/`](file:///c:/Users/ronal/fraud-detection-system/src/explainability) |
| **Calibración Financiera 40:1** | Optimización de umbral $\theta^*$ que minimiza la pérdida económica por restitución obligatoria de hasta **35 UF** ($C_{FN}=\$1M$ vs $C_{FP}=\$25K$). | [`src/evaluation/`](file:///c:/Users/ronal/fraud-detection-system/src/evaluation) |
| **Agente GenAI UAF** | Asistente generativo RAG (ChromaDB + LLM) con las Guías Oficiales de Tipologías UAF (N° 3, 7 y 12) y fallback formal determinista. | [`src/compliance/`](file:///c:/Users/ronal/fraud-detection-system/src/compliance) |
| **Gobernanza MLflow** | Registro inmutable de artefactos y transiciones formales de ciclo de vida (`Development` $\to$ `Staging_Validacion_CMF` $\to$ `Production`). | [`src/models/`](file:///c:/Users/ronal/fraud-detection-system/src/models) |
| **Ficha Metodológica MRM** | Dossier regulatorio completo en Markdown exigido por los comités de auditoría de riesgo financiero. | [`docs/ficha_metodologica_cmf_uaf.md`](file:///c:/Users/ronal/fraud-detection-system/docs/ficha_metodologica_cmf_uaf.md) |

---

## 🚀 Desacoplamiento Estricto de SLA Bancario

```
[TRANSACCIÓN ENTRADA] ──► POST /api/v2/evaluate_transaction (SLA < 30 ms)
                                      │
                                      ├── 1. Reglas Duras CMF (0.5 ms)
                                      ├── 2. Point Lookup Redis (< 2 ms)
                                      ├── 3. Scoring XGBoost en memoria (< 5 ms)
                                      └── 4. Explicabilidad TreeSHAP (< 15 ms)
                                      │
                                      ▼
                        ¿Bloqueo Preventivo o Inmediato?
                         ├── (NO) ──► Retorna Aprobado / Step-Up MFA (< 30 ms)
                         └── (SÍ) ──► Retorna Bloqueo (< 30 ms)
                                      + Encola Tarea Background (FastAPI)
                                      │
                                      ▼ (Asíncrono / Back-office)
                                     POST /api/v2/compliance/generate_ros/{tx_id}
                                     (Generación RAG UAF bajo demanda del Oficial de Cumplimiento)
```

---

## 📦 Estructura Modular del Repositorio

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
│   ├── models/                           # 5 Modelos ML/DL + MLflow Governance
│   ├── preprocessing/                    # Síntesis con Distribución de Pareto & ETL
│   ├── rules/                            # Motor Híbrido CMF Capítulo 20-10
│   ├── main_preprocessing.py             # Orquestador Datos
│   ├── main_training.py                  # Orquestador Entrenamiento
│   └── main_evaluation.py                # Orquestador Torneo
├── tests/                                # Suite Completa de Pruebas Unitarias Pytest
├── fastapii.py                           # API v2.0 Enterprise (Dual SLA)
├── Dockerfile                            # Multi-stage ultra-lean serving (< 250 MB)
└── .github/workflows/ci.yml              # Pipeline CI/CD Automatizado
```

---

## 🛠️ Instalación y Puesta en Marcha

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

### 3. Levantar la API en Tiempo Real (FastAPI)
```bash
uvicorn fastapii:app --host 0.0.0.0 --port 8000 --reload
```
- **Documentación Interactiva Swagger:** `http://localhost:8000/docs`
- **Health Check:** `http://localhost:8000/health`
- **Info v2 Enterprise:** `http://localhost:8000/api/v2/info`

---

## 🎯 Speech Táctico para la Entrevista Técnica en Banco Bci

> *"Diseñé una plataforma de defensa multicapa para riesgos no financieros inspirada directamente en la operativa bancaria chilena y la normativa CMF Capítulo 20-10 y UAF.*
> 
> *En primer lugar, implementé una **Arquitectura Dual Lambda**: en la capa batch distribuida, jobs de **PySpark en Databricks** procesan transferencias sobre Delta Lake para calcular métricas de **Teoría de Grafos** (In-Degree, PageRank y detección de Cuentas Mula para PLAFT), persistiendo los atributos en **Redis** para que la API en **FastAPI** haga un point lookup atómico en menos de 2 milisegundos sin tocar disco.*
> 
> *En segundo lugar, integré un **Motor Híbrido** donde primero actúan reglas duras deterministas de la CMF y luego el scoring predictivo de XGBoost y Autoencoders. Como Ingeniero Civil Industrial, no calibré el modelo por F1-Score simétrico, sino por **Costo Económico Real bajo la Ley 21.234**, con una matriz asimétrica 40:1 que pondera la restitución obligatoria de hasta 35 UF frente a la fricción operativa.*
> 
> *En tercer lugar, para cumplir con la explicabilidad obligatoria de rechazos exigida por el regulador, incorporé **TreeSHAP** para auditar en tiempo real las 3 causas de cada bloqueo. Todo el ciclo de vida está documentado formalmente en la **Ficha Metodológica MRM** y registrado en **MLflow**.*
> 
> *Y finalmente, desarrollé un agente **RAG con ChromaDB y LangChain** indexado con las Guías Oficiales de Tipologías de la UAF, que redacta de forma autónoma el borrador del Reporte de Operaciones Sospechosas (ROS) listo para el Oficial de Cumplimiento.*
> 
> *Todo el sistema corre containerizado en Docker con CI/CD automatizado en GitHub Actions."*
