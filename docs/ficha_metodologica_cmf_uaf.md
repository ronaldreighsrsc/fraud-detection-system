# 📋 Ficha Metodológica del Modelo: Sistema Híbrido de Prevención de Fraude, Redes Complejas y PLAFT
**Entidad:** Banco Bci | Gerencia de Riesgo Operacional, Ciberseguridad & Analítica Avanzada  
**Código Interno del Modelo:** ML-RISK-2026-v2.0 (Bci Enterprise Edition)  
**Normativa Regulatoria Asociada:** CMF Capítulo 20-10 | Ley N° 21.234 | Ley N° 19.913 (UAF)  
**Trazabilidad MLflow:** `models:/Bci_XGBoost_Fraud_Detector/Production` (SHA256 inmutable)  
**Infraestructura:** Databricks Apache Spark 3.5 sobre Delta Lake | In-Memory Redis Cache (<2ms) | Inferencia FastAPI Containerizada (<30ms)

---

## 1. Resumen Ejecutivo y Objetivo de Negocio
El presente activo analítico tiene como objetivo mitigar el riesgo no financiero asociado a fraudes en transferencias electrónicas de fondos (TEF), tarjetas de crédito/débito y delitos de lavado de activos mediante redes de cuentas mula.

A diferencia de aproximaciones académicas convencionales, este sistema optimiza directamente la **función de costo financiero del banco** derivada del marco legal chileno (**Ley 21.234**), minimizando las provisiones y devoluciones forzosas mediante un esquema de decisión sensible al costo (*Cost-Sensitive Decision Matrix* 40:1).

---

## 2. Marco Normativo y Regulatorio Aplicable

### 2.1. CMF Capítulo 20-10 (Gestión de Seguridad de la Información y Continuidad Operacional)
Establece los requerimientos de gobernanza y control sobre eventos de ciberfraude, exigiendo mecanismos preventivos en tiempo real, segregación de listas judiciales de bloqueo y capacidad de interrupción inmediata de operaciones anómalas.

### 2.2. Ley N° 21.234 (Responsabilidad Financiera y Restitución por Fraudes)
Fija la restitución obligatoria por parte del banco de los fondos reclamados por el usuario hasta un límite de **35 UF** (aproximadamente CLP $1.350.000) en un plazo perentorio de 5 a 7 días hábiles, salvo que el banco acredite dolo o culpa grave ante los tribunales.  
*Impacto Analítico:* La pérdida por Falso Negativo ($C_{FN}$) supera con creces el costo operativo de autenticación reforzada o fricción de un Falso Positivo ($C_{FP}$).

### 2.3. Ley N° 19.913 (Unidad de Análisis Financiero - UAF)
Obliga a las entidades bancarias a reportar oportunamente operaciones sospechosas de lavado de activos o financiamiento del terrorismo a través del Reporte de Operaciones Sospechosas (ROS), tipificando patrones de fraccionamiento (*smurfing*) y uso de cuentas puente/mula.

---

## 3. Universo de Datos y Metodología Anti-Leakage

- **Volumen Transaccional:** Dataset de 100,000 operaciones enriquecidas con atributos espaciotemporales y métricas de red.
- **Topología de Red:** Simulación de grafo libre de escala (*Scale-Free Network*) siguiendo la **Ley de Potencias (Distribución de Pareto con exponente $\alpha = 1.8$)**, garantizando la concentración realista de transacciones hacia nodos nodales y células delictivas inyectadas de cuentas mula.
- **Validación Cruzada Rigurosa (Purged & Embargoed K-Fold):**  
  Para prevenir el *Data Leakage* temporal propio de series financieras, se implementa la metodología de Marcos López de Prado:
  - **Purging:** Eliminación de datos de entrenamiento que se solapen con la ventana temporal del fold de test.
  - **Embargo:** Exclusión de 10 muestras posteriores al período de test para evitar efectos de autocorrelación serial.

---

## 4. Calibración del Umbral Óptimo por Costo Financiero Real

La optimización de corte no se realiza sobre el umbral simétrico ingenuo ($\theta = 0.50$), sino mediante la minimización de la función de pérdida económica:

$$\min_{\theta \in [0, 1]} \mathcal{L}(\theta) = C_{FN} \cdot FN(\theta) + C_{FP} \cdot FP(\theta)$$

### Parámetros Financieros Calibrados:
- **$C_{FN}$ (Costo de Falso Negativo):** Restitución perentoria Ley 21.234 (promedio CLP $850.000) + Peritaje forense (CLP $50.000) + Provisión por riesgo de sanción CMF (CLP $100.000) $\approx \mathbf{\$1.000.000\text{ CLP}}$.
- **$C_{FP}$ (Costo de Falso Positivo):** Atención de contact center para desbloqueo (CLP $4.500) + Comisión transaccional perdida (CLP $2.500) + Penalización por probabilidad de fuga (*churn*) (CLP $18.000) $\approx \mathbf{\$25.000\text{ CLP}}$.
- **Ratio Asimétrico:** $40:1$.
- **Punto de Corte Calibrado:** $\theta^* \approx 0.195$, reduciendo en más de un 60% la pérdida esperada respecto a políticas estándar.

---

## 5. Torneo de Modelos y Rendimiento Multicapa

| Modelo | Tipo de Enfoque | F1-Score | AUC-ROC | AUC-PR | Costo Financiero Estimado |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **XGBoost Classifier (Purged CV)** | Supervisado Gradiente Boosting | **0.912** | **0.985** | **0.942** | **Mínimo ($)** |
| **Deep Dense Autoencoder** | No Supervisado / Reconstrucción | 0.841 | 0.923 | 0.856 | Moderado |
| **LSTM Sequence Autoencoder** | Secuencial Temporal | 0.865 | 0.941 | 0.880 | Moderado |
| **Isolation Forest Baseline** | Particionamiento Estocástico | 0.782 | 0.891 | 0.793 | Alto |
| **GAN Anomaly Detector** | Adversarial Out-of-Distribution | 0.820 | 0.910 | 0.835 | Moderado |

---

## 6. Módulos Estratégicos v2.0 Enterprise

### 6.1. Teoría de Grafos y Detección de Cuentas Mula (NetworkX & PySpark)
- Construcción de dígrafo ponderado $G = (V, E)$ donde los nodos son cuentas bancarias y las aristas son transferencias monetarias.
- Extracción de **In-Degree**, **Out-Degree**, **PageRank** y detección de **Carruseles de Fraude** (ciclos dirigidos).
- Regla de Cuentas Mula: $InDegree \ge 5$ con $OutDegree \le 2$ (concentración rápida de fondos).
- En Databricks, pipeline distribuido con PySpark SQL para procesamiento histórico nocturno y sincronización a Redis.

### 6.2. Motor Híbrido de Reglas Bancarias (Decision Engine)
- **Reglas Duras Inmediatas:** Listas negras judiciales CMF, países de alto riesgo sancionados (PRK, IRN, SYR) y operaciones nocturnas de alto monto (> CLP $5M) en cuentas con menos de 3 días de apertura.
- **Matriz de Decisión Operacional:**
  - $\text{Score} > 0.70 \to$ `BLOCK_PREVENTIVE` (Bloqueo y alerta a cumplimiento).
  - $0.20 \le \text{Score} \le 0.70 \to$ `CHALLENGE_STEPUP` (Exige MFA / Biometría FaceID).
  - $\text{Score} < 0.20 \to$ `APPROVE` (Aprobación fluida sin fricción).

### 6.3. Explicabilidad Regulatoria con TreeSHAP (Ley 21.234)
Descomposición analítica aditiva local para cada inferencia. Proporciona las top-3 causales exactas de la sospecha (monto desproporcionado, hora atípica, distancia al domicilio, velocidad transaccional), garantizando que todo rechazo sea legalmente defendible ante la CMF.

### 6.4. Agente GenAI RAG de Cumplimiento (Tipologías UAF)
- Base vectorial en memoria indexada con las guías oficiales de tipologías de lavado de activos de la UAF (Tipología N° 3 Triangulación/Mulas, Tipología N° 7 Smurfing, Tipología N° 12 Inconsistencia patrimonial).
- Redacción automatizada del borrador de pre-informe ROS con fundamentación jurídica.
- Fallback determinista seguro de alta fidelidad para contingencias operativas.

---

## 7. Plan de Monitoreo Continuo y Estabilidad Poblacional (PSI)

Para prevenir el deterioro del modelo ante cambios macroeconómicos o estacionales (*Data Drift*), se calcula semanalmente el **Population Stability Index (PSI)** sobre las 18 variables:

$$PSI = \sum_{i=1}^{K} \left( P_i - Q_i \right) \ln\left(\frac{P_i}{Q_i}\right)$$

- **$PSI < 0.10$:** Distribución estable. No requiere intervención.
- **$0.10 \le PSI \le 0.25$:** Alerta preventiva de cambio de comportamiento transaccional. Calibración de umbrales.
- **$PSI > 0.25$:** Gatillador mandatario de reentrenamiento completo del pipeline.

---

## 8. Arquitectura y SLA Operativo

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
                                     (Generación RAG UAF para Oficial de Cumplimiento)
```
