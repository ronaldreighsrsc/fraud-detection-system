"""
Agente Generativo RAG de Cumplimiento y PLAFT (UAF Chile).
Recuperación semántica de Tipologías UAF con ChromaDB + Síntesis LLM vía LangChain
con Fallback Determinista Regulatorio (Ley N° 19.913).
"""
import os
import json
from typing import Dict, Any, List, Optional


class ComplianceROSAgent:
    """
    Asistente Inteligente de Cumplimiento Normativo (AML / PLAFT).
    Asiste a los peritos forenses de Banco Bci redactando el borrador oficial
    del Reporte de Operación Sospechosa (ROS) fundamentado en las tipologías oficiales de la UAF.
    """

    def __init__(self, model_name: str = "gpt-4o-mini", bank_name: str = "Banco Bci"):
        self.bank_name = bank_name
        self.model_name = model_name
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.vectorstore = None
        self.llm_available = bool(self.api_key)
        self.prompt_template = None
        self.llm = None

        # 1. Cargar Base Vectorial con Tipologías Oficiales UAF
        self._init_uaf_vectorstore()

        # 2. Configurar LLM si hay credenciales disponibles
        if self.llm_available:
            self._init_llm()

    def _init_uaf_vectorstore(self):
        """Inicializa la base vectorial en memoria con tipologías de la UAF."""
        try:
            from langchain_community.vectorstores import Chroma
            from langchain_openai import OpenAIEmbeddings
            from langchain_core.documents import Document

            uaf_docs = [
                Document(
                    page_content="Tipología UAF N° 3: Triangulación y Cuentas Puente. Uso de personas naturales para recibir fondos fraccionados de múltiples remitentes desconocidos y reenviarlos en menos de 24 horas a cuentas concentradoras o exchanges.",
                    metadata={"id": "UAF-03", "tipo": "mulas"}
                ),
                Document(
                    page_content="Tipología UAF N° 7: Fraccionamiento (Smurfing/Pitufeo). Depósitos o transferencias múltiples por montos justo por debajo del umbral de reporte (10.000 USD / 35 UF) para evadir fiscalización.",
                    metadata={"id": "UAF-07", "tipo": "smurfing"}
                ),
                Document(
                    page_content="Tipología UAF N° 12: Inconsistencia Patrimonial y Actividad Nocturna Inusual en Cuentas Nuevas sin justificación de giro comercial.",
                    metadata={"id": "UAF-12", "tipo": "inconsistencia"}
                )
            ]
            if self.api_key:
                self.vectorstore = Chroma.from_documents(uaf_docs, OpenAIEmbeddings(api_key=self.api_key))
        except Exception:
            self.vectorstore = None

    def _init_llm(self):
        """Configura el modelo generativo y template del ROS."""
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.prompts import ChatPromptTemplate

            self.llm = ChatOpenAI(model=self.model_name, temperature=0.1, api_key=self.api_key)
            self.prompt_template = ChatPromptTemplate.from_messages([
                ("system", """Eres el Oficial Forense de Cumplimiento Senior y Perito Analítico de {bank_name}.
Tu misión es redactar el borrador formal del Reporte de Operaciones Sospechosas (ROS) dirigido a la Unidad de Análisis Financiero (UAF) de Chile (Ley N° 19.913).
Debes fundamentar jurídicamente el caso asociándolo con la tipología oficial recuperada de la normativa UAF:
TIPOLOGÍA UAF RELEVANTE:
{uaf_typology_context}

Estructura el informe formal con:
1. Hechos fácticos e identificación de cuentas (Origen / Destino).
2. Justificación predictiva y explicabilidad SHAP.
3. Dictamen topológico de redes (Cuentas Mula / Smurfing).
4. Medida cautelar aplicada conforme al Capítulo 20-10 CMF."""),
                ("human", "Genera el informe formal para el siguiente expediente transaccional:\n{case_evidence_json}")
            ])
        except Exception:
            self.llm_available = False

    def generate_ros_narrative(self, tx_details: Dict[str, Any], top_risk_factors: List[Dict[str, Any]],
                                graph_insights: Optional[Dict[str, Any]] = None) -> str:
        """
        Genera el informe formal integrando RAG sobre tipologías UAF o fallback determinista.
        """
        graph_insights = graph_insights or {}
        evidence = {
            "entidad_reportante": self.bank_name,
            "transaccion": tx_details,
            "explicabilidad_shap": top_risk_factors,
            "analisis_grafos_plaft": graph_insights
        }

        # Flujo 1: RAG con LLM Real
        if self.llm_available and self.vectorstore and self.prompt_template and self.llm:
            try:
                query = f"cuentas mula {graph_insights.get('is_mule_candidate')} in_degree {graph_insights.get('in_degree')} fraccionamiento smurfing"
                matched_docs = self.vectorstore.similarity_search(query, k=1)
                typology_text = matched_docs[0].page_content if matched_docs else "Guía General de Alertas UAF Ley 19.913"

                formatted_prompt = self.prompt_template.format_messages(
                    bank_name=self.bank_name,
                    uaf_typology_context=typology_text,
                    case_evidence_json=json.dumps(evidence, indent=2, default=str)
                )
                response = self.llm.invoke(formatted_prompt)
                return str(response.content)
            except Exception as e:
                print(f"⚠️ Error en pipeline RAG ({e}). Conmutando a fallback regulatorio.")

        # Flujo 2: Fallback Seguro Determinista
        return self._deterministic_fallback_ros(tx_details, top_risk_factors, graph_insights)

    def _deterministic_fallback_ros(self, tx_details: Dict[str, Any], top_risk_factors: List[Dict[str, Any]],
                                    graph_insights: Dict[str, Any]) -> str:
        """Plantilla formal de respaldo de alta fidelidad regulatoria."""
        factors_lines = []
        for f in top_risk_factors:
            label = f.get('feature_label', f.get('feature', 'Variable'))
            shap_val = f.get('shap_impact', 0.0)
            act_val = f.get('actual_value', 0.0)
            factors_lines.append(f"  * {label}: Impacto SHAP +{shap_val:.4f} (Valor observado: {act_val:.2f})")

        factors_str = "\n".join(factors_lines) if factors_lines else "  * Sin factores atípicos destacados."

        is_mule = bool(graph_insights.get('is_mule_candidate', False))
        mule_status = "POSITIVO (ALTO RIESGO CONCENTRADOR)" if is_mule else "NEGATIVO"

        typology = "Tipología UAF N° 3 (Triangulación de Cuentas Puente / Mulas)" if is_mule else "Tipología UAF N° 7 (Fraccionamiento / Smurfing)"

        tx_id = tx_details.get('tx_id', 'TX-UNKNOWN')
        amount = float(tx_details.get('transaction_amount', 0.0))
        orig_acc = tx_details.get('origin_account', tx_details.get('customer_id', 'N/A'))
        dest_acc = tx_details.get('destination_account', 'N/A')
        risk_score = float(tx_details.get('risk_score', tx_details.get('final_risk_score', 0.85)))

        return f"""================================================================================
CONFIDENCIAL - PRE-INFORME DE OPERACIÓN SOSPECHOSA (ROS) [UAF CHILE]
DESTINATARIO: Unidad de Análisis Financiero (UAF) - Ley N° 19.913
ENTIDAD: {self.bank_name} | Gerencia de Riesgo Operacional, Ciberseguridad & PLAFT
FUNDAMENTO LEGAL: {typology}
================================================================================

1. INDIVIDUALIZACIÓN DEL EVENTO TRANSACCIONAL:
   - Identificador Transaccional (ID): {tx_id}
   - Monto de la Operación: CLP ${amount:,.0f}
   - Cuenta de Origen: {orig_acc}
   - Cuenta de Destino: {dest_acc}
   - Score Predictivo Multicapa: {risk_score:.2%}

2. FACTORES EXPLICATIVOS LOCALES (REGULACIÓN CMF / LEY 21.234):
{factors_str}

3. PERITAJE DE REDES COMPLEJAS Y TEORÍA DE GRAFOS:
   - Grado de Entrada (In-Degree): {graph_insights.get('in_degree', 0.0):.0f}
   - Grado de Salida (Out-Degree): {graph_insights.get('out_degree', 0.0):.0f}
   - Centralidad PageRank: {graph_insights.get('pagerank', 0.0):.6f}
   - Veredicto de Cuenta Concentradora (Mula): {mule_status}

4. DICTAMEN PREVENTIVO Y MEDIDAS CAUTELARES:
   Conforme al Capítulo 20-10 de la CMF y políticas internas de PLAFT, la transacción
   ha sido bloqueada preventivamente para salvaguardar el patrimonio del banco y los clientes.
   Se eleva el presente expediente al Oficial de Cumplimiento para su visado y posterior
   despacho a la UAF.
================================================================================
""".strip()
