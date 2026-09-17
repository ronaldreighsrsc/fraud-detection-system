"""
Motor Híbrido de Reglas de Negocio Bancarias y Decisión Multicriterio.
Alineado con directrices de Riesgo Operacional CMF (Capítulo 20-10) y PLAFT (Ley 19.913).
"""
from typing import Dict, Any, Tuple, List, Optional, Set


class HybridRulesEngine:
    """
    Motor de Decisión Híbrido: Combina Reglas Duras CMF + Scoring ML + Penalizaciones de Red.
    Asegura que los mandatos legales y regulatorios prevalezcan sobre los modelos predictivos,
    reduciendo el riesgo de no-conformidad y multas normativas.
    """

    def __init__(self, high_risk_countries: Optional[List[str]] = None,
                 blacklisted_accounts: Optional[List[str]] = None):
        self.blacklisted_accounts: Set[str] = set(blacklisted_accounts or [])
        self.high_risk_countries: List[str] = high_risk_countries or ["PRK", "IRN", "SYR", "CUB"]

    def add_blacklisted_account(self, account_id: Any) -> None:
        """Registra una cuenta en la lista de bloqueo formal (oficios judiciales / UAF)."""
        self.blacklisted_accounts.add(str(account_id))

    def remove_blacklisted_account(self, account_id: Any) -> None:
        """Remueve una cuenta de la lista de bloqueo."""
        self.blacklisted_accounts.discard(str(account_id))

    def evaluate_hard_rules(self, tx: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Evalúa reglas duras deterministas de seguridad y cumplimiento legal.
        Si se activa alguna regla dura, la transacción se interrumpe de inmediato.
        
        Returns:
            (triggered: bool, reason: str)
        """
        orig_acc = str(tx.get('origin_account', tx.get('customer_id', '')))
        dest_acc = str(tx.get('destination_account', ''))
        dest_country = str(tx.get('destination_country', 'CHL')).upper()
        account_age = float(tx.get('account_age_days', 999))
        amount = float(tx.get('transaction_amount', 0.0))
        hour = int(tx.get('hour_of_day', tx.get('hour', 12)))

        # Regla Dura 1: Cuenta de origen en lista de bloqueo judicial
        if orig_acc and orig_acc in self.blacklisted_accounts:
            return True, "Regla CMF-01: Cuenta de origen en lista de bloqueo judicial / OFAC"

        # Regla Dura 2: Cuenta de destino en lista de bloqueo
        if dest_acc and dest_acc in self.blacklisted_accounts:
            return True, "Regla CMF-02: Cuenta de destino en lista negra internacional"

        # Regla Dura 3: Transferencia hacia jurisdicción sancionada (PLAFT)
        if dest_country in self.high_risk_countries:
            return True, f"Regla PLAFT-03: Transferencia hacia jurisdicción de alto riesgo ({dest_country})"

        # Regla Dura 4: Monto elevado nocturno en cuenta de reciente apertura
        if account_age < 3 and amount > 5_000_000:
            if hour in [0, 1, 2, 3, 4, 5]:
                return True, "Regla SEG-04: Transacción nocturna atípica de alto monto (> CLP $5M) en cuenta reciente (< 3 días)"

        return False, "Reglas Duras Superadas"

    def combine_verdict(self, ml_score: float, graph_features: Optional[Dict[str, float]] = None,
                        hard_rule_triggered: bool = False, rule_reason: str = "") -> Dict[str, Any]:
        """
        Matriz de decisión final combinando reglas duras, score ML y grafos.
        Implementa zonificación de riesgo:
        - CRITICAL: Bloqueo Inmediato (Regla dura)
        - HIGH: Bloqueo Preventivo (Score > 0.70)
        - MEDIUM: Desafío Step-Up MFA / Biometría (Score 0.20 - 0.70)
        - LOW: Aprobación Directa (Score < 0.20)
        """
        if hard_rule_triggered:
            return {
                "action": "BLOCK_IMMEDIATE",
                "final_risk_score": 1.0,
                "risk_level": "CRITICAL",
                "reason": rule_reason,
                "requires_mfa": False,
                "notify_compliance": True
            }

        graph_features = graph_features or {}
        risk_score = float(ml_score)

        # Penalización por topología mula en el grafo (+30% de riesgo)
        is_mule = float(graph_features.get("is_mule_candidate", 0.0)) == 1.0
        applied_penalties = []
        if is_mule:
            risk_score = min(1.0, risk_score + 0.30)
            applied_penalties.append("Penalización PLAFT: Cuenta identificada como nodo concentrador mula (+0.30)")

        # Matriz Operacional Bancaria
        if risk_score > 0.70:
            action = "BLOCK_PREVENTIVE"
            risk_level = "HIGH"
            requires_mfa = False
            notify_compliance = True
            reason = "Score predictivo de alto riesgo supera umbral crítico (theta > 0.70)"
        elif risk_score >= 0.20:
            action = "CHALLENGE_STEPUP"
            risk_level = "MEDIUM"
            requires_mfa = True  # Pide FaceID / Biometría / Clave Dinámica
            notify_compliance = False
            reason = "Riesgo moderado: Requiere verificación biométrica reforzada (MFA)"
        else:
            action = "APPROVE"
            risk_level = "LOW"
            requires_mfa = False
            notify_compliance = False
            reason = "Transacción dentro de los parámetros habituales de bajo riesgo"

        if applied_penalties:
            reason = f"{reason} | " + " | ".join(applied_penalties)

        return {
            "action": action,
            "final_risk_score": round(risk_score, 4),
            "risk_level": risk_level,
            "requires_mfa": requires_mfa,
            "notify_compliance": notify_compliance,
            "reason": reason
        }
