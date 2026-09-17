import pytest
from src.rules.rules_engine import HybridRulesEngine


@pytest.fixture
def rules_engine():
    engine = HybridRulesEngine(high_risk_countries=["PRK", "IRN", "SYR"])
    engine.add_blacklisted_account("BLOCKED_999")
    return engine


def test_hard_rule_blacklisted_origin(rules_engine):
    tx = {
        "origin_account": "BLOCKED_999",
        "destination_account": "NORMAL_ACC",
        "destination_country": "CHL",
        "transaction_amount": 50000.0,
        "hour_of_day": 14,
        "account_age_days": 100
    }
    triggered, reason = rules_engine.evaluate_hard_rules(tx)
    assert triggered is True
    assert "CMF-01" in reason

    verdict = rules_engine.combine_verdict(ml_score=0.05, hard_rule_triggered=triggered, rule_reason=reason)
    assert verdict["action"] == "BLOCK_IMMEDIATE"
    assert verdict["risk_level"] == "CRITICAL"
    assert verdict["notify_compliance"] is True


def test_hard_rule_blacklisted_destination(rules_engine):
    tx = {
        "origin_account": "NORMAL_ACC",
        "destination_account": "BLOCKED_999",
        "destination_country": "CHL",
        "transaction_amount": 10000.0,
        "hour_of_day": 11,
        "account_age_days": 50
    }
    triggered, reason = rules_engine.evaluate_hard_rules(tx)
    assert triggered is True
    assert "CMF-02" in reason


def test_hard_rule_sanctioned_country(rules_engine):
    tx = {
        "origin_account": "ACC_1",
        "destination_account": "ACC_2",
        "destination_country": "PRK",
        "transaction_amount": 25000.0,
        "hour_of_day": 10,
        "account_age_days": 200
    }
    triggered, reason = rules_engine.evaluate_hard_rules(tx)
    assert triggered is True
    assert "PLAFT-03" in reason


def test_hard_rule_nocturnal_high_amount_new_account(rules_engine):
    tx = {
        "origin_account": "ACC_NEW",
        "destination_account": "ACC_DEST",
        "destination_country": "CHL",
        "transaction_amount": 6_000_000.0,  # > 5M CLP
        "hour_of_day": 3,                   # Madrugada (3 AM)
        "account_age_days": 1               # < 3 días
    }
    triggered, reason = rules_engine.evaluate_hard_rules(tx)
    assert triggered is True
    assert "SEG-04" in reason


def test_decision_matrix_zones(rules_engine):
    # 1. Zona de bajo riesgo (< 0.20)
    v_low = rules_engine.combine_verdict(ml_score=0.10)
    assert v_low["action"] == "APPROVE"
    assert v_low["risk_level"] == "LOW"
    assert v_low["requires_mfa"] is False

    # 2. Zona de desafío biométrico (0.20 - 0.70)
    v_med = rules_engine.combine_verdict(ml_score=0.45)
    assert v_med["action"] == "CHALLENGE_STEPUP"
    assert v_med["risk_level"] == "MEDIUM"
    assert v_med["requires_mfa"] is True

    # 3. Zona de bloqueo preventivo (> 0.70)
    v_high = rules_engine.combine_verdict(ml_score=0.75)
    assert v_high["action"] == "BLOCK_PREVENTIVE"
    assert v_high["risk_level"] == "HIGH"
    assert v_high["requires_mfa"] is False
    assert v_high["notify_compliance"] is True


def test_mule_penalty_escalation(rules_engine):
    # Score base de 0.15 normalmente sería APPROVE
    graph_features = {"is_mule_candidate": 1.0}
    verdict = rules_engine.combine_verdict(ml_score=0.15, graph_features=graph_features)
    # Con penalización de +0.30 pasa a 0.45 -> CHALLENGE_STEPUP
    assert verdict["final_risk_score"] == 0.45
    assert verdict["action"] == "CHALLENGE_STEPUP"
    assert verdict["requires_mfa"] is True
