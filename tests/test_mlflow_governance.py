import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from src.models.mlflow_governance import MLflowGovernanceManager


def test_mlflow_governance_lifecycle(tmp_path):
    tracking_db = tmp_path / "mlflow_test.db"
    manager = MLflowGovernanceManager(
        experiment_name="Test_Bci_Fraud_Risk",
        tracking_uri=f"sqlite:///{tracking_db}"
    )

    params = {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.05,
        "scale_pos_weight": 49.0
    }
    metrics = {
        "f1_score": 0.912,
        "auc_roc": 0.985,
        "auc_pr": 0.942,
        "optimal_threshold": 0.195
    }

    # Modelo sklearn real entrenado para evitar rechazo de tipos no seguros en skops/mlflow 3.x
    model = LogisticRegression()
    X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    y = np.array([0, 1, 0])
    model.fit(X, y)

    run_id = manager.log_model_run(
        model=model,
        params=params,
        metrics=metrics,
        model_name="Test_Bci_Model"
    )

    assert run_id is not None
    assert len(run_id) > 0

    # Test stage promotion
    promoted_staging = manager.promote_to_cmf_validation("Test_Bci_Model", version=1)
    assert promoted_staging is True

    promoted_prod = manager.promote_to_production("Test_Bci_Model", version=1)
    assert promoted_prod is True
