import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import warnings

warnings.filterwarnings("ignore")


class FraudXGBoostDetector:
    """
    Motor de detección de fraude basado en XGBoost.
    Implementa Purged K-Fold Cross-Validation con Embargo (López de Prado)
    para evitar data leakage temporal, y manejo de desbalanceo de clases
    mediante scale_pos_weight.

    Adaptado del pipeline corporativo del IPSA Hybrid Forecasting Model.
    """

    def __init__(self, n_splits: int = 5, purge_size: int = 50,
                 embargo_size: int = 10, random_state: int = 42):
        self.n_splits = n_splits
        self.purge_size = purge_size
        self.embargo_size = embargo_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None
        self.best_params = None

    def _get_purged_embargoed_folds(self, num_samples: int) -> list:
        """Aplica la teoría de López de Prado para evitar Data Leakage temporal."""
        fold_size = num_samples // self.n_splits
        folds = []
        for i in range(self.n_splits):
            val_start = i * fold_size
            val_end = val_start + fold_size if i < self.n_splits - 1 else num_samples
            train_indices = []
            for j in range(num_samples):
                if j < val_start - self.purge_size:
                    train_indices.append(j)
                elif j >= val_end + self.embargo_size:
                    train_indices.append(j)
            val_indices = list(range(val_start, val_end))
            folds.append((np.array(train_indices), np.array(val_indices)))
        return folds

    def _calculate_scale_pos_weight(self, y: np.ndarray) -> float:
        """Calcula el peso para compensar desbalanceo: n_negative / n_positive."""
        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == 0)
        return n_neg / max(n_pos, 1)

    def find_best_params(self, X_train: np.ndarray, y_train: np.ndarray) -> dict:
        """
        Búsqueda de hiperparámetros usando Purged & Embargoed CV.
        Optimiza F1-Score en lugar de accuracy debido al desbalanceo.
        """
        print("  🔍 Buscando hiperparámetros (Purged & Embargoed CV para XGBoost)...")
        folds = self._get_purged_embargoed_folds(len(X_train))
        spw = self._calculate_scale_pos_weight(y_train)

        param_grid = [
            {'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.05, 'subsample': 0.8},
            {'n_estimators': 300, 'max_depth': 5, 'learning_rate': 0.03, 'subsample': 0.8},
            {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.05, 'subsample': 0.7},
            {'n_estimators': 400, 'max_depth': 4, 'learning_rate': 0.01, 'subsample': 0.9},
        ]

        best_f1 = -1
        best_params = param_grid[0]

        for params in param_grid:
            fold_f1s = []
            for train_idx, val_idx in folds:
                if len(train_idx) == 0:
                    continue

                model = xgb.XGBClassifier(
                    **params,
                    scale_pos_weight=spw,
                    random_state=self.random_state,
                    n_jobs=-1,
                    eval_metric='logloss',
                    use_label_encoder=False,
                )
                model.fit(X_train[train_idx], y_train[train_idx])
                preds = model.predict(X_train[val_idx])
                fold_f1s.append(f1_score(y_train[val_idx], preds, zero_division=0))

            avg_f1 = np.mean(fold_f1s)
            if avg_f1 > best_f1:
                best_f1 = avg_f1
                best_params = params

        print(f"  ✅ Ganador: {best_params} (F1 Interno: {best_f1:.2%})")
        self.best_params = best_params
        return best_params

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              params: dict = None) -> None:
        """Entrena el modelo final con los mejores hiperparámetros."""
        print("  🚀 Entrenando modelo XGBoost final...")

        if params is None:
            params = self.best_params or {
                'n_estimators': 300, 'max_depth': 5,
                'learning_rate': 0.03, 'subsample': 0.8
            }

        spw = self._calculate_scale_pos_weight(y_train)

        # Escalamiento y clipping (vital para estabilizar gradiente de XGBoost)
        X_scaled = np.clip(self.scaler.fit_transform(X_train), -10, 10)

        self.model = xgb.XGBClassifier(
            **params,
            scale_pos_weight=spw,
            random_state=self.random_state,
            n_jobs=-1,
            eval_metric='logloss',
            use_label_encoder=False,
        )
        self.model.fit(X_scaled, y_train)

        # Importancia de variables
        self.feature_importances_ = self.model.feature_importances_
        print(f"  ✅ Modelo entrenado con {params['n_estimators']} estimadores")

    def predict(self, X: np.ndarray) -> tuple:
        """
        Predice fraude y retorna probabilidades.
        Retorna: (predicciones binarias, probabilidades de la clase positiva).
        """
        X_scaled = np.clip(self.scaler.transform(X), -10, 10)
        pred_probs = self.model.predict_proba(X_scaled)[:, 1]
        predictions = self.model.predict(X_scaled)
        return predictions, pred_probs

    def get_feature_importances(self, feature_names: list) -> pd.DataFrame:
        """Retorna las importancias de variables ordenadas descendientemente."""
        if self.feature_importances_ is None:
            raise ValueError("El modelo no ha sido entrenado aún.")

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importances_
        }).sort_values('importance', ascending=False).reset_index(drop=True)

        return importance_df
