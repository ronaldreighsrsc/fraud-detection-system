import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


class TransactionSynthesizer:
    """
    Generador de datos sintéticos de transacciones financieras (v2.0 Enterprise).
    Produce un dataset realista con patrones de fraude inyectados,
    diseñado para entrenar y evaluar modelos de detección de anomalías,
    teoría de grafos (PLAFT) y motores de reglas bancarias.

    Patrones de fraude simulados:
        - Velocity Attacks: múltiples transacciones en ventana temporal corta.
        - Geo-Anomalies: transacciones desde ubicaciones inusuales o internacionales.
        - Amount Spikes: montos significativamente superiores al promedio del usuario.
        - Smurfing / Redes Mula: transacciones dirigidas hacia nodos concentradores (Pareto Law).
    """

    def __init__(self, n_customers: int = 1000, n_transactions: int = 100_000,
                 fraud_ratio: float = 0.02, random_state: int = 42):
        self.n_customers = n_customers
        self.n_transactions = n_transactions
        self.fraud_ratio = fraud_ratio
        self.rng = np.random.RandomState(random_state)

    # ---------------------------------------------------------------
    # Perfil base de cada cliente (hábitos normales)
    # ---------------------------------------------------------------
    def _generate_customer_profiles(self) -> pd.DataFrame:
        """Crea perfiles con hábitos de gasto habituales por cliente."""
        profiles = pd.DataFrame({
            'customer_id': range(self.n_customers),
            'account_age_days': self.rng.randint(10, 1500, size=self.n_customers),
            'avg_amount': self.rng.lognormal(mean=4.0, sigma=0.8, size=self.n_customers),
            'std_amount': self.rng.uniform(5, 50, size=self.n_customers),
            'home_lat': self.rng.uniform(-33.60, -33.35, size=self.n_customers),   # Santiago, Chile
            'home_lon': self.rng.uniform(-70.75, -70.55, size=self.n_customers),
            'preferred_hour_start': self.rng.randint(7, 12, size=self.n_customers),
            'preferred_hour_end': self.rng.randint(18, 23, size=self.n_customers),
        })
        return profiles

    # ---------------------------------------------------------------
    # Cuentas de destino con Distribución de Pareto y Mulas PLAFT
    # ---------------------------------------------------------------
    def _generate_destination_accounts(self, n_transactions: int, is_fraud_array: np.ndarray) -> np.ndarray:
        """
        Genera cuentas de destino siguiendo una distribución de Pareto (Ley de Potencias),
        asegurando la emergencia orgánica de 'hubs' y cuentas concentradoras mulas.
        """
        # 1. Base legítima: Distribución de Pareto (comercios/servicios concentran flujo)
        pareto_samples = (self.rng.pareto(a=1.8, size=n_transactions) * 10).astype(int)
        dest_accounts = pareto_samples % self.n_customers

        # 2. Inyección Determinista de Células de Lavado (PLAFT / Smurfing):
        # 5 cuentas específicas para actuar como 'Cuentas Mula' del sindicato criminal
        mule_accounts = [101, 202, 303, 404, 505]
        
        fraud_indices = np.where(is_fraud_array == 1)[0]
        for idx in fraud_indices:
            target_mule = self.rng.choice(mule_accounts)
            dest_accounts[idx] = target_mule

        return dest_accounts

    # ---------------------------------------------------------------
    # Transacciones legítimas
    # ---------------------------------------------------------------
    def _generate_legit_transactions(self, profiles: pd.DataFrame,
                                     n_legit: int) -> pd.DataFrame:
        """Genera transacciones normales basadas en el perfil de cada cliente."""
        customer_ids = self.rng.choice(profiles['customer_id'], size=n_legit)
        rows = []

        for cid in customer_ids:
            p = profiles.loc[profiles['customer_id'] == cid].iloc[0]
            amount = max(0.01, self.rng.normal(p['avg_amount'], p['std_amount']))
            hour = self.rng.randint(p['preferred_hour_start'], p['preferred_hour_end'] + 1)
            day = self.rng.randint(0, 7)

            # Ubicación cercana al domicilio (radio < 15 km)
            lat = p['home_lat'] + self.rng.normal(0, 0.02)
            lon = p['home_lon'] + self.rng.normal(0, 0.02)
            distance = np.sqrt((lat - p['home_lat'])**2 + (lon - p['home_lon'])**2) * 111  # km aprox

            rows.append({
                'customer_id': cid,
                'origin_account': cid,
                'account_age_days': int(p['account_age_days']),
                'destination_country': 'CHL',
                'transaction_amount': round(amount, 2),
                'hour_of_day': hour % 24,
                'day_of_week': day,
                'merchant_category': self.rng.randint(0, 15),
                'distance_from_home': round(distance, 2),
                'is_international': 0,
                'is_fraud': 0,
            })

        return pd.DataFrame(rows)

    # ---------------------------------------------------------------
    # Transacciones fraudulentas (patrones realistas)
    # ---------------------------------------------------------------
    def _generate_fraud_transactions(self, profiles: pd.DataFrame,
                                     n_fraud: int) -> pd.DataFrame:
        """
        Inyecta transacciones anómalas con tres vectores de ataque:
        1. Amount Spikes (40%): montos 5-20x superiores al promedio.
        2. Geo-Anomalies (35%): ubicaciones muy lejanas al domicilio.
        3. Velocity Attacks (25%): horarios atípicos + categorías inusuales.
        """
        customer_ids = self.rng.choice(profiles['customer_id'], size=n_fraud)
        rows = []

        for i, cid in enumerate(customer_ids):
            p = profiles.loc[profiles['customer_id'] == cid].iloc[0]
            attack_type = self.rng.choice(['amount', 'geo', 'velocity'],
                                          p=[0.40, 0.35, 0.25])
            dest_country = 'CHL'
            acc_age = int(p['account_age_days'])

            if attack_type == 'amount':
                # Monto anormalmente alto (5x a 20x del promedio)
                multiplier = self.rng.uniform(5, 20)
                amount = round(p['avg_amount'] * multiplier, 2)
                hour = self.rng.randint(0, 24)
                distance = self.rng.uniform(0, 5)
                is_international = 0
                if self.rng.random() < 0.05:
                    acc_age = self.rng.choice([1, 2])

            elif attack_type == 'geo':
                # Transacción desde ubicación lejana o internacional
                amount = max(0.01, self.rng.normal(p['avg_amount'] * 1.5, p['std_amount']))
                hour = self.rng.randint(0, 24)
                distance = self.rng.uniform(50, 500)  # 50-500 km del domicilio
                is_international = int(self.rng.random() > 0.5)
                if is_international:
                    dest_country = self.rng.choice(['USA', 'ARG', 'COL', 'BRA', 'PRK'])

            else:  # velocity
                # Horario atípico (madrugada) + categoría inusual
                amount = max(0.01, self.rng.normal(p['avg_amount'] * 2, p['std_amount']))
                hour = self.rng.choice([0, 1, 2, 3, 4, 5])  # Madrugada
                distance = self.rng.uniform(10, 50)
                is_international = 0
                if self.rng.random() < 0.10:
                    acc_age = 2
                    amount = round(amount + 5_500_000, 2)

            rows.append({
                'customer_id': cid,
                'origin_account': cid,
                'account_age_days': acc_age,
                'destination_country': dest_country,
                'transaction_amount': round(amount, 2),
                'hour_of_day': hour,
                'day_of_week': self.rng.randint(0, 7),
                'merchant_category': self.rng.randint(0, 15),
                'distance_from_home': round(distance, 2),
                'is_international': is_international,
                'is_fraud': 1,
            })

        return pd.DataFrame(rows)

    # ---------------------------------------------------------------
    # Orquestador público
    # ---------------------------------------------------------------
    def generate(self, output_path: str = "./data/raw/transactions.csv") -> pd.DataFrame:
        """
        Pipeline principal de generación de datos sintéticos.
        Retorna un DataFrame con transacciones mezcladas (legítimas + fraude),
        cuentas origen y destino, y atributos para análisis de grafos PLAFT.
        """
        print("🏭 Generando datos sintéticos de transacciones (v2.0 Enterprise)...")

        n_fraud = int(self.n_transactions * self.fraud_ratio)
        n_legit = self.n_transactions - n_fraud

        profiles = self._generate_customer_profiles()
        print(f"  👤 Perfiles de {self.n_customers} clientes generados (Santiago, Chile)")

        df_legit = self._generate_legit_transactions(profiles, n_legit)
        print(f"  ✅ {n_legit:,} transacciones legítimas generadas")

        df_fraud = self._generate_fraud_transactions(profiles, n_fraud)
        print(f"  🚨 {n_fraud:,} transacciones fraudulentas inyectadas ({self.fraud_ratio:.1%})")

        # Mezclar y resetear índice
        df = pd.concat([df_legit, df_fraud], ignore_index=True)
        df = df.sample(frac=1, random_state=self.rng).reset_index(drop=True)

        # Generar cuentas de destino con Ley de Pareto y Células Mula
        df['destination_account'] = self._generate_destination_accounts(len(df), df['is_fraud'].values)
        df['tx_id'] = [f"TX_{i:07d}" for i in range(len(df))]

        # Agregar timestamp sintético (últimos 90 días)
        base_date = pd.Timestamp('2026-03-01')
        df['timestamp'] = [base_date + pd.Timedelta(hours=int(h))
                           for h in self.rng.uniform(0, 90 * 24, size=len(df))]
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Guardar en disco
        df.to_csv(output_path, index=False)
        print(f"  💾 Dataset guardado en: {output_path} ({len(df):,} registros)")

        return df
