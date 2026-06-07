import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Configuración de página
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_data():
    """Carga los datos procesados y resultados del torneo."""
    data = {}

    processed_path = "./data/processed/transactions_engineered.csv"
    if os.path.exists(processed_path):
        data['transactions'] = pd.read_csv(processed_path)

    results_dir = "./src/evaluation/results/"
    if os.path.exists(os.path.join(results_dir, "y_test.npy")):
        data['y_test'] = np.load(os.path.join(results_dir, "y_test.npy"))
        data['ae_deep_scores'] = np.load(os.path.join(results_dir, "ae_deep_scores.npy"))
        data['ae_deep_preds'] = np.load(os.path.join(results_dir, "ae_deep_predictions.npy"))
        data['ae_lstm_scores'] = np.load(os.path.join(results_dir, "ae_lstm_scores.npy"))
        data['ae_lstm_preds'] = np.load(os.path.join(results_dir, "ae_lstm_predictions.npy"))
        data['gan_scores'] = np.load(os.path.join(results_dir, "gan_scores.npy"))
        data['gan_preds'] = np.load(os.path.join(results_dir, "gan_predictions.npy"))
        data['xgb_probs'] = np.load(os.path.join(results_dir, "xgb_probs.npy"))
        data['xgb_preds'] = np.load(os.path.join(results_dir, "xgb_predictions.npy"))
        data['iso_scores'] = np.load(os.path.join(results_dir, "iso_scores.npy"))
        data['iso_preds'] = np.load(os.path.join(results_dir, "iso_predictions.npy"))

    tournament_path = os.path.join(results_dir, "fraud_tournament.csv")
    if os.path.exists(tournament_path):
        data['tournament'] = pd.read_csv(tournament_path)

    importances_path = os.path.join(results_dir, "feature_importances.csv")
    if os.path.exists(importances_path):
        data['importances'] = pd.read_csv(importances_path)

    return data


def page_overview(data: dict):
    """Página 1: Visión General del Dataset."""
    st.header("📊 Visión General del Dataset")

    if 'transactions' not in data:
        st.warning("⚠️ Datos no disponibles. Ejecuta `main_preprocessing.py` primero.")
        return

    df = data['transactions']

    # KPIs principales
    col1, col2, col3, col4 = st.columns(4)
    n_total = len(df)
    n_fraud = int(df['is_fraud'].sum())
    n_legit = n_total - n_fraud
    fraud_pct = n_fraud / n_total * 100

    col1.metric("Total Transacciones", f"{n_total:,}")
    col2.metric("Transacciones Legítimas", f"{n_legit:,}", f"{100-fraud_pct:.1f}%")
    col3.metric("Transacciones Fraudulentas", f"{n_fraud:,}", f"{fraud_pct:.1f}%")
    col4.metric("Clientes Únicos", f"{df['customer_id'].nunique():,}")

    st.divider()

    # Distribución de montos: Legítimo vs Fraude
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Distribución de Montos")
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df[df['is_fraud'] == 0]['transaction_amount'],
            name='Legítima', opacity=0.7,
            marker_color='#2ecc71', nbinsx=50
        ))
        fig.add_trace(go.Histogram(
            x=df[df['is_fraud'] == 1]['transaction_amount'],
            name='Fraude', opacity=0.7,
            marker_color='#e74c3c', nbinsx=50
        ))
        fig.update_layout(
            barmode='overlay', template='plotly_dark',
            xaxis_title='Monto ($)', yaxis_title='Frecuencia',
            height=400, legend=dict(x=0.7, y=0.95)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Distribución por Hora del Día")
        hour_fraud = df.groupby(['hour_of_day', 'is_fraud']).size().reset_index(name='count')
        fig = px.bar(
            hour_fraud, x='hour_of_day', y='count', color='is_fraud',
            color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
            labels={'is_fraud': 'Tipo', 'hour_of_day': 'Hora', 'count': 'Transacciones'},
            template='plotly_dark', barmode='group',
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Distancia vs Monto (scatter)
    st.subheader("Distancia desde el Domicilio vs Monto de la Transacción")
    sample = df.sample(min(5000, len(df)), random_state=42)
    fig = px.scatter(
        sample, x='distance_from_home', y='transaction_amount',
        color='is_fraud', color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
        opacity=0.5, template='plotly_dark',
        labels={'distance_from_home': 'Distancia (km)', 'transaction_amount': 'Monto ($)',
                'is_fraud': 'Tipo'},
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)


def page_model_results(data: dict):
    """Página 2: Resultados del Torneo de Modelos."""
    st.header("🏆 Torneo de Modelos")

    if 'tournament' not in data:
        st.warning("⚠️ Resultados no disponibles. Ejecuta `main_training.py` y `main_evaluation.py`.")
        return

    tournament = data['tournament']

    # Tabla del torneo
    st.subheader("Tabla Comparativa")
    st.dataframe(
        tournament.style.highlight_max(
            subset=['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc'],
            color='#27ae60'
        ).highlight_min(subset=['total_cost'], color='#27ae60'),
        use_container_width=True
    )

    st.divider()

    # Gráfico de barras comparativo
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Métricas de Clasificación")
        metrics_melt = tournament.melt(
            id_vars='model',
            value_vars=['accuracy', 'precision', 'recall', 'f1_score'],
            var_name='Métrica', value_name='Valor'
        )
        fig = px.bar(
            metrics_melt, x='Métrica', y='Valor', color='model',
            barmode='group', template='plotly_dark',
            color_discrete_sequence=['#3498db', '#e74c3c', '#f39c12'],
        )
        fig.update_layout(height=400, yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Costo Financiero Estimado")
        fig = px.bar(
            tournament, x='model', y='total_cost',
            template='plotly_dark',
            color='model',
            color_discrete_sequence=['#3498db', '#e74c3c', '#f39c12'],
            labels={'total_cost': 'Costo ($)', 'model': 'Modelo'},
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Feature Importances (XGBoost)
    if 'importances' in data:
        st.divider()
        st.subheader("📊 Importancia de Variables (XGBoost)")
        imp = data['importances'].head(15)
        fig = px.bar(
            imp, x='importance', y='feature', orientation='h',
            template='plotly_dark', color='importance',
            color_continuous_scale='Viridis',
        )
        fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)


def page_anomaly_analysis(data: dict):
    """Página 3: Análisis de Anomalías en Detalle."""
    st.header("🔍 Análisis de Anomalías")

    if 'y_test' not in data:
        st.warning("⚠️ Datos de test no disponibles.")
        return

    y_test = data['y_test']

    # Selector de modelo
    model_choice = st.selectbox(
        "Selecciona el modelo a analizar:",
        ['LSTM Autoencoder', 'GAN Anomaly Detector', 'Deep Autoencoder', 'XGBoost Classifier', 'Isolation Forest']
    )

    if model_choice == 'LSTM Autoencoder':
        scores = data['ae_lstm_scores']
        preds = data['ae_lstm_preds']
        score_label = 'LSTM Reconstruction Error (MSE)'
    elif model_choice == 'GAN Anomaly Detector':
        scores = data['gan_scores']
        preds = data['gan_preds']
        score_label = 'Discriminator Anomaly Score (1 - D(x))'
    elif model_choice == 'Deep Autoencoder':
        scores = data['ae_deep_scores']
        preds = data['ae_deep_preds']
        score_label = 'Reconstruction Error (MSE)'
    elif model_choice == 'XGBoost Classifier':
        scores = data['xgb_probs']
        preds = data['xgb_preds']
        score_label = 'Probabilidad de Fraude'
    else:
        scores = data['iso_scores']
        preds = data['iso_preds']
        score_label = 'Anomaly Score'

    col1, col2 = st.columns(2)

    with col1:
        # Distribución de scores
        st.subheader(f"Distribución de {score_label}")
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=scores[y_test == 0], name='Normal',
            opacity=0.7, marker_color='#2ecc71', nbinsx=80
        ))
        fig.add_trace(go.Histogram(
            x=scores[y_test == 1], name='Fraude',
            opacity=0.7, marker_color='#e74c3c', nbinsx=80
        ))
        fig.update_layout(
            barmode='overlay', template='plotly_dark',
            xaxis_title=score_label, yaxis_title='Frecuencia',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Confusion Matrix
        st.subheader("Matriz de Confusión")
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, preds)
        fig = px.imshow(
            cm, text_auto=True,
            labels=dict(x='Predicción', y='Real', color='Cantidad'),
            x=['Normal', 'Fraude'], y=['Normal', 'Fraude'],
            template='plotly_dark', color_continuous_scale='RdYlGn_r',
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Scatter de scores con threshold interactivo
    st.divider()
    st.subheader("Exploración Interactiva de Threshold")

    threshold = st.slider(
        f"Ajustar threshold de {score_label}:",
        float(scores.min()), float(scores.max()),
        float(np.percentile(scores, 95)),
        step=float((scores.max() - scores.min()) / 200)
    )

    custom_preds = (scores >= threshold).astype(int)
    from sklearn.metrics import precision_score, recall_score, f1_score
    prec = precision_score(y_test, custom_preds, zero_division=0)
    rec = recall_score(y_test, custom_preds, zero_division=0)
    f1 = f1_score(y_test, custom_preds, zero_division=0)

    c1, c2, c3 = st.columns(3)
    c1.metric("Precision", f"{prec:.4f}")
    c2.metric("Recall", f"{rec:.4f}")
    c3.metric("F1-Score", f"{f1:.4f}")


def page_data_explorer(data: dict):
    """Página 4: Explorador de Datos Interactivo."""
    st.header("🔎 Explorador de Datos")

    if 'transactions' not in data:
        st.warning("⚠️ Datos no disponibles.")
        return

    df = data['transactions']

    # Filtros
    col1, col2, col3 = st.columns(3)
    with col1:
        fraud_filter = st.selectbox("Tipo de transacción:", ['Todas', 'Solo Fraude', 'Solo Legítimas'])
    with col2:
        min_amount = st.number_input("Monto mínimo ($):", value=0.0)
    with col3:
        max_amount = st.number_input("Monto máximo ($):", value=float(df['transaction_amount'].max()))

    # Aplicar filtros
    filtered = df.copy()
    if fraud_filter == 'Solo Fraude':
        filtered = filtered[filtered['is_fraud'] == 1]
    elif fraud_filter == 'Solo Legítimas':
        filtered = filtered[filtered['is_fraud'] == 0]
    filtered = filtered[
        (filtered['transaction_amount'] >= min_amount) &
        (filtered['transaction_amount'] <= max_amount)
    ]

    st.write(f"Mostrando {len(filtered):,} de {len(df):,} transacciones")

    # Tabla interactiva
    st.dataframe(
        filtered.head(500).style.map(
            lambda v: 'color: #e74c3c' if v == 1 else '',
            subset=['is_fraud']
        ),
        use_container_width=True,
        height=500,
    )

    # Estadísticas descriptivas
    st.divider()
    st.subheader("Estadísticas Descriptivas")
    st.dataframe(filtered.describe().round(2), use_container_width=True)


# =============================================================
# MAIN APP
# =============================================================
def main():
    st.sidebar.title("🛡️ Fraud Detection")
    st.sidebar.markdown("**Sistema de Detección de Fraude**")
    st.sidebar.markdown("Basado en Autoencoders, XGBoost\ne Isolation Forest")
    st.sidebar.divider()

    page = st.sidebar.radio(
        "Navegación",
        ["📊 Visión General", "🏆 Torneo de Modelos",
         "🔍 Análisis de Anomalías", "🔎 Explorador de Datos"]
    )

    data = load_data()

    if page == "📊 Visión General":
        page_overview(data)
    elif page == "🏆 Torneo de Modelos":
        page_model_results(data)
    elif page == "🔍 Análisis de Anomalías":
        page_anomaly_analysis(data)
    elif page == "🔎 Explorador de Datos":
        page_data_explorer(data)

    st.sidebar.divider()
    st.sidebar.caption("Fraud Detection System v1.0")
    st.sidebar.caption("NVIDIA Anomaly Detection Certified")


if __name__ == "__main__":
    main()
