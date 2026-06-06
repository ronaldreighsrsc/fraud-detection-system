import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd


def create_roc_curve(y_true: np.ndarray, scores_dict: dict) -> go.Figure:
    """
    Genera la curva ROC comparativa para múltiples modelos.
    
    Args:
        y_true: Etiquetas reales.
        scores_dict: Diccionario {nombre_modelo: scores}.
    """
    from sklearn.metrics import roc_curve, auc

    fig = go.Figure()
    colors = ['#3498db', '#e74c3c', '#f39c12']

    for i, (name, scores) in enumerate(scores_dict.items()):
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, name=f'{name} (AUC={roc_auc:.4f})',
            line=dict(color=colors[i % len(colors)], width=2),
        ))

    # Línea diagonal (random classifier)
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], name='Random',
        line=dict(color='gray', dash='dash', width=1),
        showlegend=True
    ))

    fig.update_layout(
        title='Curva ROC Comparativa',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        template='plotly_dark',
        height=500,
        legend=dict(x=0.55, y=0.05),
    )
    return fig


def create_precision_recall_curve(y_true: np.ndarray, scores_dict: dict) -> go.Figure:
    """Genera la curva Precision-Recall comparativa."""
    from sklearn.metrics import precision_recall_curve, average_precision_score

    fig = go.Figure()
    colors = ['#3498db', '#e74c3c', '#f39c12']

    for i, (name, scores) in enumerate(scores_dict.items()):
        precision, recall, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)
        fig.add_trace(go.Scatter(
            x=recall, y=precision, name=f'{name} (AP={ap:.4f})',
            line=dict(color=colors[i % len(colors)], width=2),
        ))

    fig.update_layout(
        title='Curva Precision-Recall Comparativa',
        xaxis_title='Recall',
        yaxis_title='Precision',
        template='plotly_dark',
        height=500,
    )
    return fig


def create_score_distribution(y_true: np.ndarray, scores: np.ndarray,
                               title: str = "Distribución de Scores") -> go.Figure:
    """Histograma de scores separado por clase real."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=scores[y_true == 0], name='Normal',
        opacity=0.7, marker_color='#2ecc71', nbinsx=100,
    ))
    fig.add_trace(go.Histogram(
        x=scores[y_true == 1], name='Fraude',
        opacity=0.7, marker_color='#e74c3c', nbinsx=100,
    ))
    fig.update_layout(
        title=title, barmode='overlay',
        template='plotly_dark', height=400,
        xaxis_title='Score', yaxis_title='Frecuencia',
    )
    return fig


def create_confusion_matrix_heatmap(y_true: np.ndarray, y_pred: np.ndarray,
                                     model_name: str) -> go.Figure:
    """Genera un heatmap de la Confusion Matrix."""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)

    fig = px.imshow(
        cm, text_auto=True,
        labels=dict(x='Predicción', y='Real', color='Cantidad'),
        x=['Normal', 'Fraude'], y=['Normal', 'Fraude'],
        template='plotly_dark', color_continuous_scale='RdYlGn_r',
        title=f'Confusion Matrix — {model_name}',
    )
    fig.update_layout(height=400)
    return fig
