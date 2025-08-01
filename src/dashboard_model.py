import pandas as pd
from mlflow.tracking import MlflowClient
import mlflow.sklearn
import dash
from dash import dcc, html
import plotly.express as px

# Configuración
EXPERIMENT_NAME = "hotel_cancellation_prediction_tuning"
PROCESSED_DATA_PATH = "datos/procesado/hotel_bookings_processed.csv"

# Conectar a MLflow
client = MlflowClient()
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    raise ValueError(f"Experiment '{EXPERIMENT_NAME}' not found.")
exp_id = experiment.experiment_id

# Obtener runs ordenados por ROC AUC (descendente)
runs = client.search_runs(
    experiment_ids=[exp_id],
    order_by=["metrics.roc_auc DESC"]
)

# Construir DataFrame de métricas y parámetros
data = []
for run in runs:
    m = run.data.metrics
    p = run.data.params
    # Unificar posibles keys
    roc = m.get("roc_auc", m.get("roc_auc_score", None))
    lr  = p.get("learning_rate", None)
    md  = p.get("max_depth", None)
    ne  = p.get("n_estimators", None)
    data.append({
        "run_id": run.info.run_id,
        "roc_auc": roc,
        "accuracy": m.get("accuracy"),
        "precision": m.get("precision"),
        "recall": m.get("recall"),
        "f1_score": m.get("f1_score"),
        # Convertir parámetros a numérico (float)
        "learning_rate": pd.to_numeric(lr, errors="coerce"),
        "max_depth":    pd.to_numeric(md, errors="coerce"),
        "n_estimators": pd.to_numeric(ne, errors="coerce"),
    })
df_runs = pd.DataFrame(data)

# Asegurar métricas y parámetros numéricos y filtrar runs inválidos
for col in ['roc_auc', 'learning_rate', 'n_estimators']:
    df_runs[col] = pd.to_numeric(df_runs[col], errors='coerce')
df_runs = df_runs.dropna(subset=['roc_auc','learning_rate','n_estimators'])

# Verificar que haya al menos un run válido
if df_runs.empty:
    raise ValueError("No se encontraron runs con métricas válidas para 'roc_auc'.")

# Obtener el run con mayor ROC AUC
best_run_id = df_runs.loc[df_runs['roc_auc'].idxmax(), 'run_id']
model = mlflow.sklearn.load_model(f"runs:/{best_run_id}/best_model")

# Cargar datos procesados para importancias
df_processed = pd.read_csv(PROCESSED_DATA_PATH)
features = df_processed.drop(columns=['is_canceled']).columns
def get_feature_importances(model, features, top_n=20):
    importances = model.feature_importances_
    feat_df = pd.DataFrame({
        'feature': features,
        'importance': importances
    })
    return feat_df.sort_values('importance', ascending=False).head(top_n)
feat_df = get_feature_importances(model, features)

# Construir y levantar Dash app
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("Dashboard de Modelado de Cancelaciones"),

    html.Div([
        html.H3("ROC AUC por Run"),
        dcc.Graph(
            figure=px.bar(
                df_runs,
                x='run_id',
                y='roc_auc',
                labels={'run_id':'Run ID','roc_auc':'ROC AUC'},
                title='ROC AUC por Run'
            )
        )
    ], style={'margin-bottom':'40px'}),

    html.Div([
        html.H3("ROC AUC vs Learning Rate"),
        dcc.Graph(
            figure=px.scatter(
                df_runs,
                x='learning_rate',
                y='roc_auc',
                size='n_estimators',
                color='max_depth',
                labels={'learning_rate':'Learning Rate','roc_auc':'ROC AUC','max_depth':'Max Depth'},
                title='Parámetros vs ROC AUC'
            )
        )
    ], style={'margin-bottom':'40px'}),

    html.Div([
        html.H3("Top 20 Importancias de Features"),
        dcc.Graph(
            figure=px.bar(
                feat_df,
                x='importance',
                y='feature',
                orientation='h',
                labels={'importance':'Importancia','feature':'Feature'},
                title='Importancia de Features'
            )
        )
    ])
])

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
