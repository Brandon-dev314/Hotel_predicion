import pandas as pd
import numpy as np

# Mapa de meses para conversión a numérico
MONTHS_MAP = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12
}


def load_data(path: str) -> pd.DataFrame:
    """
    Carga el CSV desde la ruta especificada.
    """
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpieza básica:
      - Elimina duplicados y trabaja sobre copia
      - Imputa valores faltantes en columnas clave
      - Elimina filas que faltan valores esenciales
      - Asegura que no haya NaNs antes del feature engineering
    """
    df_clean = df.drop_duplicates().copy()
    # Imputación de valores faltantes
    if 'children' in df_clean.columns:
        df_clean.loc[:, 'children'] = df_clean['children'].fillna(0)
    if 'country' in df_clean.columns:
        df_clean.loc[:, 'country'] = df_clean['country'].fillna('Unknown')
    # Columnas numéricas con NaNs (por ejemplo agent, company)
    for col in ['agent', 'company']:
        if col in df_clean.columns:
            df_clean.loc[:, col] = df_clean[col].fillna(0)
    # Eliminar filas con datos esenciales faltantes
    required = [
        'hotel', 'arrival_date_month', 'arrival_date_day_of_month',
        'arrival_date_year', 'lead_time', 'adr', 'is_canceled'
    ]
    df_clean = df_clean.dropna(subset=required)
    return df_clean


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creación de features temporales y transformación de columnas.
    """
    df_feat = df.copy()
    # Mapear meses y construir fecha
    df_feat.loc[:, 'arrival_date_month'] = df_feat['arrival_date_month'].map(MONTHS_MAP)
    df_feat.loc[:, 'arrival_date'] = pd.to_datetime(dict(
        year = df_feat['arrival_date_year'],
        month = df_feat['arrival_date_month'],
        day = df_feat['arrival_date_day_of_month']
    ))
    # Extraer componentes de fecha
    df_feat.loc[:, 'arrival_weekday'] = df_feat['arrival_date'].dt.dayofweek
    df_feat.loc[:, 'arrival_month']   = df_feat['arrival_date'].dt.month
    df_feat.loc[:, 'arrival_year']    = df_feat['arrival_date'].dt.year
    # Ejemplo: diferencia entre reserva y llegada
    df_feat.loc[:, 'booking_to_arrival_diff'] = df_feat['lead_time']
    # Eliminar columnas originales de fecha
    df_feat = df_feat.drop(
        columns=['arrival_date', 'arrival_date_year', 'arrival_date_day_of_month'],
        errors='ignore'
    )
    return df_feat


def encode_categoricals(
    df: pd.DataFrame,
    categorical_cols: list[str] | None = None
) -> pd.DataFrame:
    """
    Aplica one-hot encoding a variables categóricas.
    """
    df_enc = df.copy()
    if categorical_cols is None:
        categorical_cols = df_enc.select_dtypes(include=['object']).columns.tolist()
    return pd.get_dummies(df_enc, columns=categorical_cols, drop_first=True)


def prepare_data(raw_path: str, processed_path: str) -> pd.DataFrame:
    """
    1. Carga
    2. Limpieza
    3. Ingeniería de features
    4. ELIMINAR columnas de alta cardinalidad (fechas/IDs)
    5. One-hot encoding
    6. Fill NaNs + cast a float64
    """
    df = load_data(raw_path)
    df = clean_data(df)
    df = feature_engineering(df)

    # ——————————————
    # Eliminamos las columnas que generan cientos de dummies
    for col in ('reservation_status', 'reservation_status_date'):
        if col in df.columns:
            df = df.drop(columns=[col])

    # Codificamos categoricals “medias” y no las fechas
    df = encode_categoricals(df)

    # Rellenamos cualquier NaN faltante y forzamos float para MLflow
    df = df.fillna(0).astype('float64')
    df.to_csv(processed_path, index=False)
    return df

def split_feature_target(
    df: pd.DataFrame,
    target: str = 'is_canceled'
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separa features de la variable objetivo.
    """
    X = df.drop(columns=[target])
    y = df[target]
    return X, y
