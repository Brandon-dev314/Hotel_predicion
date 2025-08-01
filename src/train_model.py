import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support
from datos_prep import prepare_data, split_feature_target


def setup_mlflow(experiment_name: str):
    mlflow.set_experiment(experiment_name)
    # Desactivamos logging de signature/schema para evitar warnings
    mlflow.sklearn.autolog(log_model_signatures=False)


def train_and_evaluate(
    raw_data_path: str,
    processed_data_path: str,
    experiment_name: str = "hotel_cancellation_prediction"
):
    # Preparar datos
    df = prepare_data(raw_data_path, processed_data_path)
    X, y = split_feature_target(df)

    # División train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Inicializar MLflow
    setup_mlflow(experiment_name)

    with mlflow.start_run():
        # Definir y entrenar modelo
        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        X_train = X_train.astype('float64')
        X_test  = X_test.astype('float64')
        model.fit(X_train, y_train)

        # Predicciones y probabilidades
        preds = model.predict(X_test)
        probas = model.predict_proba(X_test)[:, 1]

        # Cálculo de métricas
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probas)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, preds, average='binary'
        )

        # Registro manual de métricas (además de autolog)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Mostrar resultados
        print(f"Accuracy: {acc:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")


if __name__ == "__main__":
    RAW_PATH = "datos/raw/hotel_bookings.csv"
    PROCESSED_PATH = "datos/procesado/hotel_bookings_processed.csv"
    train_and_evaluate(RAW_PATH, PROCESSED_PATH)
