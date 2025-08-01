import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
from datos_prep import prepare_data, split_feature_target

# Configuración de MLflow para tuning
def setup_mlflow(experiment_name: str):
    mlflow.set_experiment(experiment_name + "_tuning")
    mlflow.sklearn.autolog(log_model_signatures=False)


def tune_and_log(
    raw_data_path: str,
    processed_data_path: str,
    experiment_name: str = "hotel_cancellation_prediction"
):
    # 1. Preparar datos
    df = prepare_data(raw_data_path, processed_data_path)
    X, y = split_feature_target(df)

    # 2. División train-test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Configurar experimento y autologging
    setup_mlflow(experiment_name)

    # 4. Búsqueda de hiperparámetros dentro del run
    with mlflow.start_run():
        param_dist = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 4, 5],
            'subsample': [0.6, 0.8, 1.0],
            'min_samples_split': [2, 5, 10]
        }
        base_model = GradientBoostingClassifier(random_state=42)
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_dist,
            n_iter=20,
            scoring='roc_auc',
            cv=3,
            random_state=42,
            n_jobs=-1
        )
        X_train = X_train.astype('float64')
        X_test  = X_test.astype('float64')
        search.fit(X_train, y_train)
        best = search.best_estimator_

        # 5. Calcular métricas dentro del run
        y_pred = best.predict(X_test)
        y_prob = best.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='binary'
        )

        # 6. Registrar métricas y parámetros en MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_params(search.best_params_)
        mlflow.sklearn.log_model(best, "best_model")

        # Mostrar resultados en consola
        print(f"Best params: {search.best_params_}")
        print(f"AUC: {auc:.4f}, Accuracy: {acc:.4f}")


if __name__ == "__main__":
    RAW_PATH = "datos/raw/hotel_bookings.csv"
    PROCESSED_PATH = "datos/procesado/hotel_bookings_processed.csv"
    tune_and_log(RAW_PATH, PROCESSED_PATH)
