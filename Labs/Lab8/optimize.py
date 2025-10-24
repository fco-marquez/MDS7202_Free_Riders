import pickle
import mlflow
import mlflow.xgboost
import optuna
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from optuna.visualization import plot_optimization_history, plot_param_importances
from datetime import datetime
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def get_best_model(experiment_id):
    runs = mlflow.search_runs([experiment_id])
    best_model_id = runs.sort_values("metrics.valid_f1", ascending=False)["run_id"].iloc[0]
    best_model = mlflow.xgboost.load_model("runs:/" + best_model_id + "/model")
    return best_model

def get_best_pipeline(experiment_id):
    runs = mlflow.search_runs([experiment_id])
    best_pipeline_id = runs.sort_values("metrics.valid_f1", ascending=False)["run_id"].iloc[0]
    best_pipeline = mlflow.sklearn.load_model("runs:/" + best_pipeline_id + "/pipeline")
    return best_pipeline

def optimize_model():
    experiment_name = f"XGBoost_Optuna_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id

    df = pd.read_csv("water_potability.csv")
    X = df.drop(columns=["Potability"])
    y = df["Potability"]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=1892, stratify=y
    )

    def objective(trial):
        params = {
            "learning_rate" : trial.suggest_float("learning_rate", 0.001, 0.1),
            "n_estimators" : trial.suggest_int("n_estimators", 10, 1000),
            "max_depth" : trial.suggest_int("max_depth", 3, 10),
            "max_leaves" : trial.suggest_int("max_leaves", 0, 100),
        }

        run_name = f"XGBoost_lr_{params['learning_rate']:.3f}_depth_{params['max_depth']}_ml_{params['max_leaves']}_ne_{params['n_estimators']}"

        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=run_name,
            description="Optimización de XGBoost con Optuna"
        ):

            numeric_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler())
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_pipeline, X.columns.to_list())
                ]
            )

            model = xgb.XGBClassifier(**params)

            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", model)
            ])


            pipeline.fit(X_train, y_train)

            # Evaluar en validación
            y_pred = pipeline.predict(X_valid)
            f1 = f1_score(y_valid, y_pred)

            # Loguear hiperparámetros y métrica
            mlflow.log_params(params)
            mlflow.log_metric("valid_f1", f1)

            # Guardar el modelo como artefacto del run
            fitted_model = pipeline.named_steps["model"]

            mlflow.xgboost.log_model(fitted_model, name="model")
            mlflow.sklearn.log_model(pipeline, name="pipeline")
            

        return f1
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=200)
    best_model = get_best_model(experiment_id)
    best_pipeline = get_best_pipeline(experiment_id)

    with mlflow.start_run(run_name="Final_Results"):
        mlflow.log_param("xgboost_version", xgb.__version__)
        mlflow.log_param("optuna_version", optuna.__version__)
        mlflow.log_param("mlflow_version", mlflow.__version__)
        mlflow.log_param("pandas_version", pd.__version__)

        with open("models/best_xgboost.pkl", "wb") as f:
            pickle.dump(best_model, f)

        with open("models/best_xgboost_pipeline.pkl", "wb") as f:
            pickle.dump(best_pipeline, f)

        opt_hist = plot_optimization_history(study)
        mlflow.log_figure(opt_hist, "plots/optimization_history.png")

        param_importance = plot_param_importances(study)
        mlflow.log_figure(param_importance, "plots/param_importances.png")

        best_model_config = study.best_params
        mlflow.log_dict(best_model_config, "best_model_config.json")
        

        booster = best_model.get_booster()
        importance = booster.get_score(importance_type="weight")
        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)

        plt.figure(figsize=(10, 6))
        plt.bar([k for k, v in sorted_imp], [v for k, v in sorted_imp])
        plt.xticks(rotation=45, ha="right")
        plt.title("Feature Importance (XGBoost)")
        plt.tight_layout()
        mlflow.log_figure(plt.gcf(), "plots/feature_importance.png")
        plt.close()

    print(f"Optimización completa. Mejor modelo guardado en models/best_xgboost.pkl")
    print(f"Gráficos guardados en carpeta plots/")
    print(f"Experimento MLflow: {experiment_name}")


if __name__ == "__main__":
    optimize_model()