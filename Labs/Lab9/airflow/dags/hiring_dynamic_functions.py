import os

import gradio as gr
import joblib
import pandas as pd
from airflow.models import Variable
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


def create_folders(ds, **kwargs):
    base_path = f"/root/airflow/{ds}"
    os.makedirs(f"{base_path}/raw", exist_ok=True)
    os.makedirs(f"{base_path}/preprocessed", exist_ok=True)
    os.makedirs(f"{base_path}/splits", exist_ok=True)
    os.makedirs(f"{base_path}/models", exist_ok=True)


def load_and_merge(ds, **kwargs):
    base_path = f"/root/airflow/{ds}"
    # Leer archivo data_1.csv desde la carpeta raw
    df1 = pd.read_csv(f"{base_path}/raw/data_1.csv")
    # Leer archivo data_2.csv desde la carpeta raw si es que esta disponible
    df2 = (
        pd.read_csv(f"{base_path}/raw/data_2.csv")
        if os.path.exists(f"{base_path}/raw/data_2.csv")
        else pd.DataFrame()
    )
    # concatenar los dataframes
    df = pd.concat([df1, df2], ignore_index=True)
    # Guardar el dataframe combinado
    df.to_csv(f"{base_path}/preprocessed/combined_data.csv", index=False)


def split_data(ds, **kwargs):
    base_path = f"/root/airflow/{ds}"
    df = pd.read_csv(f"{base_path}/preprocessed/combined_data.csv")
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["HiringDecision"], random_state=1892
    )
    # Guardar los conjuntos divididos en la carpeta splits
    train_df.to_csv(f"{base_path}/splits/train.csv", index=False)
    test_df.to_csv(f"{base_path}/splits/test.csv", index=False)


def train_model(ds, model, **kwargs):
    base_path = f"/root/airflow/{ds}"
    # Leer los conjuntos de entrenamiento y prueba desde la carpeta splits
    train_df = pd.read_csv(f"{base_path}/splits/train.csv")
    test_df = pd.read_csv(f"{base_path}/splits/test.csv")

    # Separar características y etiquetas
    X_train = train_df.drop("HiringDecision", axis=1)
    y_train = train_df["HiringDecision"]
    X_test = test_df.drop("HiringDecision", axis=1)
    y_test = test_df["HiringDecision"]

    # Pipeline de preprocesamiento
    numeric_features = [
        "Age",
        "DistanceFromCompany",
        "InterviewScore",
        "SkillScore",
        "PersonalityScore",
    ]
    ordinal_features = ["EducationLevel", "ExperienceYears"]

    categorical_features = ["Gender", "RecruitmentStrategy", "PreviousCompanies"]

    # Pipeline de preprocesamiento
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("ord", OrdinalEncoder(), ordinal_features),
            (
                "cat",
                OneHotEncoder(drop="first", sparse_output=False),
                categorical_features,
            ),
        ]
    )
    # Pipeline completo: preprocesamiento + modelo
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )

    pipeline.fit(X_train, y_train)

    # Guardar el modelo entrenado
    joblib.dump(pipeline, f"{base_path}/models/hiring_{model.__class__.__name__}.pkl")


def evaluate_models(ds, **kwargs):
    base_path = f"/root/airflow/{ds}"
    test_df = pd.read_csv(f"{base_path}/splits/test.csv")
    X_test = test_df.drop("HiringDecision", axis=1)
    y_test = test_df["HiringDecision"]

    best_model = None
    best_accuracy = 0.0
    best_model_name = ""

    for model_file in os.listdir(f"{base_path}/models"):
        if model_file.endswith(".pkl"):
            model_path = os.path.join(f"{base_path}/models", model_file)
            model = joblib.load(model_path)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Model: {model_file}, Accuracy: {accuracy:.4f}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_model_name = model_file.strip(".pkl")

    if best_model is not None:
        joblib.dump(best_model, f"{base_path}/models/best_hiring_model.pkl")
        print(f"Best Model: {best_model_name} with Accuracy: {best_accuracy:.4f}")
