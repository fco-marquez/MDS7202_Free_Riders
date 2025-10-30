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
    os.makedirs(f"{base_path}/splits", exist_ok=True)
    os.makedirs(f"{base_path}/models", exist_ok=True)


def split_data(ds, **kwargs):
    base_path = f"/root/airflow/{ds}"
    # Leer archivo data_1.csv desde la carpeta raw
    df = pd.read_csv(f"{base_path}/raw/data_1.csv")
    # Dividir los datos en conjuntos de entrenamiento y prueba (80% - 20%)
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["HiringDecision"], random_state=1892
    )
    # Guardar los conjuntos divididos en la carpeta splits
    train_df.to_csv(f"{base_path}/splits/train.csv", index=False)
    test_df.to_csv(f"{base_path}/splits/test.csv", index=False)


def preprocess_and_train(ds, **kwargs):
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
            ("classifier", RandomForestClassifier(random_state=1892, n_estimators=100)),
        ]
    )

    # Entrenar el pipeline completo
    pipeline.fit(X_train, y_train)

    # Realizar predicciones en el conjunto de prueba
    y_pred = pipeline.predict(X_test)

    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, pos_label=1)  # Clase positiva = 1 (Contratado)

    print(f"Model Accuracy: {accuracy:.4f}")
    print(f"F1-Score (clase positiva - Contratado): {f1:.4f}")

    # Guardar el pipeline entrenado como archivo joblib
    model_path = f"/root/airflow/{ds}/models/hiring_model.joblib"
    joblib.dump(pipeline, model_path)
    print(f"Modelo guardado en: {model_path}")


def predict(file, model_path):

    pipeline = joblib.load(model_path)
    input_data = pd.read_json(file)
    predictions = pipeline.predict(input_data)
    print(f"La prediccion es: {predictions}")
    labels = ["No contratado" if pred == 0 else "Contratado" for pred in predictions]

    return {"Predicción": labels[0]}


def gradio_interface(ds, **kwargs):

    model_path = f"/root/airflow/{ds}/models/hiring_model.joblib"

    interface = gr.Interface(
        fn=lambda file: predict(file, model_path),
        inputs=gr.File(label="Sube un archivo JSON"),
        outputs="json",
        title="Hiring Decision Prediction",
        description="Sube un archivo JSON con las características de entrada para predecir si Vale será contratada o no.",
    )
    interface.launch(share=True, server_name="0.0.0.0", server_port=7860)
