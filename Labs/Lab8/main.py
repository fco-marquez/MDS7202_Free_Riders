# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import uvicorn
import pandas as pd

with open("models/best_xgboost_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI(
    title="API de Potabilidad de Agua",
    description="Esta API predice si una muestra de agua es potable (1) o no (0) en base a 9 mediciones físico-químicas.",
    version="1.0.0"
)

class WaterMeasurement(BaseModel):
    ph: float
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float
    Conductivity: float
    Organic_carbon: float
    Trihalomethanes: float
    Turbidity: float

@app.get("/")
def home():
    return {
        "mensaje": "Modelo de predicción de potabilidad de agua",
        "descripcion": "Envíe 9 características de la muestra en un POST a /potabilidad/ para obtener la predicción (0 = no potable, 1 = potable)."
    }

@app.post("/potabilidad/")
def predict_potabilidad(medicion: WaterMeasurement):
    data = {
        "ph": [medicion.ph],
        "Hardness": [medicion.Hardness],
        "Solids": [medicion.Solids],
        "Chloramines": [medicion.Chloramines],
        "Sulfate": [medicion.Sulfate],
        "Conductivity": [medicion.Conductivity],
        "Organic_carbon": [medicion.Organic_carbon],
        "Trihalomethanes": [medicion.Trihalomethanes],
        "Turbidity": [medicion.Turbidity],
    }

    features_df = pd.DataFrame(data)

    pred = model.predict(features_df)[0]

    return {"potabilidad": int(pred)}

if __name__ == '__main__':
    uvicorn.run('main:app', port = 8000)