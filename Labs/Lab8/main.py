# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import uvicorn

with open("models/best_xgboost.pkl", "rb") as f:
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
    features = np.array([[
        medicion.ph,
        medicion.Hardness,
        medicion.Solids,
        medicion.Chloramines,
        medicion.Sulfate,
        medicion.Conductivity,
        medicion.Organic_carbon,
        medicion.Trihalomethanes,
        medicion.Turbidity
    ]])

    # Hacer la predicción
    # TODO: tenemos que revisar que el modelo guardado en verdad sea el pipeline completo, 
    # para que la data que recibimos acá sea automáticamente transformada al hacer predict.
    pred = model.predict(features)[0]

    return {"potabilidad": int(pred)}

if __name__ == '__main__':
    uvicorn.run('fastapi_app:app', port = 8000)