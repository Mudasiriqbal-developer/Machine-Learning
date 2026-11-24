from pathlib import Path
import json
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Adult Income Prediction API")

MODEL_PATH = Path(__file__).parent / "models" / "best_model.pkl"
METRICS_PATH = Path(__file__).parent / "models" / "metrics.json"

model = joblib.load(MODEL_PATH)
with open(METRICS_PATH, "r", encoding="utf-8") as f:
    metrics = json.load(f)


class Person(BaseModel):
    age: int
    workclass: str
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = 0
    capital_loss: int = 0
    hours_per_week: int
    native_country: str


@app.get("/metrics")
def get_metrics():
    return metrics


@app.post("/predict")
def predict(person: Person):
    row = {
        "age": person.age,
        "workclass": person.workclass,
        "education": person.education,
        "education-num": person.education_num,
        "marital-status": person.marital_status,
        "occupation": person.occupation,
        "relationship": person.relationship,
        "race": person.race,
        "sex": person.sex,
        "capital-gain": person.capital_gain,
        "capital-loss": person.capital_loss,
        "hours-per-week": person.hours_per_week,
        "native-country": person.native_country,
        "capital_net": person.capital_gain - person.capital_loss,
        "has_capital": int((person.capital_gain > 0) or (person.capital_loss > 0))
    }

    df_single = pd.DataFrame([row])
    df_single["age_group"] = pd.cut(
        df_single["age"],
        bins=[0, 25, 40, 60, 100],
        labels=["under 25", "25-40", "41-60", "above 60"]
    )

    df_single = df_single[metrics["feature_columns"]]

    pred = int(model.predict(df_single)[0])
    prob = float(model.predict_proba(df_single)[0][1])
    label = ">50K" if pred == 1 else "<=50K"

    return {
        "prediction": pred,
        "label": label,
        "probability": round(prob, 4)
    }
