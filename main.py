from typing import Optional, Union
from fastapi import FastAPI, HTTPException
import joblib
from pydantic import BaseModel

app = FastAPI()


@app.get("/")
def root():
    return {"message": "Welcome To my model"}


@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}


model = joblib.load("K-mean_model.joblib")


scaler = joblib.load("scaler.joblib")


class InputFeaturesModel1(BaseModel):
    appearance: str
    minutes_played: int
    games_injured: int
    award: str
    highest_value: str
    days_injured: int


@app.post("/predict")
async def predict(
    model_type: str,
    input_features: Union[InputFeaturesModel1,],
):
    if model_type == "model":
        model = model
        scaler = scaler

    else:
        raise HTTPException(status_code=400, detail="Invalid model_type")

    features = None
    if isinstance(input_features, InputFeaturesModel1):
        features = [
            input_features.appearance,
            input_features.minutes_played,
            input_features.games_injured,
            input_features.award,
            input_features.highest_value,
            input_features.days_injured,
        ]

    if features is None:
        raise HTTPException(status_code=400, detail="Invalid input features")

    scaled_features = scaler.transform([features])
    y_pred = model.predict(scaled_features)

    return {"pred": y_pred.tolist()[0]}
