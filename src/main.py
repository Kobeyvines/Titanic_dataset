from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.pipeline.predict_pipeline import TitanicClassifier

app = FastAPI(title="Titanic Survival Predictor", version="1.0.0")
classifier = TitanicClassifier()


class Passenger(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str
    Name: str
    Ticket: str
    Cabin: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "Pclass": 3,
                "Sex": "male",
                "Age": 22.0,
                "SibSp": 1,
                "Parch": 0,
                "Fare": 7.25,
                "Embarked": "S",
                "Name": "Braund, Mr. Owen Harris",
                "Ticket": "A/5 21171",
                "Cabin": "U",
            }
        }


@app.get("/")
def home():
    return {"message": "Titanic API is running!"}


@app.post("/predict")
def predict_survival(passenger: Passenger):
    try:
        # Convert Pydantic object to simple dict
        data = passenger.dict()

        # Pass data to our Prediction Pipeline
        result = classifier.predict(data)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
