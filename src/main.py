import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from src.pipeline.predict_pipeline import TitanicClassifier

# Setup Paths
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))


app = FastAPI(title="Titanic Survival Predictor", version="1.0.0")

# --- NEW: MOUNT STATIC FILES & TEMPLATES ---
# 1. Tell FastAPI where 'static' files (CSS, JS) are
app.mount("/static", StaticFiles(directory="src/static"), name="static")

# 2. Tell FastAPI where HTML templates are
templates = Jinja2Templates(directory="src/templates")

# Initialize Classifier
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


# --- NEW: HOME ROUTE RETURNS HTML ---
@app.get("/")
def home(request: Request):
    # This renders the index.html file
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
def predict_survival(passenger: Passenger):
    try:
        data = passenger.dict()
        result = classifier.predict(data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
