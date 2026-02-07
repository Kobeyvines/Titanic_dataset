# Add this near the top of src/main.py
# import os
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Import your custom pipeline
from src.pipeline.predict_pipeline import TitanicClassifier

# 1. Find the Project Root
# This moves up 2 levels from src/main.py to the Root
BASE_DIR = Path(__file__).resolve().parent.parent

# 2. Define Template Directory
# Try Root first (Vercel usually puts it here)
TEMPLATES_DIR = BASE_DIR / "templates"

# Fallback: Check if it's inside src/ (Local setup sometimes)
if not TEMPLATES_DIR.exists():
    TEMPLATES_DIR = BASE_DIR / "src" / "templates"

# 3. Mount Templates
if TEMPLATES_DIR.exists():
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
else:
    # If this prints in Vercel logs, we know the folder is missing
    print(f"❌ CRITICAL: Templates not found at {TEMPLATES_DIR}")

# 1. Setup Paths (Robust for Vercel)
# This finds the root folder regardless of where the code runs
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))


app = FastAPI(title="Titanic Survival Predictor", version="1.0.0")

# 2. Setup Templates & Static Files
# We check where they are located to avoid "Directory Not Found" errors
if (BASE_DIR / "templates").exists():
    templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
    # Mount static if it exists
    if (BASE_DIR / "static").exists():
        app.mount(
            "/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static"
        )
else:
    # Fallback to src/templates if not in root
    templates = Jinja2Templates(directory=str(BASE_DIR / "src/templates"))
    if (BASE_DIR / "src/static").exists():
        app.mount(
            "/static",
            StaticFiles(directory=str(BASE_DIR / "src/static")),
            name="static",
        )

# 3. Initialize Classifier
# This calls your class from predict_pipeline.py
classifier = TitanicClassifier()


class Passenger(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str
    Name: str = "Unknown"
    Ticket: str = "0000"
    Cabin: Optional[str] = None


# --- ROUTES ---


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """
    Serves the HTML form.
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
def predict_survival(passenger: Passenger):
    """
    Passes data directly to your TitanicClassifier.
    """
    try:
        # Convert Pydantic object to simple Python dictionary
        data = passenger.dict()

        # DIRECT CALL: We trust your class to handle DataFrame conversion and formatting
        result = classifier.predict(data)

        return result

    except Exception as e:
        print(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
