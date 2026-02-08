import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# --- 1. PATH SETUP (Crucial for Vercel) ---
# Current file is in: /project/api/index.py
current_dir = Path(__file__).resolve().parent

# Root is one level up: /project/
root_dir = current_dir.parent

# Add Root to Python Path so we can import from 'src'
sys.path.append(str(root_dir))

# --- 2. IMPORTS (Must happen AFTER sys.path append) ---
# We use '# noqa: E402' to tell flake8 to ignore "module level import not at top of file"
from src.pipeline.predict_pipeline import TitanicClassifier  # noqa: E402

# --- 3. INITIALIZATION ---
app = FastAPI(title="Titanic Survival API")

# Setup Paths to Static/Templates (They are inside src/)
static_path = root_dir / "src" / "static"
templates_path = root_dir / "src" / "templates"

# Mount Static Files (CSS/JS)
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Setup HTML Templates
if templates_path.exists():
    templates = Jinja2Templates(directory=str(templates_path))
else:
    templates = None
    print(f"Warning: Templates not found at {templates_path}")

# Load the Model
try:
    classifier = TitanicClassifier()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Model failed to load: {e}")
    classifier = None


# --- 4. DATA SCHEMA ---
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
    Cabin: str = None


# --- 5. ROUTES ---
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    if templates is None:
        return "<h1>Error: Templates not found</h1>"
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(passenger: Passenger):
    if not classifier:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    try:
        data = passenger.dict()

        # UNPACK the tuple: (0, 0.76)
        prediction, probability = classifier.predict(data)

        prediction_value = int(prediction)
        message = "Survived" if prediction_value == 1 else "Did Not Survive"

        # Format as percentage (e.g., 0.762 -> 76.2)
        prob_percent = round(probability * 100, 1)

        return {
            "prediction": prediction_value,
            "result": message,
            "probability": prob_percent,
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
