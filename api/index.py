import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Now we can import your modules safely
from src.pipeline.predict_pipeline import TitanicClassifier

# --- 1. PATH SETUP (Crucial for Vercel) ---
# Current file is in: /project/api/index.py
current_dir = Path(__file__).resolve().parent

# Root is one level up: /project/
root_dir = current_dir.parent

# Add Root to Python Path so we can import from 'src'
sys.path.append(str(root_dir))

# --- 2. INITIALIZATION ---
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


# --- 3. DATA SCHEMA ---
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


# --- 4. ROUTES ---


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
        # Convert Pydantic -> Dict -> DataFrame
        data = passenger.dict()

        # Call your pipeline
        result = classifier.predict(data)

        # Handle the result (Assuming pipeline returns simple 0 or 1, or string)
        # If your pipeline returns a complex object, we simplify it here:
        prediction_value = (
            int(result[0]) if hasattr(result, "__iter__") else int(result)
        )
        message = "Survived" if prediction_value == 1 else "Did Not Survive"

        return {"prediction": prediction_value, "result": message}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
