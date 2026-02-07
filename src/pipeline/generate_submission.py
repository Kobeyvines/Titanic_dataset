from pathlib import Path

import joblib
import pandas as pd

from src.pipeline.preprocessing import engineer_features
from src.utils import load_config

# --- 1. SETUP PATHS ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Load Config
config = load_config()

# Define paths from Config
DATA_PATH = BASE_DIR / config["data"]["test_path"]
MODEL_DIR = BASE_DIR / config["data"]["model_dir"]
PREDICTIONS_DIR = BASE_DIR / Path(config["data"]["predictions_path"]).parent
OUTPUT_FILE = BASE_DIR / config["data"]["predictions_path"]


# test
print(f"DEBUG: Script is running from: {Path(__file__).resolve()}")
print(f"DEBUG: Model Directory is set to: {BASE_DIR / config['data']['model_dir']}")

# Create output directory if it doesn't exist
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Loading Test Data from: {DATA_PATH}")
print(f"Loading Models from:    {MODEL_DIR}")

# --- 2. LOAD ARTIFACTS ---
try:
    model = joblib.load(MODEL_DIR / "model.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    model_columns = joblib.load(MODEL_DIR / "model_columns.pkl")
    print("Artifacts loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: Missing {e}. Run train_pipeline.py first.")
    exit()

# --- 3. LOAD DATA ---
try:
    test_df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"Error: Could not find test.csv at {DATA_PATH}")
    exit()

passenger_ids = test_df["PassengerId"]

# Normalize columns to lowercase (matches training logic)
test_df.columns = test_df.columns.str.lower()

print("Running Feature Engineering...")

# --- 4. APPLY SHARED LOGIC ---
test_df = engineer_features(test_df)

# --- 5. ENCODING & ALIGNMENT ---
print("Aligning Columns...")

cols_to_encode = ["pclass", "sex", "embarked", "title", "deck"]
if "pclass" in test_df.columns:
    test_df["pclass"] = test_df["pclass"].astype(str)

# One-Hot Encode
df_encoded = pd.get_dummies(test_df[cols_to_encode], drop_first=True)
df_ticket = pd.get_dummies(test_df["ticketprefix"], prefix="ticket", drop_first=True)

# Clean up raw columns
cols_to_drop = cols_to_encode + [
    "ticketprefix",
    "name",
    "ticket",
    "sibsp",
    "parch",
    "cabin",
    "passengerid",
]
df_processed = test_df.drop(columns=cols_to_drop, errors="ignore")

# Combine
df_final = pd.concat([df_processed, df_encoded, df_ticket], axis=1)

# CRITICAL: Force the test columns to match the Training columns exactly
df_final = df_final.reindex(columns=model_columns, fill_value=0)

# --- 6. SCALING ---
scale_cols = ["age", "fare", "familysize"]
# Handle missing age/fare for scaling using median
df_final[scale_cols] = df_final[scale_cols].fillna(df_final[scale_cols].median())
df_final[scale_cols] = scaler.transform(df_final[scale_cols])

# --- 7. PREDICT & SAVE ---
print("Making Predictions...")
predictions = model.predict(df_final)

submission = pd.DataFrame({"PassengerId": passenger_ids, "Survived": predictions})

submission.to_csv(OUTPUT_FILE, index=False)

print("SUCCESS! File generated at:")
print(f"  {OUTPUT_FILE}")
print(f"  Rows: {len(submission)}")
