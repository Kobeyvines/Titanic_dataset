from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Load artifacts (cached globally so we don't load them on every request)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCALER = joblib.load(PROJECT_ROOT / "models" / "scaler.pkl")
MODEL_COLUMNS = joblib.load(PROJECT_ROOT / "models" / "model_columns.pkl")


def preprocess_input(input_data: dict) -> pd.DataFrame:
    """
    Takes raw dictionary from API, returns scaled DataFrame ready for model.
    """
    # 1. Convert Dictionary to DataFrame
    df = pd.DataFrame([input_data])  # Wrap in list to make it a row

    # ---------------------------------------------------------
    # FEATURE ENGINEERING (Copy-Paste logic from Notebook)
    # ---------------------------------------------------------

    # A. Family Size
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # B. Title Extraction
    df["title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    # (Simplified mapping for demo - normally you'd use the full dict)
    title_mapping = {"Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master"}
    df["title"] = df["title"].map(title_mapping).fillna("Rare")

    # C. Fare Log Transform
    # Note: If Fare is missing in input, you might need a default value here
    df["Fare"] = np.log1p(df["Fare"])

    # D. Ticket Prefix (Simplifying: just dropping it for now or adding logic)
    # If you used TicketPrefix, you must repeat the extraction logic here.

    # ---------------------------------------------------------
    # ENCODING & ALIGNMENT (The Magic Step)
    # ---------------------------------------------------------

    # 2. One-Hot Encode
    categorical_cols = ["Pclass", "Sex", "Embarked", "title"]  # Add others used
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # 3. ALIGN COLUMNS
    # This is critical: Enforce the DataFrame to have exactly the training columns
    # Any missing column (like 'Title_Rare') gets filled with 0
    df_encoded = df_encoded.reindex(columns=MODEL_COLUMNS, fill_value=0)

    # ---------------------------------------------------------
    # SCALING
    # ---------------------------------------------------------

    # 4. Scale Numerical Columns
    scale_cols = ["Age", "Fare", "FamilySize"]
    df_encoded[scale_cols] = SCALER.transform(df_encoded[scale_cols])

    return df_encoded
