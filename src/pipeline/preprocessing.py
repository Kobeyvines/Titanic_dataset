from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Load artifacts (cached globally so we don't load them on every request)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCALER = joblib.load(PROJECT_ROOT / "models" / "scaler.pkl")
MODEL_COLUMNS = joblib.load(PROJECT_ROOT / "models" / "model_columns.pkl")


def clean_ticket(ticket):
    """
    Extracts the prefix from the ticket string.
    e.g., "A/5 21171" -> "A5"
    e.g., "12345" -> "X"
    """
    ticket = str(ticket)
    if ticket.isdigit():
        return "X"
    else:
        # Get the first part, remove special characters
        prefix = ticket.split(" ")[0]
        prefix = prefix.replace(".", "").replace("/", "")
        return prefix


def preprocess_input(input_data: dict) -> pd.DataFrame:
    """
    Takes raw dictionary from API, returns scaled DataFrame ready for model.

    Expected input keys (case-insensitive, will be lowercased):
    - pclass, name, sex, age, sibsp, parch, fare, ticket, embarked
    """
    # 1. Convert Dictionary to DataFrame
    df = pd.DataFrame([input_data])

    # 2. Lowercase all columns for consistency with training data
    df.columns = df.columns.str.lower()

    # ---------------------------------------------------------
    # FEATURE ENGINEERING (Matching Notebook Logic)
    # ---------------------------------------------------------

    # A. Family Size
    df["familysize"] = df["sibsp"] + df["parch"] + 1
    df["isalone"] = (df["familysize"] == 1).astype(int)

    # B. Title Extraction
    if "name" in df.columns:
        df["title"] = df["name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
        title_mapping = {
            "Mr": "Mr",
            "Miss": "Miss",
            "Mrs": "Mrs",
            "Master": "Master",
            "Dr": "Rare",
            "Rev": "Rare",
            "Col": "Rare",
            "Major": "Rare",
            "Mlle": "Miss",
            "Countess": "Rare",
            "Ms": "Miss",
            "Lady": "Rare",
            "Jonkheer": "Rare",
            "Don": "Rare",
            "Dona": "Rare",
            "Mme": "Mrs",
            "Capt": "Rare",
            "Sir": "Rare",
        }
        df["title"] = df["title"].map(title_mapping).fillna("Rare")
    else:
        df["title"] = "Rare"

    # C. Fare Log Transform (handle missing first)
    if "fare" in df.columns:
        if df["fare"].isnull().any():
            df["fare"] = df["fare"].fillna(7.75)  # Approximate median
        df["fare"] = np.log1p(df["fare"])

    # D. Deck Extraction from Cabin
    if "cabin" in df.columns:
        df["deck"] = df["cabin"].str[0].fillna("U")
    else:
        df["deck"] = "U"  # Default for missing cabin

    # E. Ticket Prefix
    if "ticket" in df.columns:
        df["ticketprefix"] = df["ticket"].apply(clean_ticket)
    else:
        df["ticketprefix"] = "X"

    # ---------------------------------------------------------
    # ENCODING & ALIGNMENT (CRITICAL FIXES HERE)
    # ---------------------------------------------------------

    # Define columns to encode (must match notebook exactly)
    cols_to_encode = ["pclass", "sex", "embarked", "title", "deck"]

    # Ensure they exist
    for col in cols_to_encode:
        if col not in df.columns:
            df[col] = "unknown"

    # Convert pclass to string for encoding (handling int vs str input mismatch)
    df["pclass"] = df["pclass"].astype(str)

    # FIX 1: Set drop_first=False
    # If we input a single row (e.g., sex="male"), drop_first=True would delete
    # the "sex_male" column, leaving us with NO data.
    # By keeping it False, we generate "sex_male".
    df_encoded = pd.get_dummies(df[cols_to_encode], drop_first=False)

    # FIX 2: Set drop_first=False and prefix='ticket'
    # Same logic for tickets. Ensures we don't erase the ticket info.
    df_ticket = pd.get_dummies(df["ticketprefix"], prefix="ticket", drop_first=False)

    # Drop raw text columns and combine
    cols_to_drop = cols_to_encode + [
        "ticketprefix",
        "name",
        "ticket",
        "sibsp",
        "parch",
        "cabin",
    ]
    df_processed = df.drop(columns=cols_to_drop, errors="ignore")

    df_final = pd.concat([df_processed, df_encoded, df_ticket], axis=1)

    # FIX 3: Reindex acts as the filter
    # This aligns the columns to exactly what the model saw during training.
    # It drops the "extra" columns we kept (like sex_female if unnecessary)
    # and fills missing columns (like other ticket prefixes) with 0.
    df_final = df_final.reindex(columns=MODEL_COLUMNS, fill_value=0)

    # ---------------------------------------------------------
    # SCALING (Lowercase column names to match training)
    # ---------------------------------------------------------

    scale_cols = ["age", "fare", "familysize"]
    # Only scale if columns exist in the final dataframe
    scale_cols_present = [col for col in scale_cols if col in df_final.columns]

    if scale_cols_present:
        df_final[scale_cols_present] = SCALER.transform(df_final[scale_cols_present])

    return df_final
