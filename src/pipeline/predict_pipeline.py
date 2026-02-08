from pathlib import Path

import joblib
import pandas as pd

# Import your feature engineering logic
from src.pipeline.preprocessing import engineer_features

# Define Base Directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = BASE_DIR / "models"


class TitanicClassifier:
    def __init__(self):
        """
        Loads Model, Scaler, and Column names on initialization.
        """
        # 1. Load the Model
        model_path = MODEL_DIR / "model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        self.model = joblib.load(model_path)
        print(f"✅ Model loaded from {model_path}")

        # 2. Load the Scaler (CRITICAL for your manual preprocessing)
        scaler_path = MODEL_DIR / "scaler.pkl"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
            print(f"✅ Scaler loaded from {scaler_path}")
        else:
            print("⚠️ Warning: scaler.pkl not found. Predictions may be inaccurate.")
            self.scaler = None

        # 3. Load Model Columns (CRITICAL for aligning dummy variables)
        columns_path = MODEL_DIR / "model_columns.pkl"
        if columns_path.exists():
            self.model_columns = joblib.load(columns_path)
            print(f"✅ Model Columns loaded from {columns_path}")
        else:
            print("⚠️ Warning: model_columns.pkl not found. Crash likely.")
            self.model_columns = []

    def predict(self, data: dict):
        # 1. Convert Dict to DataFrame
        df = pd.DataFrame([data])

        # 2. Lowercase columns to match training
        df.columns = df.columns.str.lower()

        # 3. Engineer Features (Title, FamilySize, etc.)
        df = engineer_features(df)

        # 4. Manual Encoding (Get Dummies)
        # We must re-encode manually because we aren't using a pipeline
        categorical_cols = ["pclass", "sex", "embarked", "title", "deck"]
        # Convert to string to ensure consistent get_dummies behavior
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)

        df = pd.get_dummies(df)

        # 5. Align Columns (The Magic Fix)
        # This ensures the dataframe has exactly the same columns as training
        # Fills missing columns (like 'Title_Rare') with 0
        if self.model_columns:
            df = df.reindex(columns=self.model_columns, fill_value=0)

        # 6. Scaling
        scale_cols = ["age", "fare", "familysize"]
        if self.scaler:
            try:
                # Only scale if columns exist
                existing_scale_cols = [c for c in scale_cols if c in df.columns]
                if existing_scale_cols:
                    df[existing_scale_cols] = self.scaler.transform(
                        df[existing_scale_cols]
                    )
            except Exception as e:
                print(f"Scaling failed: {e}")

        # 7. Predict & Probability
        prediction = self.model.predict(df)[0]

        try:
            # Get probability of Class 1 (Survival)
            # The [0][1] gets the probability of the Positive class
            probability = self.model.predict_proba(df)[0][1]
        except AttributeError:
            probability = 0.0

        return int(prediction), float(probability)
