from pathlib import Path

import joblib

from src.pipeline.preprocessing import preprocess_input

# Define paths to artifacts
# (Using .parent logic ensures this works no matter where you run it from)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "model.pkl"


class TitanicClassifier:
    def __init__(self):
        """
        Loads the model and scaler when the class is initialized.
        This prevents loading them 1000 times for 1000 requests.
        """
        self.model = joblib.load(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")

    def predict(self, input_data: dict) -> dict:
        """
        Orchestrates the prediction process:
        1. Preprocess the input (using your existing logic)
        2. Make prediction
        3. Format the result
        """
        # 1. Preprocess
        # This calls the function we polished earlier in preprocessing.py
        df_processed = preprocess_input(input_data)

        # 2. Predict Probability
        # Returns something like [[0.15, 0.85]] -> We want the second number (Survival %)
        prob_survival = self.model.predict_proba(df_processed)[0][1]

        # 3. Predict Class (0 or 1)
        prediction = self.model.predict(df_processed)[0]

        # 4. Return clean dictionary
        return {
            "survival_probability": round(float(prob_survival), 4),
            "prediction": int(prediction),
            "label": "Survived" if prediction == 1 else "Did Not Survive",
        }
