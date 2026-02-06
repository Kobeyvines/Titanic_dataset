import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Shared Logic
from src.pipeline.preprocessing import engineer_features

# Config Utilities
from src.utils import configure_logging, load_config

# Load Config & Setup Logging
config = load_config()
configure_logging(config["logging"]["level"])
logger = logging.getLogger(__name__)


class TrainingPipeline:
    """
    Object-Oriented Pipeline for training the Titanic Survival Model.
    Configured via config/local.yaml.
    """

    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent.parent.parent

        # Load paths from config
        self.data_path = self.base_dir / config["data"]["train_path"]
        self.model_dir = self.base_dir / config["data"]["model_dir"]

        self.model_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        logger.info("=" * 50)
        logger.info("Starting Titanic Training Pipeline")
        logger.info("=" * 50)

        # 1. Load Data
        if not self.data_path.exists():
            raise FileNotFoundError(f"Train file not found at {self.data_path}")

        df = pd.read_csv(self.data_path)
        df.columns = df.columns.str.lower()

        target_col = config["data"]["target_column"].lower()
        y = df[target_col]
        X = df.drop(columns=[target_col])

        logger.info(f"Data shape: X={X.shape}, y={y.shape}")

        # 2. Feature Engineering
        logger.info("Applying Shared Feature Engineering...")
        X = engineer_features(X)

        # 3. Encoding
        X_encoded = self._encode_data(X)

        # Save Model Columns (Critical for API alignment)
        joblib.dump(list(X_encoded.columns), self.model_dir / "model_columns.pkl")

        # 4. Split Data (Using config parameters)
        test_size = config["preprocessing"]["test_size"]
        random_state = config["preprocessing"]["random_state"]

        X_train, X_val, y_train, y_val = train_test_split(
            X_encoded, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # 5. Scaling
        logger.info("Scaling features...")
        scaler = StandardScaler()

        # Fill missing Age for scaling
        scale_cols = ["age", "fare", "familysize"]
        X_train[scale_cols] = X_train[scale_cols].fillna(X_train[scale_cols].median())
        X_val[scale_cols] = X_val[scale_cols].fillna(X_train[scale_cols].median())

        X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
        X_val[scale_cols] = scaler.transform(X_val[scale_cols])

        joblib.dump(scaler, self.model_dir / "scaler.pkl")

        # 6. Train Model (Dynamic Building)
        logger.info(f"Training Model: {config['model']['type']}")
        model = self._build_model()
        model.fit(X_train, y_train)

        joblib.dump(model, self.model_dir / "model.pkl")

        # 7. Evaluate
        acc = accuracy_score(y_val, model.predict(X_val))
        logger.info(f"Validation Accuracy: {acc:.4%}")

        logger.info("Pipeline Complete.")

    def _build_model(self):
        """
        Factory method to create the model based on config.
        """
        model_type = config["model"]["type"]
        params = config["model"]["parameters"]

        if model_type == "SVM":
            return SVC(**params)
        elif model_type == "RandomForest":
            return RandomForestClassifier(**params)
        else:
            raise ValueError(f"Unknown model type in config: {model_type}")

    def _encode_data(self, X: pd.DataFrame) -> pd.DataFrame:
        cols_to_encode = ["pclass", "sex", "embarked", "title", "deck"]
        X["pclass"] = X["pclass"].astype(str)

        X_encoded = pd.get_dummies(X[cols_to_encode], drop_first=True)
        X_ticket = pd.get_dummies(X["ticketprefix"], prefix="ticket", drop_first=True)

        cols_to_drop = cols_to_encode + [
            "ticketprefix",
            "name",
            "ticket",
            "sibsp",
            "parch",
            "cabin",
            "passengerid",
        ]
        X_processed = X.drop(columns=cols_to_drop, errors="ignore")

        return pd.concat([X_processed, X_encoded, X_ticket], axis=1)


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run()
