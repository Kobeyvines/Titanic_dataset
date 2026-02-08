import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Safe import for XGBoost
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

# Shared Logic
from src.pipeline.preprocessing import engineer_features

# Config Utilities
from src.utils import configure_logging, load_config

# Load Config & Setup Logging
config = load_config()
configure_logging(config["logging"]["level"])
logger = logging.getLogger(__name__)

# --- DEFINE BASE_DIR GLOBALLY ---
# This fixes the scope issues and allows us to use it in the class
BASE_DIR = Path(__file__).resolve().parent.parent.parent
print(f"DEBUG: Script is running from: {Path(__file__).resolve()}")
print(f"DEBUG: Model Directory is set to: {BASE_DIR / config['data']['model_dir']}")


class TrainingPipeline:
    """
    Object-Oriented Pipeline for training the Titanic Survival Model.
    Configured via config/local.yaml using the 'Model Registry' pattern.
    """

    def __init__(self):
        self.base_dir = BASE_DIR

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

        # 3. Split FIRST (prevents leakage)
        test_size = config["preprocessing"]["test_size"]
        random_state = config["preprocessing"]["random_state"]

        X_train_raw, X_val_raw, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # 4. Encode train + align validation
        X_train = self._encode_data(X_train_raw)
        X_val = self._encode_data(X_val_raw)

        # Align columns
        X_val = X_val.reindex(columns=X_train.columns, fill_value=0)

        # Save columns for API
        joblib.dump(list(X_train.columns), self.model_dir / "model_columns.pkl")

        # 5. Scaling
        logger.info("Scaling features...")
        scaler = StandardScaler()
        scale_cols = ["age", "fare", "familysize"]

        X_train[scale_cols] = X_train[scale_cols].fillna(X_train[scale_cols].median())
        X_val[scale_cols] = X_val[scale_cols].fillna(X_train[scale_cols].median())

        X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
        X_val[scale_cols] = scaler.transform(X_val[scale_cols])

        joblib.dump(scaler, self.model_dir / "scaler.pkl")

        # 6. Build Model
        active_model = config["model"]["active"]
        logger.info(f"Training Active Model: {active_model}")
        model = self._build_model()

        # 7. Cross-validation (stable score)
        logger.info("Running cross-validation...")
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)

        logger.info(f"CV Accuracy: {cv_scores.mean():.4%} ± {cv_scores.std():.4%}")

        # 8. Optional Hyperparameter Tuning (RandomForest example)
        if active_model == "RandomForest" and config["model"].get("tune", False):
            logger.info("Running GridSearch tuning...")

            param_grid = {
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 5, 10],
            }

            grid = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
            grid.fit(X_train, y_train)

            model = grid.best_estimator_
            logger.info(f"Best params: {grid.best_params_}")

        else:
            model.fit(X_train, y_train)

        joblib.dump(model, self.model_dir / "model.pkl")

        # 9. Final Validation Score
        acc = accuracy_score(y_val, model.predict(X_val))
        logger.info(f"Validation Accuracy ({active_model}): {acc:.4%}")

        logger.info("Pipeline Complete.")

    def _build_model(self):
        active_model = config["model"]["active"]
        params = config["model"].get(active_model, {})

        if active_model == "SVM":
            return SVC(**params)

        elif active_model == "RandomForest":
            return RandomForestClassifier(**params)

        elif active_model == "LogisticRegression":
            return LogisticRegression(**params)

        elif active_model == "KNN":
            return KNeighborsClassifier(**params)

        elif active_model == "NeuralNetwork":
            return MLPClassifier(**params)

        # --- RESTORED THIS BLOCK ---
        elif active_model == "XGBoost":
            if XGBClassifier is None:
                raise ImportError("XGBoost is not installed. Run 'pip install xgboost'")
            return XGBClassifier(**params)
        # ---------------------------

        elif active_model == "Voting":
            logger.info("Building 2-Model Ensemble (SVM + Random Forest)...")

            # 1. Base Models
            clf_svm = SVC(**config["model"]["SVM"])
            clf_rf = RandomForestClassifier(**config["model"]["RandomForest"])

            # 2. Strict List: ONLY SVM and RF
            estimators = [("svm", clf_svm), ("rf", clf_rf)]

            return VotingClassifier(
                estimators=estimators,
                voting=config["model"]["Voting"]["voting"],
                weights=config["model"]["Voting"]["weights"],
            )

        else:
            raise ValueError(f"Unknown model type in config: {active_model}")

    def _encode_data(self, X: pd.DataFrame) -> pd.DataFrame:
        cols_to_encode = ["pclass", "sex", "embarked", "title", "deck"]
        X["pclass"] = X["pclass"].astype(str)

        # drop_first=False is CRITICAL for the API to handle single-row inputs correctly
        X_encoded = pd.get_dummies(X[cols_to_encode], drop_first=False)
        X_ticket = pd.get_dummies(X["ticketprefix"], prefix="ticket", drop_first=False)

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
