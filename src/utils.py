import logging
import os
from pathlib import Path

import yaml


def load_config(config_path=None):
    """
    Loads the YAML configuration file based on the environment.

    Priority:
    1. Explicit config_path argument (if provided).
    2. 'ENV' environment variable -> config/env-{ENV}.yaml
    3. Default -> config/env-dev.yaml
    """
    if config_path is None:
        # 1. Get the environment name (default to 'dev' if not set)
        # On Vercel, we will set ENV="prod"
        env = os.getenv("ENV", "dev")

        # 2. Resolve project root
        base_path = Path(__file__).resolve().parent.parent

        # 3. Construct the config path
        # e.g., config/env-dev.yaml or config/env-prod.yaml
        config_file = f"env-{env}.yaml"
        config_path = base_path / "config" / config_file

    # 4. Safety Check & Fallback
    if not config_path.exists():
        # If env-dev.yaml doesn't exist yet, try falling back to the old local.yaml
        fallback_path = config_path.parent / "local.yaml"
        if fallback_path.exists():
            print(f"Warning: {config_path.name} not found. Falling back to local.yaml")
            config_path = fallback_path
        else:
            raise FileNotFoundError(
                f"Configuration file not found at {config_path}. "
                f"Make sure you have created config/env-{os.getenv('ENV', 'dev')}.yaml"
            )

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config


def configure_logging(level="INFO"):
    """
    Sets up the logging configuration.
    """
    # Safety: Default to INFO if an invalid level string is passed
    log_level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
