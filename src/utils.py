import logging
import os
from pathlib import Path

import yaml


def load_config(config_path=None):
    """
    Loads the YAML configuration file.

    Priority:
    1. Explicit config_path argument.
    2. Root folder -> env-{ENV}.yaml
    3. Config folder -> local.yaml (Fallback)
    """
    if config_path is None:
        # 1. Get Environment
        env = os.getenv("ENV", "dev")

        # 2. Define Base Paths
        project_root = Path(__file__).resolve().parent.parent

        # 3. Look for env file in ROOT
        root_config = project_root / f"env-{env}.yaml"

        # 4. Look for local.yaml in CONFIG folder
        fallback_config = project_root / "config" / "local.yaml"

        if root_config.exists():
            config_path = root_config
            print(f"Loaded config from ROOT: {config_path.name}")
        elif fallback_config.exists():
            config_path = fallback_config
            print(
                f"env-{env}.yaml not found in root. Falling back to config/local.yaml"
            )
        else:
            raise FileNotFoundError(
                f"Configuration file not found.\n"
                f"Checked: {root_config}\n"
                f"Checked: {fallback_config}"
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
