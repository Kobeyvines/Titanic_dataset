import logging
from pathlib import Path

import yaml


def load_config(config_path=None):
    """
    Loads the YAML configuration file.
    """
    if config_path is None:
        # Default to config/local.yaml relative to the project root
        base_path = Path(__file__).resolve().parent.parent
        config_path = base_path / "config" / "local.yaml"

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config


def configure_logging(level="INFO"):
    """
    Sets up the logging configuration.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
