import logging
import sys
import json
import torch
from pathlib import Path
import numpy as np
import random
import os

def load_settings(settings_path: str | Path) -> dict:
    settings_path = Path(settings_path)

    if not settings_path.exists():
        raise FileNotFoundError(f"Settings file not found: {settings_path}")

    with open(settings_path, "r") as f:
        settings = json.load(f)

    return settings

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True   # important: avoids duplicate handlers
    )
