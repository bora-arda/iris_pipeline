import logging
import sys
import json
from pathlib import Path


def load_settings(settings_path: str | Path) -> dict:
    settings_path = Path(settings_path)

    if not settings_path.exists():
        raise FileNotFoundError(f"Settings file not found: {settings_path}")

    with open(settings_path, "r") as f:
        settings = json.load(f)

    return settings

def is_file_exist(file_path: str):
    """
    Ensure that a file exists. Raises FileNotFoundError otherwise.
    Returns Path object for safe downstream usage.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not file_path.is_file():
        raise ValueError(f"File is not a file: {file_path}")
    
    return file_path

def is_dir_exists(path: str) -> Path:
    """
    Ensure that a directory exists. Creates it if missing.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    if not path.is_dir():
        raise ValueError(f"Path exists but is not a directory: {path}")

    return path

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True   # important: avoids duplicate handlers
    )
