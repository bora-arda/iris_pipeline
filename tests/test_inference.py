import pytest
from inference.inference import IrisInference


def test_missing_model_raises():
    with pytest.raises(FileNotFoundError):
        IrisInference(settings_path="invalid_settings.json")
