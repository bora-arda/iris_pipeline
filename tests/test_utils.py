import pytest
from pathlib import Path
from core.utils import is_dir_exists, is_file_exist


def test_is_file_exist_ok(tmp_path):
    file_path = tmp_path / "test.txt"
    file_path.write_text("hello")

    result = is_file_exist(file_path)
    assert result.exists()


def test_is_file_exist_missing():
    with pytest.raises(FileNotFoundError):
        is_file_exist("non_existing_file.txt")


def test_is_dir_exists_creates(tmp_path):
    new_dir = tmp_path / "my_dir"
    is_dir_exists(new_dir)
    assert new_dir.exists()
