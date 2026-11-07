"""Loading errors must be explicit, so a container that starts without its files fails with a clear message."""
import pytest

from src.serving import ForecastService


def test_missing_files_raise_a_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        ForecastService.load(history_path=tmp_path / "missing.csv", model_dir=tmp_path)
