"""Pin the observed compatibility boundary; fail when it changes."""

import albumentations as A
import pytest

pytest.importorskip("supervision")
pytest.importorskip("rfdetr")
pytest.importorskip("ultralytics")
from integration_checks import check_integrations


def test_released_adapters():
    result = check_integrations()
    assert result["supervision_boxes"] == "passed"
    assert result["rfdetr_pixel"] == "passed"
    assert result["ultralytics_default"] == "passed"
    assert result["ultralytics_pixel"] == "passed"
    if A.__version__ == "2.0.8":
        assert result["rfdetr_spatial"] == "passed"
        assert result["ultralytics_spatial"] == "passed"
    else:
        assert "unexpected keyword argument 'format'" in result["rfdetr_spatial"]
        assert result["ultralytics_spatial"].startswith("incompatible:")
