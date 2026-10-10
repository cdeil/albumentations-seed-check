"""Check the observed defect separately from the maintained release contract."""

import albumentations as A
import pytest
from reproduce import check


@pytest.mark.parametrize("seed", [0, 1, 42, 123])
def test_seeded_flips(seed):
    result = check(seed, calls=400)
    assert result["repeatable"]
    assert result["reseed_repeatable"]
    counts = result["counts"]
    if A.__version__ == "2.0.8":
        assert counts["horizontal_only"] == counts["vertical_only"] == 0
        assert result["phi_correlation"] == pytest.approx(1)
    else:
        assert A.__version__ == "2.5.0", "re-review expectations when upgrading"
        # A generous smoke gate catches the legacy zero-diversity defect.
        # These finite draws do not prove independence for all pipelines/seeds.
        assert all(50 <= value <= 150 for value in counts.values())
        assert abs(result["phi_correlation"]) < 0.2
