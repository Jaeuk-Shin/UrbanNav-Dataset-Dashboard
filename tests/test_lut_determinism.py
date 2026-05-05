"""LUT determinism check.

Rebuild the validation LUT from ``youtube.db`` and assert that every
key in the resulting ``.npz`` is byte-equivalent to the committed
baseline ``lut_val.npz``. This is the parity guarantee from the
refactor plan, formalised as a pytest.

The test is skipped when the database or baseline LUT is unavailable
(e.g. fresh checkout, CI without dataset access).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
DB_PATH = REPO / "youtube.db"
BASELINE_LUT = REPO / "lut_val.npz"


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


@pytest.mark.skipif(
    not DB_PATH.exists() or not BASELINE_LUT.exists(),
    reason="requires youtube.db and lut_val.npz checked out alongside the repo",
)
def test_val_lut_is_deterministic(tmp_path):
    from curation.build_lut import build_lut
    from curation.filters import FilterConfig

    baseline = _load_npz(BASELINE_LUT)

    cfg = FilterConfig(**json.loads(str(baseline["filter_cfg"])))
    out = tmp_path / "lut_val_rebuilt.npz"

    build_lut(
        db_path=str(DB_PATH),
        split=str(baseline["split"]),
        context_size=int(baseline["context_size"]),
        wp_length=int(baseline["wp_length"]),
        pose_step=int(baseline["pose_step"]),
        filter_cfg=cfg,
        output_path=str(out),
    )

    rebuilt = _load_npz(out)
    assert set(rebuilt) == set(baseline), "npz key set diverged"
    for key in baseline:
        assert np.array_equal(rebuilt[key], baseline[key]), f"{key} differs"
