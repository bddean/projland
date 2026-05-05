import numpy as np

from projland.markers import Marker
from projland.tracking import MarkerSmoother


def _marker(mid: int, x: float, y: float, s: float = 50.0) -> Marker:
    corners = np.array(
        [[x, y], [x + s, y], [x + s, y + s], [x, y + s]], dtype=np.float32
    )
    return Marker(id=mid, corners=corners)


def test_smoother_returns_input_on_first_seen():
    sm = MarkerSmoother(alpha=0.5)
    out = sm.update([_marker(7, 100, 100)])
    assert len(out) == 1
    assert out[0].id == 7
    np.testing.assert_allclose(out[0].center, (125, 125))


def test_smoother_blends_over_time():
    sm = MarkerSmoother(alpha=0.5)
    sm.update([_marker(7, 100, 100)])
    out = sm.update([_marker(7, 200, 100)])
    # Halfway between 100 and 200 corner = 150 → center 175
    np.testing.assert_allclose(out[0].center, (175, 125))


def test_smoother_forgets_after_misses():
    sm = MarkerSmoother(alpha=1.0, forget_after_misses=2)
    sm.update([_marker(7, 100, 100)])
    sm.update([])
    sm.update([])
    sm.update([])  # third miss — should forget
    assert 7 not in sm._corners
