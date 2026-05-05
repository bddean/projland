import numpy as np
import pytest

from projland.calibration import identity_calibration
from projland.markers import Marker
from projland.presets import PRESETS, build
from projland.render import Renderer


def _marker(mid, x, y, s=60):
    corners = np.array(
        [[x, y], [x + s, y], [x + s, y + s], [x, y + s]], dtype=np.float32
    )
    return Marker(id=mid, corners=corners)


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_preset_renders(name):
    cal = identity_calibration((640, 480))
    r = Renderer(projector_size=(640, 480))
    scene = build(name)
    img = r.render(
        scene,
        [_marker(7, 100, 100), _marker(41, 200, 100), _marker(12, 300, 100)],
        cal,
        t=0.5,
    )
    assert img.shape == (480, 640, 3)


def test_unknown_preset_raises():
    with pytest.raises(ValueError):
        build("nonexistent")
