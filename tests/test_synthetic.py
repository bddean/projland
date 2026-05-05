import numpy as np

from projland.markers import MarkerDetector
from projland.synthetic import PrintedMarker, SyntheticWorld


def test_printed_marker_detected_in_camera():
    world = SyntheticWorld()
    sw, sh = world.surface_size
    world.markers.append(PrintedMarker(id=7, center=(sw / 2, sh / 2), size=200.0))
    cam = world.render_camera()
    detected = MarkerDetector().detect(cam)
    ids = {m.id for m in detected}
    assert 7 in ids


def test_projection_visible_in_camera():
    """A bright projector image should make the camera image brighter on
    average than the un-projected version."""
    world = SyntheticWorld()
    sw, sh = world.surface_size
    world.markers.append(PrintedMarker(id=7, center=(sw / 2, sh / 2), size=200.0))
    cam_dark = world.render_camera()
    bright = np.full(
        (world.projector_size[1], world.projector_size[0], 3), 200, dtype=np.uint8
    )
    cam_lit = world.render_camera(projector_image=bright)
    assert cam_lit.mean() > cam_dark.mean() + 10
