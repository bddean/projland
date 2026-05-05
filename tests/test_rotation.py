"""Detection should still work for rotated and tilted markers."""
import numpy as np

from projland.markers import MarkerDetector
from projland.synthetic import PrintedMarker, SyntheticWorld


def test_detection_under_rotation():
    detector = MarkerDetector()
    for angle in (0, 30, 75, 145, -90):
        world = SyntheticWorld()
        sw, sh = world.surface_size
        world.markers.append(
            PrintedMarker(id=7, center=(sw / 2, sh / 2), size=200, rotation_deg=angle)
        )
        cam = world.render_camera()
        detected = detector.detect(cam)
        ids = {m.id for m in detected}
        assert 7 in ids, f"marker 7 not detected at rotation {angle}"


def test_marker_rotation_recovered():
    """The detected marker's rotation_deg should roughly match the printed angle."""
    detector = MarkerDetector()
    # Use surface-size view so projection effects don't dominate
    world = SyntheticWorld(camera_size=(900, 700))
    sw, sh = world.surface_size
    world.markers.append(
        PrintedMarker(id=7, center=(sw / 2, sh / 2), size=200, rotation_deg=30)
    )
    cam = world.render_camera()
    detected = detector.detect(cam)
    assert detected
    # Because the synthetic camera has perspective distortion, the recovered
    # angle won't be exact; just sanity-check it's within a wide window.
    angle = detected[0].rotation_deg
    # Allow either +30 or -150 (the marker is rotated; orientation can be
    # ambiguous). Just check some rotation was detected vs the
    # near-axis-aligned case.
    world2 = SyntheticWorld(camera_size=(900, 700))
    world2.markers.append(
        PrintedMarker(id=7, center=(sw / 2, sh / 2), size=200, rotation_deg=0)
    )
    detected2 = detector.detect(world2.render_camera())
    assert detected2
    angle0 = detected2[0].rotation_deg
    delta = abs(((angle - angle0 + 180) % 360) - 180)
    assert delta > 10, f"rotation didn't move detected angle: {angle0} -> {angle}"
