import numpy as np

from projland.calibration import CalibrationPattern, solve_calibration
from projland.markers import MarkerDetector
from projland.synthetic import PrintedMarker, SyntheticWorld


def _world_with_calibration_corners(pat: CalibrationPattern) -> SyntheticWorld:
    """Place printed markers on the synthetic surface at locations that
    correspond to the calibration pattern's projector positions."""
    import cv2

    world = SyntheticWorld()
    proj_pts = pat.projector_points()
    proj_to_surface = np.linalg.inv(world.surface_to_projector)
    surface_pts = cv2.perspectiveTransform(
        proj_pts.reshape(1, -1, 2), proj_to_surface
    )[0]
    for marker_id, (x, y) in zip(pat.ids, surface_pts):
        world.markers.append(
            PrintedMarker(id=marker_id, center=(float(x), float(y)), size=130.0)
        )
    return world


def test_solve_calibration_round_trip():
    pat = CalibrationPattern(
        projector_size=(1280, 800), marker_size_px=180, margin_px=80
    )
    world = _world_with_calibration_corners(pat)
    cam = world.render_camera(projector_image=None)
    detector = MarkerDetector()
    detected = detector.detect(cam)
    cal = solve_calibration(pat, detected, world.camera_size)
    assert cal is not None

    # Project the projector-space corner centers into camera space and check
    # they roughly match the detected centers.
    detected_by_id = {m.id: m for m in detected}
    proj_pts = pat.projector_points()
    cam_pts_predicted = cal.project_to_camera(proj_pts)
    for i, mid in enumerate(pat.ids):
        observed = detected_by_id[mid].center
        pred = cam_pts_predicted[i]
        err = float(np.linalg.norm(observed - pred))
        assert err < 3.0, f"calibration error for id {mid}: {err:.2f}px"


def test_solve_calibration_returns_none_with_missing_corners():
    pat = CalibrationPattern(
        projector_size=(1280, 800), marker_size_px=180, margin_px=80
    )
    world = _world_with_calibration_corners(pat)
    # remove one
    world.markers.pop()
    cam = world.render_camera(projector_image=None)
    detected = MarkerDetector().detect(cam)
    cal = solve_calibration(pat, detected, world.camera_size)
    assert cal is None
