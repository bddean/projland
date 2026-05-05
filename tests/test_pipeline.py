"""End-to-end pipeline test: synthetic camera → detect → calibrate → render
→ project → composite back through the camera, then verify that the
projected halo lands near the marker."""

from __future__ import annotations

import cv2
import numpy as np

from projland.calibration import CALIBRATION_IDS, CalibrationPattern, solve_calibration
from projland.markers import MarkerDetector
from projland.render import Halo, Renderer, Scene
from projland.synthetic import PrintedMarker, SyntheticWorld


def _build_world():
    pat = CalibrationPattern(projector_size=(1280, 800), marker_size_px=180, margin_px=80)
    world = SyntheticWorld()
    proj_to_surface = np.linalg.inv(world.surface_to_projector)
    proj_pts = pat.projector_points()
    surface_pts = cv2.perspectiveTransform(proj_pts.reshape(1, -1, 2), proj_to_surface)[0]
    for marker_id, (x, y) in zip(pat.ids, surface_pts):
        world.markers.append(
            PrintedMarker(id=marker_id, center=(float(x), float(y)), size=130.0)
        )
    # content marker in the middle of the surface
    sw, sh = world.surface_size
    content_id = 41
    world.markers.append(
        PrintedMarker(id=content_id, center=(sw / 2, sh / 2), size=140.0)
    )
    return world, pat, content_id


def test_halo_lands_on_marker():
    world, pat, content_id = _build_world()
    detector = MarkerDetector()

    cam_pre = world.render_camera(projector_image=None)
    detected_pre = detector.detect(cam_pre)
    cal = solve_calibration(pat, detected_pre, world.camera_size)
    assert cal is not None

    content_marker = next(m for m in detected_pre if m.id == content_id)
    expected_center = content_marker.center  # in camera space

    renderer = Renderer(projector_size=world.projector_size)
    # Use just a halo, plain bright color, so we can find it visually.
    halo = Halo(radius_scale=1.6, thickness=18)
    scene = Scene(effects=[halo])
    proj_img = renderer.render(scene, [content_marker], cal)

    cam_lit = world.render_camera(projector_image=proj_img)

    # Compute a "where the projector added the most light" image: difference
    # between lit camera and unlit camera.
    diff = cv2.absdiff(cam_lit, cam_pre).astype(np.float32).sum(axis=2)
    # Threshold and find the centroid of the brightest blob.
    mask = (diff > 60).astype(np.uint8)
    assert mask.sum() > 0, "projector light should be visible in camera diff"

    ys, xs = np.where(mask > 0)
    cx = float(xs.mean())
    cy = float(ys.mean())

    err = float(np.linalg.norm(np.array([cx, cy]) - expected_center))
    assert err < 30.0, f"halo centroid {cx:.1f},{cy:.1f} vs marker {expected_center}: err={err:.1f}"


def test_default_scene_runs_without_error():
    from projland.render import default_scene

    world, pat, content_id = _build_world()
    detector = MarkerDetector()
    cam = world.render_camera(projector_image=None)
    detected = detector.detect(cam)
    cal = solve_calibration(pat, detected, world.camera_size)
    assert cal is not None
    skip = set(pat.ids)
    content = [m for m in detected if m.id not in skip]
    renderer = Renderer(projector_size=world.projector_size)
    scene = default_scene(skip_ids=skip)
    img = renderer.render(scene, content, cal, t=0.5)
    assert img.shape == (world.projector_size[1], world.projector_size[0], 3)
    # Some pixels should be nonzero
    assert img.sum() > 0
