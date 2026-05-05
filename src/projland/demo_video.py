"""Render a fully synthetic demo video, end-to-end, with no real hardware.

Useful as a smoke test of the whole pipeline: synthetic camera → detection
→ calibration → render → projector → composite back through the camera.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from projland.calibration import CALIBRATION_IDS, CalibrationPattern, solve_calibration
from projland.markers import MarkerDetector
from projland.presets import build as build_preset
from projland.render import Renderer
from projland.synthetic import PrintedMarker, SyntheticWorld


def build_world(t: float) -> SyntheticWorld:
    """Build a world with calibration corners + animated content markers."""
    world = SyntheticWorld()
    sw, sh = world.surface_size

    # Calibration corner markers near the *surface* corners (so they appear
    # near the projector's calibration positions in the synthetic camera).
    cw, ch = world.projector_size
    pat = CalibrationPattern(
        projector_size=world.projector_size,
        marker_size_px=180,
        margin_px=80,
    )
    proj_pts = pat.projector_points()  # in projector space
    proj_to_surface = np.linalg.inv(world.surface_to_projector)
    surface_pts = cv2.perspectiveTransform(
        proj_pts.reshape(1, -1, 2), proj_to_surface
    )[0]
    for marker_id, (x, y) in zip(CALIBRATION_IDS, surface_pts):
        world.markers.append(
            PrintedMarker(id=marker_id, center=(float(x), float(y)), size=140.0)
        )

    # A few content markers, moving in a circle.
    base_cx, base_cy = sw / 2, sh / 2
    radius = min(sw, sh) * 0.22
    content_ids = [7, 9, 41, 12]
    for i, mid in enumerate(content_ids):
        ang = t * 0.6 + i * (2 * math.pi / len(content_ids))
        x = base_cx + radius * math.cos(ang)
        y = base_cy + radius * math.sin(ang)
        world.markers.append(
            PrintedMarker(
                id=mid,
                center=(float(x), float(y)),
                size=110.0,
                rotation_deg=math.degrees(ang) * 0.2,
            )
        )

    return world, pat


def write_demo_video(output: Path, frames: int = 120, fps: int = 30, preset: str = "full") -> None:
    detector = MarkerDetector()

    world, pat = build_world(0.0)
    renderer = Renderer(projector_size=world.projector_size)

    # Initial calibration with no projection
    cam_pre = world.render_camera(projector_image=None)
    markers = detector.detect(cam_pre)
    cal = solve_calibration(pat, markers, world.camera_size)
    if cal is None:
        raise RuntimeError("synthetic calibration failed — corners not detected")

    cw, ch = world.camera_size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output), fourcc, fps, (cw, ch))

    skip_ids = set(pat.ids)
    scene = build_preset(preset, skip_ids=skip_ids)

    try:
        for i in range(frames):
            t = i / fps
            world, _ = build_world(t)
            # Stage 1: surface with no projection — used for detection ONLY in
            # frame 0; after that we keep the same calibration and re-detect
            # on the lit camera image.
            cam_lit = world.render_camera(projector_image=None)
            markers = detector.detect(cam_lit)
            content_markers = [m for m in markers if m.id not in skip_ids]
            proj_img = renderer.render(scene, content_markers, cal, t=t)
            cam_final = world.render_camera(projector_image=proj_img)
            writer.write(cam_final)
    finally:
        writer.release()
    print(f"Wrote {output} ({frames} frames @ {fps}fps)")


def write_demo_snapshot(output: Path, t: float = 1.5, preset: str = "full") -> None:
    """Render a single end-to-end frame at simulated time t."""
    detector = MarkerDetector()
    world, pat = build_world(0.0)
    renderer = Renderer(projector_size=world.projector_size)
    cam_pre = world.render_camera(projector_image=None)
    cal = solve_calibration(pat, detector.detect(cam_pre), world.camera_size)
    if cal is None:
        raise RuntimeError("synthetic calibration failed")
    skip_ids = set(pat.ids)
    scene = build_preset(preset, skip_ids=skip_ids)
    world, _ = build_world(t)
    cam_lit = world.render_camera(projector_image=None)
    markers = detector.detect(cam_lit)
    content = [m for m in markers if m.id not in skip_ids]
    proj_img = renderer.render(scene, content, cal, t=t)
    cam_final = world.render_camera(projector_image=proj_img)
    cv2.imwrite(str(output), cam_final)
    print(f"Wrote {output}")
