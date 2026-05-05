"""Synthetic projector + camera simulator for headless testing.

Models a flat surface with two views:
  * Projector: emits an image (the rendered scene)
  * Camera: observes the surface, sees both physical printed markers AND
    whatever the projector emits (additively combined)

Both views are related to the surface via separate homographies.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from projland.markers import render_marker


@dataclass
class PrintedMarker:
    """A physical printed ArUco marker placed on the surface."""

    id: int
    center: tuple[float, float]  # surface coords
    size: float                  # surface units
    rotation_deg: float = 0.0
    margin: float = 0.15         # white quiet zone fraction


@dataclass
class SyntheticWorld:
    """A flat surface with placed printed markers, viewed by a camera and
    illuminated by a projector. All in the same surface coordinate space.

    Surface is a [0, surface_size] x [0, surface_size] plane.
    """

    surface_size: tuple[int, int] = (1000, 700)
    camera_size: tuple[int, int] = (1280, 720)
    projector_size: tuple[int, int] = (1280, 800)
    surface_to_camera: np.ndarray = field(default_factory=lambda: None)
    surface_to_projector: np.ndarray = field(default_factory=lambda: None)
    markers: list[PrintedMarker] = field(default_factory=list)
    surface_color: tuple[int, int, int] = (220, 215, 205)  # warm off-white

    def __post_init__(self):
        if self.surface_to_camera is None:
            self.surface_to_camera = _surface_to_view_homography(
                self.surface_size,
                self.camera_size,
                # Slight perspective skew so it's not the trivial identity case
                offsets=((20, 30), (-25, 35), (-15, -40), (30, -20)),
            )
        if self.surface_to_projector is None:
            self.surface_to_projector = _surface_to_view_homography(
                self.surface_size,
                self.projector_size,
                offsets=((-10, 5), (15, -10), (-20, 8), (12, -8)),
            )

    # -- rendering ----------------------------------------------------

    def render_surface(self) -> np.ndarray:
        """Paint the surface with all printed markers (no projection)."""
        sw, sh = self.surface_size
        canvas = np.full((sh, sw, 3), self.surface_color, dtype=np.uint8)
        for m in self.markers:
            _stamp_marker(canvas, m)
        return canvas

    def render_camera(self, projector_image: np.ndarray | None = None) -> np.ndarray:
        """Render what the camera sees: surface + projected light, warped."""
        surface = self.render_surface().astype(np.float32)

        if projector_image is not None:
            # warp projector → surface
            proj_to_surface = np.linalg.inv(self.surface_to_projector)
            sw, sh = self.surface_size
            proj_on_surface = cv2.warpPerspective(
                projector_image.astype(np.float32),
                proj_to_surface,
                (sw, sh),
                flags=cv2.INTER_LINEAR,
                borderValue=(0, 0, 0),
            )
            # additively combine, simulating light addition (clip to 255)
            surface = np.clip(surface + proj_on_surface, 0, 255)

        # warp surface → camera
        cw, ch = self.camera_size
        camera_view = cv2.warpPerspective(
            surface,
            self.surface_to_camera,
            (cw, ch),
            flags=cv2.INTER_LINEAR,
            borderValue=(40, 40, 40),  # dark surroundings
        )
        return camera_view.astype(np.uint8)


def _surface_to_view_homography(
    surface_size: tuple[int, int],
    view_size: tuple[int, int],
    offsets: tuple[tuple[float, float], ...],
) -> np.ndarray:
    sw, sh = surface_size
    vw, vh = view_size
    src = np.array(
        [[0, 0], [sw, 0], [sw, sh], [0, sh]],
        dtype=np.float32,
    )
    # Map surface corners to view corners with small per-corner offsets to
    # introduce perspective distortion.
    margin_x = vw * 0.05
    margin_y = vh * 0.05
    base = np.array(
        [
            [margin_x, margin_y],
            [vw - margin_x, margin_y],
            [vw - margin_x, vh - margin_y],
            [margin_x, vh - margin_y],
        ],
        dtype=np.float32,
    )
    dst = base + np.array(offsets, dtype=np.float32)
    H, _ = cv2.findHomography(src, dst)
    return H


def _stamp_marker(canvas: np.ndarray, m: PrintedMarker) -> None:
    """Paint a printed marker (with quiet zone) onto the surface canvas."""
    s = int(round(m.size))
    if s < 8:
        s = 8
    marker_img = render_marker(m.id, s)
    quiet = max(2, int(round(s * m.margin)))
    full = s + 2 * quiet
    tile = np.full((full, full, 3), 255, dtype=np.uint8)
    tile[quiet : quiet + s, quiet : quiet + s] = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
    if abs(m.rotation_deg) > 0.01:
        center = (full / 2, full / 2)
        rot = cv2.getRotationMatrix2D(center, m.rotation_deg, 1.0)
        tile = cv2.warpAffine(tile, rot, (full, full), borderValue=(255, 255, 255))

    cx, cy = m.center
    x0 = int(round(cx - full / 2))
    y0 = int(round(cy - full / 2))
    h, w = canvas.shape[:2]
    x1, y1 = x0 + full, y0 + full
    # clip
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(w, x1), min(h, y1)
    if sx0 >= sx1 or sy0 >= sy1:
        return
    tx0, ty0 = sx0 - x0, sy0 - y0
    tx1, ty1 = tx0 + (sx1 - sx0), ty0 + (sy1 - sy0)
    canvas[sy0:sy1, sx0:sx1] = tile[ty0:ty1, tx0:tx1]
