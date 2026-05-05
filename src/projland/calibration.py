"""Auto-calibration of the projector→camera transform.

The model: assume both projector and camera see a planar surface (the table /
wall). Then there is a single homography mapping projector pixel coordinates
to camera pixel coordinates (and its inverse).

Method: project a calibration pattern of 4 ArUco markers near the corners of
the projector image. The camera observes them. We compute the homography from
the known projector-space marker centers to the observed camera-space centers.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from projland.markers import Marker, MarkerDetector, render_marker


# IDs we use for calibration corners — large values chosen so they don't
# collide with typical "content" markers (which tend to be small numbers in
# the printed-out kit).
CALIBRATION_IDS = (200, 201, 202, 203)  # TL, TR, BR, BL


@dataclass(frozen=True)
class CalibrationPattern:
    """A renderable calibration pattern."""

    projector_size: tuple[int, int]  # (width, height)
    marker_size_px: int
    margin_px: int
    ids: tuple[int, int, int, int] = CALIBRATION_IDS

    def render(self) -> np.ndarray:
        """Render the calibration image (BGR)."""
        w, h = self.projector_size
        img = np.zeros((h, w, 3), dtype=np.uint8)
        s = self.marker_size_px
        m = self.margin_px
        positions = self._projector_positions()
        for marker_id, (x, y) in zip(self.ids, positions):
            patch = render_marker(marker_id, s)
            x0 = int(round(x - s / 2))
            y0 = int(round(y - s / 2))
            patch_bgr = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
            # Invert: most projectors look better with white background for
            # markers? Actually ArUco needs black on white — keep as-is on
            # black background but draw a white quiet zone.
            quiet = max(2, s // 8)
            img[y0 - quiet : y0 + s + quiet, x0 - quiet : x0 + s + quiet] = 255
            img[y0 : y0 + s, x0 : x0 + s] = patch_bgr
        return img

    def _projector_positions(self) -> list[tuple[float, float]]:
        w, h = self.projector_size
        m = self.margin_px
        s = self.marker_size_px
        # centers of markers
        cx_l = m + s / 2
        cx_r = w - m - s / 2
        cy_t = m + s / 2
        cy_b = h - m - s / 2
        return [(cx_l, cy_t), (cx_r, cy_t), (cx_r, cy_b), (cx_l, cy_b)]

    def projector_points(self) -> np.ndarray:
        """Return projector-space (x, y) centers for the 4 ids, shape (4, 2)."""
        return np.array(self._projector_positions(), dtype=np.float32)


@dataclass(frozen=True)
class Calibration:
    """Homography between projector pixel space and camera pixel space."""

    projector_to_camera: np.ndarray  # (3, 3) float64
    camera_to_projector: np.ndarray  # (3, 3) float64
    projector_size: tuple[int, int]
    camera_size: tuple[int, int]

    def project_to_camera(self, points: np.ndarray) -> np.ndarray:
        """Map projector-space (N, 2) points to camera-space."""
        return _apply_homography(self.projector_to_camera, points)

    def camera_to_projector_pts(self, points: np.ndarray) -> np.ndarray:
        return _apply_homography(self.camera_to_projector, points)


def _apply_homography(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    homo = np.hstack([pts, ones])
    out = (H @ homo.T).T
    out = out[:, :2] / out[:, 2:3]
    return out.astype(np.float32)


def identity_calibration(camera_size: tuple[int, int]) -> "Calibration":
    """A pass-through calibration: projector space *is* camera space.

    Useful for the preview/no-projector mode where we composite effects on
    top of the camera feed and show it in a window.
    """
    eye = np.eye(3, dtype=np.float64)
    return Calibration(
        projector_to_camera=eye,
        camera_to_projector=eye,
        projector_size=camera_size,
        camera_size=camera_size,
    )


def solve_calibration(
    pattern: CalibrationPattern,
    observed: list[Marker],
    camera_size: tuple[int, int],
) -> Calibration | None:
    """Compute a Calibration given the observed markers.

    Returns None if not all 4 calibration markers were observed.
    """
    by_id = {m.id: m for m in observed}
    if not all(i in by_id for i in pattern.ids):
        return None

    projector_pts = pattern.projector_points()
    camera_pts = np.array([by_id[i].center for i in pattern.ids], dtype=np.float32)

    H, _ = cv2.findHomography(projector_pts, camera_pts, method=0)
    if H is None:
        return None
    H_inv = np.linalg.inv(H)
    return Calibration(
        projector_to_camera=H,
        camera_to_projector=H_inv,
        projector_size=pattern.projector_size,
        camera_size=camera_size,
    )
