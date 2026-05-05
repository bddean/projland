"""ArUco marker detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np


DEFAULT_DICT = cv2.aruco.DICT_ARUCO_ORIGINAL


@dataclass(frozen=True)
class Marker:
    """A single detected marker.

    `corners` are 4x2 float pixel coordinates in the source image, in the
    OpenCV order: top-left, top-right, bottom-right, bottom-left (relative to
    the marker's own frame, so they rotate with the marker).
    """

    id: int
    corners: np.ndarray  # (4, 2) float32

    @property
    def center(self) -> np.ndarray:
        return self.corners.mean(axis=0)

    @property
    def rotation_deg(self) -> float:
        """Rotation of the marker in degrees, 0 = top edge horizontal."""
        tl, tr, _, _ = self.corners
        dx, dy = tr - tl
        return float(np.degrees(np.arctan2(dy, dx)))

    @property
    def size_px(self) -> float:
        """Approximate marker side length in pixels."""
        tl, tr, br, bl = self.corners
        sides = [
            np.linalg.norm(tr - tl),
            np.linalg.norm(br - tr),
            np.linalg.norm(bl - br),
            np.linalg.norm(tl - bl),
        ]
        return float(np.mean(sides))


class MarkerDetector:
    """Wraps cv2.aruco.ArucoDetector with a tidier API."""

    def __init__(self, dictionary: int = DEFAULT_DICT) -> None:
        self._aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)
        self._params = cv2.aruco.DetectorParameters()
        self._detector = cv2.aruco.ArucoDetector(self._aruco_dict, self._params)

    @property
    def dictionary(self):
        return self._aruco_dict

    def detect(self, image: np.ndarray) -> list[Marker]:
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        corners, ids, _ = self._detector.detectMarkers(gray)
        if ids is None:
            return []
        out: list[Marker] = []
        for marker_corners, marker_id in zip(corners, ids):
            out.append(
                Marker(id=int(marker_id[0]), corners=marker_corners[0].astype(np.float32))
            )
        return out


def render_marker(
    marker_id: int,
    size_px: int,
    dictionary: int = DEFAULT_DICT,
    border_bits: int = 1,
) -> np.ndarray:
    """Render a single ArUco marker as a grayscale image (size_px x size_px)."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)
    return cv2.aruco.generateImageMarker(aruco_dict, marker_id, size_px, borderBits=border_bits)


def select_present_ids(markers: Iterable[Marker]) -> set[int]:
    return {m.id for m in markers}
