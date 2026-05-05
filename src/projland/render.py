"""Render the projector image given detected markers and a calibration."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Protocol

import cv2
import numpy as np

from projland.calibration import Calibration
from projland.markers import Marker


# Pleasant high-saturation palette (BGR).
PALETTE_BGR = [
    (255, 80, 80),     # blue-ish
    (80, 255, 80),     # green
    (80, 80, 255),     # red
    (80, 255, 255),    # yellow
    (255, 80, 255),    # magenta
    (255, 255, 80),    # cyan
    (80, 180, 255),    # orange-red
    (180, 80, 255),    # pink-violet
]


def color_for_id(marker_id: int) -> tuple[int, int, int]:
    return PALETTE_BGR[marker_id % len(PALETTE_BGR)]


class Effect(Protocol):
    def draw(self, canvas: np.ndarray, markers: list[Marker], cal: Calibration, t: float) -> None: ...


@dataclass
class Scene:
    """A composable scene of projector-space effects."""

    effects: list[Effect] = field(default_factory=list)

    def add(self, effect: Effect) -> "Scene":
        self.effects.append(effect)
        return self


@dataclass
class Renderer:
    projector_size: tuple[int, int]  # (width, height)
    background: tuple[int, int, int] = (0, 0, 0)

    def render(
        self,
        scene: Scene,
        markers: list[Marker],
        calibration: Calibration,
        t: float = 0.0,
    ) -> np.ndarray:
        w, h = self.projector_size
        canvas = np.full((h, w, 3), self.background, dtype=np.uint8)
        for effect in scene.effects:
            effect.draw(canvas, markers, calibration, t)
        return canvas


# -- camera→projector mapping helpers --------------------------------------

def marker_corners_in_projector(marker: Marker, cal: Calibration) -> np.ndarray:
    return cal.camera_to_projector_pts(marker.corners)


def marker_center_in_projector(marker: Marker, cal: Calibration) -> np.ndarray:
    pts = cal.camera_to_projector_pts(marker.corners)
    return pts.mean(axis=0)


# -- effects ----------------------------------------------------------------


@dataclass
class Halo:
    """Draw a ring around each marker."""

    radius_scale: float = 1.6
    thickness: int = 6
    skip_ids: set[int] = field(default_factory=set)

    def draw(self, canvas, markers, cal, t):
        for m in markers:
            if m.id in self.skip_ids:
                continue
            corners = marker_corners_in_projector(m, cal)
            center = corners.mean(axis=0)
            # radius derived from marker side in projector space
            side = float(np.linalg.norm(corners[1] - corners[0]))
            r = max(8, int(side * self.radius_scale / 2))
            cv2.circle(
                canvas,
                (int(center[0]), int(center[1])),
                r,
                color_for_id(m.id),
                self.thickness,
                lineType=cv2.LINE_AA,
            )


@dataclass
class IdLabel:
    """Render the marker's ID near it."""

    offset: tuple[int, int] = (0, -30)
    font_scale: float = 0.9
    thickness: int = 2
    skip_ids: set[int] = field(default_factory=set)

    def draw(self, canvas, markers, cal, t):
        for m in markers:
            if m.id in self.skip_ids:
                continue
            center = marker_center_in_projector(m, cal)
            x, y = int(center[0]) + self.offset[0], int(center[1]) + self.offset[1]
            text = str(m.id)
            color = color_for_id(m.id)
            cv2.putText(
                canvas,
                text,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                color,
                self.thickness,
                cv2.LINE_AA,
            )


@dataclass
class Constellations:
    """Connect every pair of markers with a colored line."""

    thickness: int = 2
    skip_ids: set[int] = field(default_factory=set)

    def draw(self, canvas, markers, cal, t):
        active = [m for m in markers if m.id not in self.skip_ids]
        for i, a in enumerate(active):
            ca = marker_center_in_projector(a, cal)
            for b in active[i + 1 :]:
                cb = marker_center_in_projector(b, cal)
                color = color_for_id((a.id + b.id) % 256)
                cv2.line(
                    canvas,
                    (int(ca[0]), int(ca[1])),
                    (int(cb[0]), int(cb[1])),
                    color,
                    self.thickness,
                    cv2.LINE_AA,
                )


@dataclass
class Pulse:
    """Pulsing concentric rings around each marker (animated)."""

    rings: int = 3
    period_sec: float = 1.6
    base_scale: float = 1.2
    growth: float = 0.9
    thickness: int = 2
    skip_ids: set[int] = field(default_factory=set)

    def draw(self, canvas, markers, cal, t):
        phase = (t % self.period_sec) / self.period_sec  # 0..1
        for m in markers:
            if m.id in self.skip_ids:
                continue
            corners = marker_corners_in_projector(m, cal)
            center = corners.mean(axis=0)
            side = float(np.linalg.norm(corners[1] - corners[0]))
            color = color_for_id(m.id)
            for i in range(self.rings):
                local_phase = (phase + i / self.rings) % 1.0
                scale = self.base_scale + local_phase * self.growth
                r = max(4, int(side * scale / 2))
                alpha = 1.0 - local_phase  # fade as it grows
                ring_color = tuple(int(c * alpha) for c in color)
                cv2.circle(
                    canvas,
                    (int(center[0]), int(center[1])),
                    r,
                    ring_color,
                    self.thickness,
                    cv2.LINE_AA,
                )


def default_scene(skip_ids: set[int] | None = None) -> Scene:
    skip_ids = skip_ids or set()
    return Scene(
        effects=[
            Constellations(skip_ids=skip_ids),
            Halo(skip_ids=skip_ids),
            Pulse(skip_ids=skip_ids),
            IdLabel(skip_ids=skip_ids),
        ]
    )
