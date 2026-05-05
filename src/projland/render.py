"""Render the projector image given detected markers and a calibration."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Protocol

import cv2
import numpy as np

from projland.calibration import Calibration
from projland.letters import ARCUO_TO_LETTER
from projland.markers import Marker
from projland.spelling import group_words


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
    """Connect markers with a colored line.

    If `max_distance_px` is set, only connect markers whose centers (in
    projector space) are within that distance — handy when you have many
    markers and don't want a hairball.
    """

    thickness: int = 2
    max_distance_px: float | None = None
    skip_ids: set[int] = field(default_factory=set)

    def draw(self, canvas, markers, cal, t):
        active = [m for m in markers if m.id not in self.skip_ids]
        centers = [marker_center_in_projector(m, cal) for m in active]
        for i, a in enumerate(active):
            ca = centers[i]
            for j in range(i + 1, len(active)):
                cb = centers[j]
                if self.max_distance_px is not None:
                    dist = float(np.linalg.norm(ca - cb))
                    if dist > self.max_distance_px:
                        continue
                color = color_for_id((a.id + active[j].id) % 256)
                cv2.line(
                    canvas,
                    (int(ca[0]), int(ca[1])),
                    (int(cb[0]), int(cb[1])),
                    color,
                    self.thickness,
                    cv2.LINE_AA,
                )


@dataclass
class LetterLabel:
    """If a marker has a known letter mapping, render that letter above it."""

    offset: tuple[int, int] = (0, -50)
    font_scale: float = 1.6
    thickness: int = 3
    skip_ids: set[int] = field(default_factory=set)

    def draw(self, canvas, markers, cal, t):
        for m in markers:
            if m.id in self.skip_ids:
                continue
            letter = ARCUO_TO_LETTER.get(m.id)
            if letter is None:
                continue
            center = marker_center_in_projector(m, cal)
            x = int(center[0]) + self.offset[0]
            y = int(center[1]) + self.offset[1]
            color = color_for_id(m.id)
            cv2.putText(
                canvas,
                letter,
                (x, y),
                cv2.FONT_HERSHEY_DUPLEX,
                self.font_scale,
                color,
                self.thickness,
                cv2.LINE_AA,
            )


@dataclass
class OrientationArrow:
    """Draw an arrow from each marker's center along its top edge."""

    length_scale: float = 1.4
    thickness: int = 4
    skip_ids: set[int] = field(default_factory=set)

    def draw(self, canvas, markers, cal, t):
        for m in markers:
            if m.id in self.skip_ids:
                continue
            corners = marker_corners_in_projector(m, cal)
            tl, tr, _, _ = corners
            center = corners.mean(axis=0)
            direction = (tr + tl) / 2 - center  # midpoint of top edge − center
            # We actually want the arrow to point AWAY from the marker — flip
            # so it points "up" relative to the marker's frame.
            direction = -direction * self.length_scale
            tip = center + direction
            cv2.arrowedLine(
                canvas,
                (int(center[0]), int(center[1])),
                (int(tip[0]), int(tip[1])),
                color_for_id(m.id),
                self.thickness,
                cv2.LINE_AA,
                tipLength=0.3,
            )


@dataclass
class Trail:
    """Per-id breadcrumb trail of recent center positions."""

    length: int = 12
    thickness: int = 3
    skip_ids: set[int] = field(default_factory=set)
    _history: dict[int, list[tuple[float, float]]] = field(default_factory=dict)

    def draw(self, canvas, markers, cal, t):
        seen = set()
        for m in markers:
            if m.id in self.skip_ids:
                continue
            seen.add(m.id)
            center = marker_center_in_projector(m, cal)
            hist = self._history.setdefault(m.id, [])
            hist.append((float(center[0]), float(center[1])))
            if len(hist) > self.length:
                del hist[: len(hist) - self.length]
            color = color_for_id(m.id)
            for i in range(1, len(hist)):
                # fade older segments
                alpha = i / len(hist)
                seg_color = tuple(int(c * alpha) for c in color)
                cv2.line(
                    canvas,
                    (int(hist[i - 1][0]), int(hist[i - 1][1])),
                    (int(hist[i][0]), int(hist[i][1])),
                    seg_color,
                    self.thickness,
                    cv2.LINE_AA,
                )
        # decay trails for ids not seen this frame
        for mid in list(self._history.keys()):
            if mid not in seen:
                self._history[mid] = self._history[mid][1:]
                if not self._history[mid]:
                    self._history.pop(mid)


@dataclass
class SpelledWord:
    """If two or more letter markers are roughly aligned in a row, render
    the spelled word above them."""

    font_scale: float = 2.2
    thickness: int = 4
    min_letters: int = 2
    color: tuple[int, int, int] = (255, 255, 255)
    skip_ids: set[int] = field(default_factory=set)

    def draw(self, canvas, markers, cal, t):
        active = [m for m in markers if m.id not in self.skip_ids]
        words = group_words(active)
        for w in words:
            if len(w.text) < self.min_letters:
                continue
            # Compute centroid in projector space
            centers = [marker_center_in_projector(m, cal) for m in w.markers]
            cx = float(np.mean([c[0] for c in centers]))
            cy = float(np.mean([c[1] for c in centers]))
            # Place text above the row
            sizes = [
                float(np.linalg.norm(marker_corners_in_projector(m, cal)[1] -
                                     marker_corners_in_projector(m, cal)[0]))
                for m in w.markers
            ]
            avg_size = float(np.mean(sizes))
            tx = int(cx)
            ty = int(cy - avg_size * 1.4)
            (tw, th), _ = cv2.getTextSize(
                w.text, cv2.FONT_HERSHEY_DUPLEX, self.font_scale, self.thickness
            )
            cv2.putText(
                canvas,
                w.text,
                (tx - tw // 2, ty),
                cv2.FONT_HERSHEY_DUPLEX,
                self.font_scale,
                self.color,
                self.thickness,
                cv2.LINE_AA,
            )


@dataclass
class Sparkles:
    """Rotating colored particles around each marker."""

    count: int = 8
    radius_scale: float = 1.9
    particle_radius: int = 5
    angular_speed: float = 1.2  # radians/sec
    skip_ids: set[int] = field(default_factory=set)

    def draw(self, canvas, markers, cal, t):
        for m in markers:
            if m.id in self.skip_ids:
                continue
            corners = marker_corners_in_projector(m, cal)
            center = corners.mean(axis=0)
            side = float(np.linalg.norm(corners[1] - corners[0]))
            r = side * self.radius_scale / 2
            base_angle = t * self.angular_speed + m.id * 0.37
            color = color_for_id(m.id)
            for i in range(self.count):
                ang = base_angle + i * (2 * math.pi / self.count)
                x = int(center[0] + r * math.cos(ang))
                y = int(center[1] + r * math.sin(ang))
                cv2.circle(canvas, (x, y), self.particle_radius, color, -1, cv2.LINE_AA)


@dataclass
class Glow:
    """Apply a Gaussian blur over the rendered canvas, then add it back, to
    give a bloom effect. Should be added LAST in a scene."""

    kernel: int = 31
    sigma: float = 9.0
    weight: float = 0.7

    def draw(self, canvas, markers, cal, t):
        if self.kernel % 2 == 0:
            self.kernel += 1
        blurred = cv2.GaussianBlur(canvas, (self.kernel, self.kernel), self.sigma)
        boost = (blurred.astype(np.float32) * self.weight).clip(0, 255).astype(np.uint8)
        # saturate-add (no wraparound)
        np.copyto(canvas, cv2.add(canvas, boost))


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
            Trail(skip_ids=skip_ids),
            Constellations(skip_ids=skip_ids, max_distance_px=600),
            Halo(skip_ids=skip_ids),
            Pulse(skip_ids=skip_ids),
            Sparkles(skip_ids=skip_ids),
            OrientationArrow(skip_ids=skip_ids),
            LetterLabel(skip_ids=skip_ids),
            SpelledWord(skip_ids=skip_ids),
            IdLabel(skip_ids=skip_ids, offset=(0, 28), font_scale=0.5),
            Glow(weight=0.35),
        ]
    )
