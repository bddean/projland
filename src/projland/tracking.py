"""Smooth marker positions across frames to reduce jitter."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from projland.markers import Marker


@dataclass
class MarkerSmoother:
    """Exponential moving average over marker corner positions, keyed by id.

    Markers that go missing for longer than `forget_after_misses` frames are
    dropped so they don't re-appear stale when they return.
    """

    alpha: float = 0.4   # higher = more responsive, lower = smoother
    forget_after_misses: int = 5
    _corners: dict[int, np.ndarray] = field(default_factory=dict)
    _miss: dict[int, int] = field(default_factory=dict)

    def update(self, markers: list[Marker]) -> list[Marker]:
        present_ids = {m.id for m in markers}
        out: list[Marker] = []
        for m in markers:
            prev = self._corners.get(m.id)
            if prev is None:
                smoothed = m.corners.astype(np.float32)
            else:
                smoothed = (self.alpha * m.corners + (1 - self.alpha) * prev).astype(np.float32)
            self._corners[m.id] = smoothed
            self._miss[m.id] = 0
            out.append(Marker(id=m.id, corners=smoothed))

        # bump miss counters for absent ids
        for mid in list(self._corners.keys()):
            if mid in present_ids:
                continue
            self._miss[mid] = self._miss.get(mid, 0) + 1
            if self._miss[mid] > self.forget_after_misses:
                self._corners.pop(mid, None)
                self._miss.pop(mid, None)
        return out
