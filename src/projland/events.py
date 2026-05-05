"""Marker arrival/departure events with debouncing.

Inspired by the existing opencv_ipcam_detector debouncing logic. A marker
that flickers in/out for less than `debounce_sec` shouldn't generate spurious
events.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from projland.markers import Marker


@dataclass
class MarkerEvents:
    debounce_sec: float = 0.5
    on_arrive: Callable[[Marker], None] | None = None
    on_depart: Callable[[int], None] | None = None
    _last_seen: dict[int, float] = field(default_factory=dict)
    _is_present: set[int] = field(default_factory=set)
    _now: Callable[[], float] = field(default=time.monotonic)

    def update(self, markers: list[Marker]) -> None:
        now = self._now()
        present_ids = {m.id for m in markers}
        by_id = {m.id: m for m in markers}

        # arrivals
        for mid in present_ids - self._is_present:
            self._is_present.add(mid)
            if self.on_arrive is not None:
                self.on_arrive(by_id[mid])
        # update last-seen
        for mid in present_ids:
            self._last_seen[mid] = now
        # debounced departures
        for mid in list(self._is_present - present_ids):
            last = self._last_seen.get(mid, 0)
            if now - last >= self.debounce_sec:
                self._is_present.discard(mid)
                self._last_seen.pop(mid, None)
                if self.on_depart is not None:
                    self.on_depart(mid)
