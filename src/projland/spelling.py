"""Group letter-bearing markers into rows of words.

A row is a left-to-right sequence of markers whose centers are roughly
co-linear horizontally, with consistent spacing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from projland.letters import ARCUO_TO_LETTER
from projland.markers import Marker


@dataclass
class Word:
    text: str
    markers: list[Marker]

    @property
    def center(self) -> np.ndarray:
        return np.mean([m.center for m in self.markers], axis=0)

    @property
    def height(self) -> float:
        return float(np.mean([m.size_px for m in self.markers]))


def group_words(
    markers: list[Marker],
    row_tolerance: float = 0.7,    # multiples of marker size
    gap_tolerance: float = 2.5,    # multiples of marker size
) -> list[Word]:
    """Group letter-bearing markers into words by row+spacing."""
    letter_markers = [m for m in markers if m.id in ARCUO_TO_LETTER]
    if not letter_markers:
        return []

    # Sort by y, then x.
    sorted_m = sorted(letter_markers, key=lambda m: (m.center[1], m.center[0]))

    rows: list[list[Marker]] = []
    for m in sorted_m:
        cy = m.center[1]
        size = m.size_px
        placed = False
        for row in rows:
            row_cy = float(np.mean([r.center[1] for r in row]))
            row_size = float(np.mean([r.size_px for r in row]))
            if abs(cy - row_cy) <= row_tolerance * row_size:
                row.append(m)
                placed = True
                break
        if not placed:
            rows.append([m])

    words: list[Word] = []
    for row in rows:
        row.sort(key=lambda m: m.center[0])
        # split on big gaps
        current: list[Marker] = []
        for m in row:
            if not current:
                current.append(m)
                continue
            prev = current[-1]
            gap = m.center[0] - prev.center[0]
            allowed = gap_tolerance * np.mean([prev.size_px, m.size_px])
            if gap > allowed:
                if len(current) >= 1:
                    words.append(_finish_word(current))
                current = [m]
            else:
                current.append(m)
        if current:
            words.append(_finish_word(current))
    return words


def _finish_word(group: list[Marker]) -> Word:
    text = "".join(ARCUO_TO_LETTER[m.id] for m in group)
    return Word(text=text, markers=list(group))
