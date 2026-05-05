"""Named scene presets."""

from __future__ import annotations

from typing import Callable

from projland.render import (
    Constellations,
    Glow,
    Halo,
    IdLabel,
    LetterLabel,
    OrientationArrow,
    Pulse,
    Radar,
    Scene,
    Sparkles,
    SpelledWord,
    Trail,
)


def _full(skip_ids: set[int]) -> Scene:
    return Scene(
        effects=[
            Trail(skip_ids=skip_ids),
            Radar(skip_ids=skip_ids),
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


def _minimal(skip_ids: set[int]) -> Scene:
    return Scene(
        effects=[
            Halo(skip_ids=skip_ids, thickness=4),
            IdLabel(skip_ids=skip_ids),
        ]
    )


def _spelling(skip_ids: set[int]) -> Scene:
    return Scene(
        effects=[
            Halo(skip_ids=skip_ids),
            LetterLabel(skip_ids=skip_ids),
            SpelledWord(skip_ids=skip_ids, font_scale=3.0),
            Glow(weight=0.3),
        ]
    )


def _stars(skip_ids: set[int]) -> Scene:
    return Scene(
        effects=[
            Constellations(skip_ids=skip_ids),
            Sparkles(skip_ids=skip_ids, count=12, particle_radius=4),
            Pulse(skip_ids=skip_ids, rings=4, period_sec=2.4),
            Glow(weight=0.6),
        ]
    )


PRESETS: dict[str, Callable[[set[int]], Scene]] = {
    "full": _full,
    "minimal": _minimal,
    "spelling": _spelling,
    "stars": _stars,
}


def build(name: str, skip_ids: set[int] | None = None) -> Scene:
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; available: {sorted(PRESETS)}")
    return PRESETS[name](skip_ids or set())
