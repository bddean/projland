"""macOS display enumeration via Quartz.

Returns each connected display's pixel bounds in the global coordinate space,
which is what `cv2.moveWindow` expects.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Display:
    id: int
    x: int
    y: int
    width: int
    height: int
    is_main: bool
    is_builtin: bool

    @property
    def label(self) -> str:
        tags = []
        if self.is_main:
            tags.append("main")
        if self.is_builtin:
            tags.append("builtin")
        else:
            tags.append("external")
        tag = ",".join(tags)
        return f"#{self.id} [{tag}] {self.width}x{self.height}@({self.x},{self.y})"


def list_displays() -> list[Display]:
    """List active displays. macOS only (returns [] elsewhere or if Quartz
    isn't installed)."""
    if sys.platform != "darwin":
        return []
    try:
        import Quartz  # type: ignore
    except ImportError:
        return []
    max_displays = 16
    err, ids, count = Quartz.CGGetActiveDisplayList(max_displays, None, None)
    if err != 0:
        return []
    main = Quartz.CGMainDisplayID()
    out: list[Display] = []
    for did in ids[:count]:
        rect = Quartz.CGDisplayBounds(did)
        out.append(
            Display(
                id=int(did),
                x=int(rect.origin.x),
                y=int(rect.origin.y),
                width=int(rect.size.width),
                height=int(rect.size.height),
                is_main=(int(did) == int(main)),
                is_builtin=bool(Quartz.CGDisplayIsBuiltin(did)),
            )
        )
    return out


def pick_projector(displays: list[Display]) -> Display | None:
    """Best guess at which display is the projector.

    Heuristic, in order:
      1. The single non-builtin display (a projector is never the laptop panel)
      2. If multiple externals, prefer the non-main one (user is presumably
         working on the main display and projecting to the other)
      3. Otherwise no guess.
    """
    externals = [d for d in displays if not d.is_builtin]
    if len(externals) == 1:
        return externals[0]
    if len(externals) > 1:
        non_main = [d for d in externals if not d.is_main]
        if len(non_main) == 1:
            return non_main[0]
    return None
