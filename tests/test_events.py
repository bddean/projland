import numpy as np

from projland.events import MarkerEvents
from projland.markers import Marker


def _m(mid: int) -> Marker:
    corners = np.zeros((4, 2), dtype=np.float32)
    return Marker(id=mid, corners=corners)


def test_arrive_called_once():
    arrivals: list[int] = []
    departures: list[int] = []
    t = [0.0]
    ev = MarkerEvents(
        debounce_sec=0.5,
        on_arrive=lambda m: arrivals.append(m.id),
        on_depart=lambda mid: departures.append(mid),
        _now=lambda: t[0],
    )
    ev.update([_m(7)])
    ev.update([_m(7)])  # still present, shouldn't fire again
    assert arrivals == [7]
    assert departures == []


def test_depart_after_debounce():
    arrivals: list[int] = []
    departures: list[int] = []
    t = [0.0]
    ev = MarkerEvents(
        debounce_sec=0.5,
        on_arrive=lambda m: arrivals.append(m.id),
        on_depart=lambda mid: departures.append(mid),
        _now=lambda: t[0],
    )
    ev.update([_m(7)])
    t[0] = 0.1
    ev.update([])  # missing — but within debounce window
    assert departures == []
    t[0] = 0.7
    ev.update([])  # past debounce, fire
    assert departures == [7]


def test_flicker_does_not_fire_depart():
    departures: list[int] = []
    t = [0.0]
    ev = MarkerEvents(
        debounce_sec=0.5,
        on_depart=lambda mid: departures.append(mid),
        _now=lambda: t[0],
    )
    ev.update([_m(7)])
    t[0] = 0.1
    ev.update([])
    t[0] = 0.2
    ev.update([_m(7)])  # came back before debounce
    t[0] = 0.3
    ev.update([_m(7)])
    assert departures == []
