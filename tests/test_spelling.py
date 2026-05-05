import numpy as np

from projland.markers import Marker
from projland.spelling import group_words


def _m(mid: int, x: float, y: float, s: float = 60.0) -> Marker:
    corners = np.array(
        [[x, y], [x + s, y], [x + s, y + s], [x, y + s]], dtype=np.float32
    )
    return Marker(id=mid, corners=corners)


def test_group_words_single_word():
    # ids 7 'a', 2 't' → spells "at" if placed in a row
    markers = [_m(7, 100, 100), _m(2, 200, 100)]
    words = group_words(markers)
    assert len(words) == 1
    assert words[0].text == "at"


def test_group_words_separates_by_gap():
    # two pairs separated by a big gap → two words
    markers = [
        _m(7, 100, 100),
        _m(2, 200, 100),
        _m(12, 800, 100),  # 's'
        _m(2, 900, 100),
    ]
    words = group_words(markers, gap_tolerance=2.5)
    texts = sorted(w.text for w in words)
    assert texts == ["at", "st"]


def test_group_words_multiple_rows():
    markers = [
        _m(7, 100, 100),
        _m(2, 200, 100),
        _m(12, 100, 400),  # 's'
        _m(2, 200, 400),   # 't'
    ]
    words = group_words(markers)
    texts = sorted(w.text for w in words)
    assert texts == ["at", "st"]


def test_ignores_non_letter_ids():
    markers = [_m(999, 100, 100), _m(7, 200, 100)]
    words = group_words(markers)
    assert len(words) == 1
    assert words[0].text == "a"
