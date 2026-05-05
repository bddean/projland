import cv2
import numpy as np

from projland.markers import MarkerDetector, render_marker


def test_render_marker_shape():
    img = render_marker(7, 200)
    assert img.shape == (200, 200)
    assert img.dtype == np.uint8


def test_detect_single_marker_self_consistency():
    detector = MarkerDetector()
    marker_id = 13
    s = 300
    marker = render_marker(marker_id, s)
    # Pad with white quiet zone
    pad = 60
    canvas = np.full((s + 2 * pad, s + 2 * pad), 255, dtype=np.uint8)
    canvas[pad : pad + s, pad : pad + s] = marker
    bgr = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    result = detector.detect(bgr)
    assert len(result) == 1
    assert result[0].id == marker_id
    # Center should be near the middle
    cx, cy = result[0].center
    expected = (canvas.shape[1] / 2, canvas.shape[0] / 2)
    assert abs(cx - expected[0]) < 3
    assert abs(cy - expected[1]) < 3
    # Side ~= s
    assert abs(result[0].size_px - s) < 4


def test_detect_multiple():
    detector = MarkerDetector()
    s = 200
    pad = 50
    width = (s + 2 * pad) * 3
    height = s + 2 * pad
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    ids = [7, 41, 12]
    for i, mid in enumerate(ids):
        m = render_marker(mid, s)
        x = i * (s + 2 * pad) + pad
        canvas[pad : pad + s, x : x + s] = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
    detected = detector.detect(canvas)
    assert sorted(m.id for m in detected) == sorted(ids)
