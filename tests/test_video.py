import tempfile
from pathlib import Path

import cv2

from projland.demo_video import write_demo_video, write_demo_snapshot


def test_demo_video_produces_expected_framecount(tmp_path: Path):
    out = tmp_path / "demo.mp4"
    write_demo_video(out, frames=20, fps=20)
    assert out.exists()
    cap = cv2.VideoCapture(str(out))
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # mp4 metadata is sometimes off-by-one or off-by-a-few.
        assert 15 <= n <= 25, f"expected ~20 frames, got {n}"
    finally:
        cap.release()


def test_demo_snapshot_writes_png(tmp_path: Path):
    out = tmp_path / "snap.png"
    write_demo_snapshot(out, t=0.5)
    img = cv2.imread(str(out))
    assert img is not None
    h, w = img.shape[:2]
    assert h > 100 and w > 100
