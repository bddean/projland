# projland

Dynamic-land-style projector + webcam AR. Point a webcam and a projector at the
same flat surface; place printed ArUco markers on it; projland decorates them
with halos, labels, pulses, and constellations.

It auto-calibrates by projecting four ArUco fiducials at the projector's
corners and solving the projector→camera homography from where the camera
sees them.

The whole pipeline (detection, calibration, render, projection, observation)
also runs in a fully synthetic mode so it can be exercised headlessly with no
real hardware — see `tests/test_pipeline.py` and `projland demo`.

## Install

```bash
uv sync --extra dev
```

Requires Python 3.13 and a working camera + projector for the live app.
`opencv-contrib-python` brings in the ArUco module.

## Quick demo (no hardware)

Renders a 60-frame synthetic video that simulates the entire pipeline:

```bash
uv run projland demo -o demo.mp4 --frames 60
```

The synthetic camera observes printed markers on a virtual surface; projland
calibrates against four corner fiducials and projects decorations around the
content markers; the synthetic camera observes the lit-up surface and that's
the video frame.

## Live app

Plug in a webcam and a projector (treat the projector as a second display).
Then:

```bash
uv run projland run --projector-width 1280 --projector-height 800
```

Drag the `projland — projector` window onto the projector display.
Markers from `cv2.aruco.DICT_ARUCO_ORIGINAL` with IDs 200/201/202/203 are
reserved — projland projects those itself for calibration. Place any *other*
IDs as content markers.

Keys:

- `q` / Esc — quit
- `c` — recalibrate

## Print markers

```bash
uv run projland marker 7 --size 600 -o marker_7.png
uv run projland calibration-image -o calibration.png   # mostly for debugging
```

## Tests

```bash
uv run pytest
```

The synthetic harness (`projland.synthetic`) sets up a virtual flat surface
with two homography-distorted views (camera + projector). The pipeline test
asserts a halo drawn for a detected marker actually shows up centered on that
marker in the camera's view of the projection — i.e., the calibration is
correct end-to-end.

## Layout

- `projland.markers` — ArUco wrappers
- `projland.calibration` — projected-fiducial homography solver
- `projland.render` — Scene/Renderer + composable Effects
- `projland.synthetic` — virtual camera + projector for headless tests
- `projland.app` — live app loop
- `projland.cli` — `projland` CLI entry point
- `projland.demo_video` — synthetic end-to-end demo video writer
