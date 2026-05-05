"""CLI entry points for projland."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from projland.app import AppConfig, run
from projland.calibration import CalibrationPattern
from projland.markers import render_marker, DEFAULT_DICT
from projland.presets import PRESETS


def _probe_camera(source: int | str) -> tuple[bool, tuple[int, int] | None]:
    """Open `source`, try one read; return (ok, (w, h) or None)."""
    cap = cv2.VideoCapture(source)
    try:
        if not cap.isOpened():
            return False, None
        ok, frame = cap.read()
        if not ok or frame is None:
            return False, None
        h, w = frame.shape[:2]
        return True, (w, h)
    finally:
        cap.release()


def _list_cameras(max_index: int, urls: list[str]) -> int:
    print(f"Probing USB indices 0..{max_index - 1}…")
    found = 0
    for i in range(max_index):
        ok, size = _probe_camera(i)
        if ok:
            found += 1
            print(f"  [{i}] OK — {size[0]}x{size[1]}")
        else:
            print(f"  [{i}] —")
    for url in urls:
        ok, size = _probe_camera(url)
        if ok:
            found += 1
            print(f"  [{url}] OK — {size[0]}x{size[1]}")
        else:
            print(f"  [{url}] could not open")
    print(f"{found} working source(s).")
    return 0 if found else 1


def _camera_arg(value: str) -> int | str:
    """Parse --camera: int → USB device index; otherwise pass through as URL/path."""
    try:
        return int(value)
    except ValueError:
        return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="projland", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run the live projector + camera app")
    p_run.add_argument(
        "--camera",
        type=_camera_arg,
        default=0,
        help="USB index (e.g. 0, 1) OR a stream URL (e.g. http://192.168.1.245:8080/video)",
    )
    p_run.add_argument("--projector-width", type=int, default=1280)
    p_run.add_argument("--projector-height", type=int, default=800)
    p_run.add_argument("--no-fullscreen", action="store_true")
    p_run.add_argument(
        "--window-size",
        default="960x600",
        help="WxH for the projector window when --no-fullscreen (default 960x600)",
    )
    p_run.add_argument("--no-debug", action="store_true")
    p_run.add_argument("--recalibrate-every", type=float, default=0.0)
    p_run.add_argument(
        "--preview",
        action="store_true",
        help="Skip projector — composite effects onto the camera feed instead",
    )
    p_run.add_argument(
        "--preset", default="full", choices=sorted(PRESETS),
    )
    p_run.add_argument(
        "--projector",
        default="auto",
        help=(
            "macOS only. 'auto' (default) places the projector window on the "
            "single non-main display if there is one. 'main' or an integer "
            "display id selects explicitly. 'off' disables placement."
        ),
    )

    p_pat = sub.add_parser("calibration-image", help="Save the calibration image")
    p_pat.add_argument("--width", type=int, default=1280)
    p_pat.add_argument("--height", type=int, default=800)
    p_pat.add_argument("--marker-size", type=int, default=180)
    p_pat.add_argument("--margin", type=int, default=80)
    p_pat.add_argument("-o", "--output", default="calibration.png")

    p_mark = sub.add_parser("marker", help="Render an ArUco marker as a printable PNG")
    p_mark.add_argument("id", type=int)
    p_mark.add_argument("--size", type=int, default=600)
    p_mark.add_argument("-o", "--output", default=None)

    p_demo = sub.add_parser("demo", help="Run the synthetic headless demo to a video file")
    p_demo.add_argument("-o", "--output", default="demo.mp4")
    p_demo.add_argument("--frames", type=int, default=120)
    p_demo.add_argument("--fps", type=int, default=30)
    p_demo.add_argument("--preset", default="full", choices=sorted(PRESETS))

    p_snap = sub.add_parser("snapshot", help="Render one synthetic demo frame as PNG")
    p_snap.add_argument("-o", "--output", default="snapshot.png")
    p_snap.add_argument("--t", type=float, default=1.5)
    p_snap.add_argument("--preset", default="full", choices=sorted(PRESETS))

    p_disp = sub.add_parser(
        "displays",
        help="List connected displays (macOS only). Useful for picking a projector.",
    )

    p_list = sub.add_parser(
        "list-cameras",
        help="Probe USB camera indices and report which open. Optionally include a URL too.",
    )
    p_list.add_argument(
        "--max", type=int, default=5,
        help="Highest USB index to try (default 5)",
    )
    p_list.add_argument(
        "--url", action="append", default=[],
        help="Also probe this stream URL. Repeatable.",
    )

    p_test = sub.add_parser(
        "test-image",
        help=(
            "Run detection+render on a static image (treated as the camera "
            "frame). Effects are composited onto a copy of the image using "
            "the identity calibration."
        ),
    )
    p_test.add_argument("input")
    p_test.add_argument("-o", "--output", default="test_out.png")
    p_test.add_argument("--preset", default="full", choices=sorted(PRESETS))

    args = parser.parse_args(argv)

    if args.cmd == "run":
        try:
            ww, wh = (int(x) for x in args.window_size.lower().split("x"))
        except ValueError:
            print(f"--window-size must be WxH, got {args.window_size!r}")
            return 2
        cfg = AppConfig(
            camera_index=args.camera,
            projector_size=(args.projector_width, args.projector_height),
            fullscreen=not args.no_fullscreen,
            show_debug=not args.no_debug,
            recalibrate_every=args.recalibrate_every,
            preview_mode=args.preview,
            preset=args.preset,
            projector=args.projector,
            window_size=(ww, wh),
        )
        return run(cfg)

    if args.cmd == "calibration-image":
        pat = CalibrationPattern(
            projector_size=(args.width, args.height),
            marker_size_px=args.marker_size,
            margin_px=args.margin,
        )
        img = pat.render()
        cv2.imwrite(args.output, img)
        print(f"Wrote {args.output}")
        return 0

    if args.cmd == "marker":
        img = render_marker(args.id, args.size)
        out = args.output or f"marker_{args.id}.png"
        cv2.imwrite(out, img)
        print(f"Wrote {out}")
        return 0

    if args.cmd == "demo":
        from projland.demo_video import write_demo_video

        write_demo_video(
            Path(args.output), frames=args.frames, fps=args.fps, preset=args.preset
        )
        return 0

    if args.cmd == "snapshot":
        from projland.demo_video import write_demo_snapshot

        write_demo_snapshot(Path(args.output), t=args.t, preset=args.preset)
        return 0

    if args.cmd == "displays":
        from projland.displays import list_displays, pick_projector

        ds = list_displays()
        if not ds:
            print("No displays detected (macOS-only; install with --extra macos).")
            return 1
        for d in ds:
            print(d.label)
        picked = pick_projector(ds)
        if picked is not None:
            print(f"projector → {picked.label}")
        return 0

    if args.cmd == "list-cameras":
        return _list_cameras(args.max, args.url)

    if args.cmd == "test-image":
        from projland.calibration import identity_calibration
        from projland.markers import MarkerDetector
        from projland.presets import build as build_preset
        from projland.render import Renderer

        img = cv2.imread(args.input)
        if img is None:
            print(f"Could not read {args.input}")
            return 2
        h, w = img.shape[:2]
        markers = MarkerDetector().detect(img)
        cal = identity_calibration((w, h))
        scene = build_preset(args.preset)
        proj = Renderer(projector_size=(w, h)).render(scene, markers, cal)
        composed = cv2.add(img, proj)
        cv2.imwrite(args.output, composed)
        print(
            f"Detected {len(markers)} markers "
            f"({sorted(m.id for m in markers)}); wrote {args.output}"
        )
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
