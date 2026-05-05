"""CLI entry points for projland."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from projland.app import AppConfig, run
from projland.calibration import CalibrationPattern
from projland.markers import render_marker, DEFAULT_DICT


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="projland", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run the live projector + camera app")
    p_run.add_argument("--camera", type=int, default=0)
    p_run.add_argument("--projector-width", type=int, default=1280)
    p_run.add_argument("--projector-height", type=int, default=800)
    p_run.add_argument("--no-fullscreen", action="store_true")
    p_run.add_argument("--no-debug", action="store_true")
    p_run.add_argument("--recalibrate-every", type=float, default=0.0)

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

    args = parser.parse_args(argv)

    if args.cmd == "run":
        cfg = AppConfig(
            camera_index=args.camera,
            projector_size=(args.projector_width, args.projector_height),
            fullscreen=not args.no_fullscreen,
            show_debug=not args.no_debug,
            recalibrate_every=args.recalibrate_every,
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

        write_demo_video(Path(args.output), frames=args.frames, fps=args.fps)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
