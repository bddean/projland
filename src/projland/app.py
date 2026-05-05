"""Live application loop: capture → detect → render → display."""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from projland.calibration import (
    CALIBRATION_IDS,
    Calibration,
    CalibrationPattern,
    identity_calibration,
    solve_calibration,
)
from projland.events import MarkerEvents
from projland.letters import ARCUO_TO_LETTER
from projland.markers import MarkerDetector
from projland.presets import build as build_preset
from projland.render import Renderer, Scene
from projland.tracking import MarkerSmoother


PROJECTOR_WINDOW = "projland — projector"
DEBUG_WINDOW = "projland — camera debug"


def _resolve_projector_target(cfg: "AppConfig"):
    """Return the Display we should put the projector window on, or None."""
    if cfg.projector == "off":
        return None
    from projland.displays import list_displays, pick_projector

    displays = list_displays()
    if not displays:
        return None
    if cfg.projector == "auto":
        target = pick_projector(displays)
        if target is None:
            print(
                "projland: couldn't auto-pick a projector display "
                f"(have {len(displays)}); drag manually or pass --projector <id>"
            )
        return target
    if cfg.projector == "main":
        return next((d for d in displays if d.is_main), None)
    try:
        wanted = int(cfg.projector)
    except ValueError:
        print(f"projland: --projector must be auto/off/main/<id>, got {cfg.projector!r}")
        return None
    target = next((d for d in displays if d.id == wanted), None)
    if target is None:
        print(f"projland: no display with id {wanted}")
    return target


def _place_projector_window(window: str, cfg: "AppConfig") -> None:
    """Move the projector window onto the chosen display, then fullscreen.

    macOS-safe order: force the window into existence with a real imshow,
    pump the event loop, THEN move/resize/fullscreen, pumping between each.
    """
    target = _resolve_projector_target(cfg)

    if cfg.fullscreen:
        win_w, win_h = (target.width, target.height) if target else cfg.projector_size
    else:
        win_w, win_h = cfg.window_size
    placeholder = np.zeros((win_h, win_w, 3), dtype=np.uint8)

    # Force the NSWindow into existence before any move/resize.
    cv2.imshow(window, placeholder)
    cv2.waitKey(100)

    if target is not None:
        print(
            f"projland: placing projector window on {target.label} "
            f"({'fullscreen' if cfg.fullscreen else f'{win_w}x{win_h}'})"
        )
        # Center smaller windows on the target display; otherwise top-left.
        if cfg.fullscreen:
            x, y = target.x, target.y
        else:
            x = target.x + max(0, (target.width - win_w) // 2)
            y = target.y + max(0, (target.height - win_h) // 2)
        cv2.moveWindow(window, x, y)
        cv2.waitKey(50)
    cv2.resizeWindow(window, win_w, win_h)
    cv2.waitKey(50)
    cv2.imshow(window, placeholder)
    cv2.waitKey(100)

    if cfg.fullscreen:
        cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.waitKey(50)
    # Non-fullscreen: WINDOW_NORMAL leaves it user-resizable.


@dataclass
class AppConfig:
    camera_index: int | str = 0  # int (USB index) or str (URL/file path)
    projector_size: tuple[int, int] = (1280, 800)
    calibration_marker_px: int = 180
    calibration_margin_px: int = 80
    fullscreen: bool = True
    show_debug: bool = True
    recalibrate_every: float = 0.0  # seconds; 0 = only once
    preview_mode: bool = False  # if True, skip projector — overlay on camera
    preset: str = "full"
    projector: str = "auto"  # "auto" | "off" | "main" | str(display_id)
    window_size: tuple[int, int] = (960, 600)  # used only when non-fullscreen


def _make_pattern(cfg: AppConfig) -> CalibrationPattern:
    return CalibrationPattern(
        projector_size=cfg.projector_size,
        marker_size_px=cfg.calibration_marker_px,
        margin_px=cfg.calibration_margin_px,
    )


def calibrate_with_camera(
    cap: cv2.VideoCapture,
    detector: MarkerDetector,
    pattern: CalibrationPattern,
    projector_window: str,
    settle_frames: int = 10,
    timeout_sec: float = 30.0,
) -> Calibration | None:
    """Display the calibration pattern; wait for the camera to see all 4
    markers; solve homography."""
    pattern_img = pattern.render()
    cv2.imshow(projector_window, pattern_img)
    cv2.waitKey(1)

    deadline = time.time() + timeout_sec
    last: Calibration | None = None
    consecutive_good = 0

    while time.time() < deadline:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue
        markers = detector.detect(frame)
        cal = solve_calibration(pattern, markers, (frame.shape[1], frame.shape[0]))
        if cal is not None:
            last = cal
            consecutive_good += 1
            if consecutive_good >= settle_frames:
                return cal
        else:
            consecutive_good = 0
        cv2.waitKey(15)
    return last


def run(cfg: AppConfig) -> int:
    cap = cv2.VideoCapture(cfg.camera_index)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    if not cap.isOpened():
        print(f"Could not open camera {cfg.camera_index}")
        return 2

    detector = MarkerDetector()
    pattern = _make_pattern(cfg)
    smoother = MarkerSmoother()

    if cfg.preview_mode:
        # Preview: skip projector, composite effects onto the camera frame.
        ok, probe = cap.read()
        if not ok:
            print("Failed to read first frame from camera.")
            cap.release()
            return 3
        cam_size = (probe.shape[1], probe.shape[0])
        renderer = Renderer(projector_size=cam_size)
        calibration = identity_calibration(cam_size)
        skip_ids: set[int] = set()
    else:
        renderer = Renderer(projector_size=cfg.projector_size)
        cv2.namedWindow(PROJECTOR_WINDOW, cv2.WINDOW_NORMAL)
        _place_projector_window(PROJECTOR_WINDOW, cfg)
        print("Calibrating — please hold still while the corners are detected…")
        calibration = calibrate_with_camera(cap, detector, pattern, PROJECTOR_WINDOW)
        if calibration is None:
            print("Calibration failed. Aborting.")
            cap.release()
            cv2.destroyAllWindows()
            return 3
        print("Calibrated.")
        skip_ids = set(pattern.ids)

    if cfg.show_debug:
        cv2.namedWindow(DEBUG_WINDOW, cv2.WINDOW_NORMAL)

    scene: Scene = build_preset(cfg.preset, skip_ids=skip_ids)

    def _describe(m):
        letter = ARCUO_TO_LETTER.get(m.id)
        if letter is None:
            return f"marker {m.id}"
        return f"letter {letter} (id {m.id})"

    events = MarkerEvents(
        on_arrive=lambda m: print(f"+ {_describe(m)}", flush=True),
        on_depart=lambda mid: print(f"- marker {mid}", flush=True),
    )

    last_cal_t = time.time()
    t0 = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            markers = detector.detect(frame)
            markers = smoother.update(markers)
            t = time.time() - t0
            content_markers = [m for m in markers if m.id not in skip_ids]
            events.update(content_markers)
            projector_img = renderer.render(scene, content_markers, calibration, t=t)
            if cfg.preview_mode:
                composed = cv2.add(frame, projector_img)
                cv2.imshow("projland — preview", composed)
            else:
                cv2.imshow(PROJECTOR_WINDOW, projector_img)

            if cfg.show_debug:
                debug = frame.copy()
                if markers:
                    corners_list = [m.corners.reshape(1, 4, 2) for m in markers]
                    ids_arr = np.array([[m.id] for m in markers], dtype=np.int32)
                    cv2.aruco.drawDetectedMarkers(debug, corners_list, ids_arr)
                cv2.imshow(DEBUG_WINDOW, debug)

            # opportunistic recalibration
            if cfg.recalibrate_every > 0 and (time.time() - last_cal_t) > cfg.recalibrate_every:
                cal_markers = [m for m in markers if m.id in pattern.ids]
                new_cal = solve_calibration(
                    pattern, cal_markers, (frame.shape[1], frame.shape[0])
                )
                if new_cal is not None:
                    calibration = new_cal
                    last_cal_t = time.time()

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            if key == ord("c"):
                # force recalibrate
                new_cal = calibrate_with_camera(cap, detector, pattern, PROJECTOR_WINDOW)
                if new_cal is not None:
                    calibration = new_cal
                    print("Recalibrated.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0
