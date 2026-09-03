#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import textwrap
import time
from typing import Any, Dict

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detectors.sam3_object_detection import Sam3Detector
from hardware.hardware_init import HardwareInitializer
from qbot.mask_asset_tools import normalize_mask
from qbot.qbot_cycles import ScrewCycleManager


def _build_cycle_manager(
    *,
    camera_arm_name: str,
    camera_fixed_name: str,
    tool_name: str,
    debug: bool,
) -> tuple[HardwareInitializer, ScrewCycleManager]:
    hw_init = HardwareInitializer(
        camera_arm_name=camera_arm_name,
        camera_fixed_name=camera_fixed_name,
        ignore_gripper=True,
        tool_name=tool_name,
        debug=debug,
    )
    handles = hw_init.initialize()
    detector = Sam3Detector()
    cycle = ScrewCycleManager(
        handles,
        detector,
        screwdriver_client=None,
        voice_client=None,
        move_to_start=False,
    )
    return hw_init, cycle


def test_screw_state(
    *,
    camera_arm_name: str = "camera_drill",
    camera_fixed_name: str = "camera_fixed",
    tool_name: str = "tcp_drill",
    debug: bool = False,
    test_arm: bool = True,
    test_fixed: bool = True,
) -> Dict[str, Any]:
    """
    Run the two pickup verification checks in ScrewCycleManager:
      - verify_pickup_arm_camera()
      - verify_screw_attached_fixed_camera()
    """
    hw_init = None
    cycle = None

    try:
        hw_init, cycle = _build_cycle_manager(
            camera_arm_name=camera_arm_name,
            camera_fixed_name=camera_fixed_name,
            tool_name=tool_name,
            debug=debug,
        )

        result: Dict[str, Any] = {}

        if test_arm:
            arm_ok, arm_info = cycle.verify_pickup_arm_camera(debug=debug)
            result["arm_camera"] = {"ok": bool(arm_ok), "info": arm_info}
        else:
            result["arm_camera"] = {"skipped": True}

        if test_fixed:
            fixed_ok = bool(cycle.verify_screw_attached_fixed_camera(debug=debug))
            result["fixed_camera_gpt"] = {"ok": fixed_ok}
        else:
            result["fixed_camera_gpt"] = {"skipped": True}

        return result
    finally:
        try:
            if hw_init is not None:
                hw_init.shutdown()
        except Exception as e:
            print(f"[WARN] Failed to shutdown hardware cleanly: {e}")


def _rgb_to_bgr(image_rgb: Any) -> np.ndarray | None:
    if not isinstance(image_rgb, np.ndarray):
        return None
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        return None
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


def _overlay_mask_outline(image_rgb: Any, mask: Any) -> np.ndarray | None:
    base = _rgb_to_bgr(image_rgb)
    if base is None:
        return None
    if isinstance(mask, np.ndarray):
        try:
            mask_u8 = normalize_mask(mask, base.shape[:2])
            contours, _ = cv2.findContours((mask_u8 > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(base, contours, -1, (0, 255, 255), 2, cv2.LINE_AA)
        except Exception:
            pass
    return base


def _fit_image(image_bgr: np.ndarray | None, width: int, height: int) -> np.ndarray:
    canvas = np.full((height, width, 3), 18, dtype=np.uint8)
    if image_bgr is None or image_bgr.size == 0:
        cv2.putText(canvas, "No image", (24, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 180, 180), 2, cv2.LINE_AA)
        return canvas

    src_h, src_w = image_bgr.shape[:2]
    scale = min(width / max(1, src_w), height / max(1, src_h))
    dst_w = max(1, int(round(src_w * scale)))
    dst_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(image_bgr, (dst_w, dst_h), interpolation=cv2.INTER_AREA)
    x0 = (width - dst_w) // 2
    y0 = (height - dst_h) // 2
    canvas[y0:y0 + dst_h, x0:x0 + dst_w] = resized
    return canvas


def _render_tile(
    title: str,
    image_bgr: np.ndarray | None,
    *,
    width: int,
    height: int,
    footer_lines: list[str] | None = None,
    border_color: tuple[int, int, int] = (70, 70, 70),
) -> np.ndarray:
    footer_lines = footer_lines or []
    tile = np.full((height, width, 3), 28, dtype=np.uint8)
    cv2.rectangle(tile, (0, 0), (width - 1, height - 1), border_color, 2, cv2.LINE_AA)
    cv2.rectangle(tile, (0, 0), (width - 1, 42), (38, 38, 38), -1)
    cv2.putText(tile, title, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (240, 240, 240), 2, cv2.LINE_AA)

    footer_h = 22 * max(1, len(footer_lines)) + 18 if footer_lines else 0
    image_area_h = max(80, height - 54 - footer_h)
    tile[48:48 + image_area_h, 8:width - 8] = _fit_image(image_bgr, width - 16, image_area_h)

    if footer_lines:
        y = 48 + image_area_h + 22
        for line in footer_lines:
            cv2.putText(tile, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (205, 205, 205), 1, cv2.LINE_AA)
            y += 22
    return tile


def _draw_wrapped_lines(
    image: np.ndarray,
    lines: list[str],
    *,
    origin: tuple[int, int],
    max_chars: int = 36,
    line_gap: int = 22,
    color: tuple[int, int, int] = (235, 235, 235),
) -> None:
    x, y = origin
    for line in lines:
        wrapped = textwrap.wrap(str(line), width=max_chars) or [""]
        for part in wrapped:
            cv2.putText(image, part, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)
            y += line_gap


def _build_status_card(
    *,
    arm_info: Dict[str, Any],
    arm_ok: bool | None,
    fixed_ok: bool | None,
    loop_idx: int,
    rate_hz: float,
    test_arm: bool,
    test_fixed: bool,
    width: int,
    height: int,
) -> np.ndarray:
    card = np.full((height, width, 3), 24, dtype=np.uint8)
    cv2.rectangle(card, (0, 0), (width - 1, height - 1), (70, 70, 70), 2, cv2.LINE_AA)
    cv2.rectangle(card, (0, 0), (width - 1, 42), (38, 38, 38), -1)
    cv2.putText(card, "Live Metrics", (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (240, 240, 240), 2, cv2.LINE_AA)

    arm_error = str(arm_info.get("error", "") or "")
    arm_status = "Skipped"
    if test_arm:
        if arm_error:
            arm_status = "Error"
        elif arm_ok:
            arm_status = "Picked up"
        else:
            arm_status = "Not picked up"

    fixed_status = "Skipped"
    if test_fixed:
        fixed_status = "Attached" if fixed_ok else "Not attached"

    comparison_method = str(arm_info.get("comparison_method", "mask_similarity"))
    score = arm_info.get("contour_match_score")
    threshold = arm_info.get("contour_match_threshold")
    masked_pixels = arm_info.get("masked_pixel_count")
    asset_name = arm_info.get("asset_name", "-")
    area_ratio = arm_info.get("contour_area_ratio")
    length_ratio = arm_info.get("contour_length_ratio")
    overlap_ratio = arm_info.get("contour_overlap_ratio")

    lines = [
        f"Loop: {loop_idx}",
        f"Rate: {rate_hz:.2f} Hz",
        "",
        f"Arm status: {arm_status}",
        f"Method: {comparison_method}",
        f"Contour match: {score:.4f}" if score is not None else "Contour match: -",
        f"Match threshold: {threshold:.4f}" if threshold is not None else "Match threshold: -",
        f"Area ratio: {area_ratio:.2f}" if area_ratio is not None else "Area ratio: -",
        f"Length ratio: {length_ratio:.2f}" if length_ratio is not None else "Length ratio: -",
        f"Overlap ratio: {overlap_ratio:.2f}" if overlap_ratio is not None else "Overlap ratio: -",
        f"Masked pixels: {masked_pixels}" if masked_pixels is not None else "Masked pixels: -",
        f"Asset: {asset_name}",
        "",
        f"Fixed status: {fixed_status}",
    ]
    if arm_error:
        lines.extend(["", f"Error: {arm_error}"])
    lines.extend(["", "Keys: q / ESC to quit"])
    _draw_wrapped_lines(card, lines, origin=(14, 72))
    return card


def _build_realtime_dashboard(
    *,
    arm_info: Dict[str, Any],
    arm_ok: bool | None,
    fixed_ok: bool | None,
    loop_idx: int,
    rate_hz: float,
    test_arm: bool,
    test_fixed: bool,
) -> np.ndarray:
    if test_arm and test_fixed:
        if arm_ok is True and fixed_ok is True:
            headline = "PICKED UP"
            banner_color = (28, 120, 44)
            subline = "Arm and fixed checks both pass."
        elif arm_ok is False and fixed_ok is False:
            headline = "NOT PICKED UP"
            banner_color = (32, 32, 170)
            subline = "Arm and fixed checks both fail."
        else:
            headline = "CHECK PICKUP"
            banner_color = (0, 140, 210)
            subline = "Arm and fixed checks disagree."
    elif test_arm:
        if arm_info.get("error"):
            headline = "PICKUP CHECK ERROR"
            banner_color = (0, 140, 210)
            subline = "See the error panel for details."
        elif bool(arm_info.get("skipped")):
            headline = "PICKUP CHECK DISABLED"
            banner_color = (90, 90, 90)
            subline = "Arm-camera verification is disabled in config."
        elif arm_ok:
            headline = "PICKED UP"
            banner_color = (28, 120, 44)
            subline = "Arm-camera SAM contour comparison passes."
        else:
            headline = "NOT PICKED UP"
            banner_color = (32, 32, 170)
            subline = "Arm-camera SAM contour comparison fails."
    elif test_fixed:
        if fixed_ok:
            headline = "PICKED UP"
            banner_color = (28, 120, 44)
            subline = "Fixed-camera attachment check passes."
        else:
            headline = "NOT PICKED UP"
            banner_color = (32, 32, 170)
            subline = "Fixed-camera attachment check fails."
    else:
        headline = "NO CHECKS ENABLED"
        banner_color = (90, 90, 90)
        subline = "Enable at least one pickup check."

    width = 1520
    banner_h = 120
    tile_w = 360
    tile_h = 340
    gap = 14
    height = banner_h + gap + tile_h + gap
    canvas = np.full((height, width, 3), 14, dtype=np.uint8)

    cv2.rectangle(canvas, (0, 0), (width, banner_h), banner_color, -1)
    cv2.putText(canvas, headline, (26, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.45, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(canvas, subline, (28, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (235, 235, 235), 2, cv2.LINE_AA)

    raw_bgr = _overlay_mask_outline(arm_info.get("raw_rgb"), arm_info.get("mask"))
    current_masked_bgr = _rgb_to_bgr(arm_info.get("input_masked_rgb"))
    reference_masked_bgr = _rgb_to_bgr(arm_info.get("reference_masked_rgb"))

    score = arm_info.get("contour_match_score")
    threshold = arm_info.get("contour_match_threshold")
    footer = []
    if score is not None and threshold is not None:
        footer.append(f"match={float(score):.4f}  thr<={float(threshold):.4f}")

    row_y = banner_h + gap
    x = gap
    raw_tile = _render_tile(
        "Arm Camera",
        raw_bgr,
        width=tile_w,
        height=tile_h,
        footer_lines=["Reference mask outline over live arm frame"] if raw_bgr is not None else ["No arm frame"],
    )
    canvas[row_y:row_y + tile_h, x:x + tile_w] = raw_tile
    x += tile_w + gap

    current_tile = _render_tile(
        "Live Contour",
        current_masked_bgr,
        width=tile_w,
        height=tile_h,
        footer_lines=footer or ["Current live contour candidate"],
        border_color=(60, 110, 60) if arm_ok else (90, 90, 90),
    )
    canvas[row_y:row_y + tile_h, x:x + tile_w] = current_tile
    x += tile_w + gap

    reference_tile = _render_tile(
        "Template Contour",
        reference_masked_bgr,
        width=tile_w,
        height=tile_h,
        footer_lines=[f"asset={arm_info.get('asset_name', '-')}"],
    )
    canvas[row_y:row_y + tile_h, x:x + tile_w] = reference_tile
    x += tile_w + gap

    status_tile = _build_status_card(
        arm_info=arm_info,
        arm_ok=arm_ok,
        fixed_ok=fixed_ok,
        loop_idx=loop_idx,
        rate_hz=rate_hz,
        test_arm=test_arm,
        test_fixed=test_fixed,
        width=tile_w,
        height=tile_h,
    )
    canvas[row_y:row_y + tile_h, x:x + tile_w] = status_tile
    return canvas


def run_realtime_screw_state(
    *,
    camera_arm_name: str = "camera_drill",
    camera_fixed_name: str = "camera_fixed",
    tool_name: str = "tcp_drill",
    debug: bool = False,
    gui: bool = False,
    test_arm: bool = True,
    test_fixed: bool = True,
    interval_s: float = 0.5,
) -> None:
    """
    Continuously run the screw-state checks until interrupted.
    """
    hw_init = None
    cycle = None
    loop_idx = 0
    ema_period_s = None
    last_t = None

    try:
        hw_init, cycle = _build_cycle_manager(
            camera_arm_name=camera_arm_name,
            camera_fixed_name=camera_fixed_name,
            tool_name=tool_name,
            debug=debug,
        )

        print("[INFO] Realtime screw-state test running. Press Ctrl-C to stop.")
        if gui:
            cv2.namedWindow("pickup_status_dashboard", cv2.WINDOW_NORMAL)
        while True:
            t0 = time.perf_counter()
            loop_idx += 1
            out: Dict[str, Any] = {}

            if test_arm:
                if gui:
                    arm_info = cycle.get_pickup_arm_camera_status()
                    arm_ok = bool(arm_info.get("passed", False))
                else:
                    arm_ok, arm_info = cycle.verify_pickup_arm_camera(debug=debug)
                out["arm_camera"] = {"ok": bool(arm_ok), "info": arm_info}
            else:
                out["arm_camera"] = {"skipped": True}
                arm_ok = None
                arm_info = {}

            if test_fixed:
                fixed_ok = bool(cycle.verify_screw_attached_fixed_camera(debug=(debug and not gui)))
                out["fixed_camera"] = {"ok": fixed_ok}
            else:
                out["fixed_camera"] = {"skipped": True}
                fixed_ok = None

            now = time.perf_counter()
            if last_t is not None:
                period = now - last_t
                ema_period_s = period if ema_period_s is None else (0.2 * period + 0.8 * ema_period_s)
            last_t = now
            rate_hz = (1.0 / ema_period_s) if (ema_period_s and ema_period_s > 0) else 0.0

            arm_ok_val = out.get("arm_camera", {}).get("ok")
            fixed_ok_val = out.get("fixed_camera", {}).get("ok")
            print(
                f"[{loop_idx:05d}] arm={arm_ok_val} fixed={fixed_ok_val} "
                f"rate={rate_hz:.2f}Hz"
            )
            if test_arm:
                arm_info = out["arm_camera"].get("info", {}) or {}
                if isinstance(arm_info, dict):
                    score = arm_info.get("contour_match_score")
                    thr = arm_info.get("contour_match_threshold")
                    if score is not None:
                        print(f"        arm contour_match={score:.4f} threshold<={thr}")

            if gui:
                dashboard = _build_realtime_dashboard(
                    arm_info=arm_info if isinstance(arm_info, dict) else {},
                    arm_ok=arm_ok,
                    fixed_ok=fixed_ok,
                    loop_idx=loop_idx,
                    rate_hz=rate_hz,
                    test_arm=test_arm,
                    test_fixed=test_fixed,
                )
                cv2.imshow("pickup_status_dashboard", dashboard)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    print("[INFO] GUI close requested.")
                    break

            dt = time.perf_counter() - t0
            sleep_s = max(0.0, float(interval_s) - dt)
            if sleep_s > 0:
                time.sleep(sleep_s)
    except KeyboardInterrupt:
        print("\n[INFO] Realtime screw-state test stopped.")
    finally:
        try:
            if hw_init is not None:
                hw_init.shutdown()
        except Exception as e:
            print(f"[WARN] Failed to shutdown hardware cleanly: {e}")
        if gui:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass


def main() -> None:
    ap = argparse.ArgumentParser(description="Test screw pickup state checks (arm mask + fixed-camera GPT).")
    ap.add_argument("--debug", action="store_true", help="Show debug windows for both checks where supported.")
    ap.add_argument("--realtime", action="store_true", help="Run continuously until Ctrl-C.")
    ap.add_argument("--gui", action="store_true", help="Show a live pickup-status dashboard window.")
    ap.add_argument("--interval-s", type=float, default=0.5, help="Realtime mode loop interval in seconds.")
    ap.add_argument("--arm-only", action="store_true", help="Run only verify_pickup_arm_camera().")
    ap.add_argument("--fixed-only", action="store_true", help="Run only verify_screw_attached_fixed_camera().")
    ap.add_argument("--camera-arm-name", default="camera_drill")
    ap.add_argument("--camera-fixed-name", default="camera_fixed")
    ap.add_argument("--tool-name", default="tcp_drill")
    ap.add_argument("--fail-on-false", action="store_true", help="Exit non-zero if any executed check returns False.")
    args = ap.parse_args()

    if args.arm_only and args.fixed_only:
        raise SystemExit("Use only one of --arm-only or --fixed-only.")

    test_arm = not args.fixed_only
    test_fixed = not args.arm_only

    if args.gui:
        args.realtime = True

    if args.realtime:
        run_realtime_screw_state(
            camera_arm_name=args.camera_arm_name,
            camera_fixed_name=args.camera_fixed_name,
            tool_name=args.tool_name,
            debug=args.debug,
            gui=args.gui,
            test_arm=test_arm,
            test_fixed=test_fixed,
            interval_s=args.interval_s,
        )
        return

    result = test_screw_state(
        camera_arm_name=args.camera_arm_name,
        camera_fixed_name=args.camera_fixed_name,
        tool_name=args.tool_name,
        debug=args.debug,
        test_arm=test_arm,
        test_fixed=test_fixed,
    )

    print("[RESULT] Screw state checks")
    if "arm_camera" in result:
        print(f"  arm_camera: {result['arm_camera']}")
    if "fixed_camera_gpt" in result:
        print(f"  fixed_camera_gpt: {result['fixed_camera_gpt']}")

    if args.fail_on_false:
        failures = []
        if test_arm and not bool(result.get("arm_camera", {}).get("ok", False)):
            failures.append("arm_camera")
        if test_fixed and not bool(result.get("fixed_camera_gpt", {}).get("ok", False)):
            failures.append("fixed_camera_gpt")
        if failures:
            raise SystemExit(f"Failed checks: {', '.join(failures)}")


if __name__ == "__main__":
    main()
