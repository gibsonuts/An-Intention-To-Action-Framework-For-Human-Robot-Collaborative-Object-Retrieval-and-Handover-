#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hardware.hardware_init import HardwareInitializer
from qbot.mask_asset_tools import save_mask_asset


def capture_camera_rgb(
    camera: str,
    camera_arm_name: str,
    camera_fixed_name: str,
    tool_name: str,
    warmup_frames: int,
    debug: bool = False,
) -> np.ndarray:
    hw = HardwareInitializer(
        ignore_arm=True,
        ignore_gripper=True,
        camera_arm_name=camera_arm_name,
        camera_fixed_name=camera_fixed_name,
        camera_warmup_secs=warmup_frames,
        tool_name=tool_name,
        debug=debug,
    )
    handles = None
    try:
        handles = hw.initialize()
        cam = handles.cam_fixed if camera == "fixed" else handles.cam_arm
        _depth_u16, color_rgb = cam.get_rgbd()
        return color_rgb
    finally:
        try:
            hw.shutdown()
        except Exception:
            pass


class MaskPainter:
    def __init__(self, image_rgb: np.ndarray):
        if image_rgb is None or image_rgb.ndim != 3:
            raise ValueError("Expected RGB image")
        self.image_rgb = image_rgb.astype(np.uint8, copy=False)
        self.image_bgr = cv2.cvtColor(self.image_rgb, cv2.COLOR_RGB2BGR)
        self.mask = np.zeros(self.image_rgb.shape[:2], dtype=np.uint8)
        self.brush_size = 20
        self.opacity = 55  # percent
        self.show_overlay = True
        self.drawing = False
        self.erase_mode = False
        self.cursor_xy: Optional[tuple[int, int]] = None
        self.mode = "brush"  # "brush" or "polygon"
        self.polygon_points: list[tuple[int, int]] = []

    def _draw_at(self, x: int, y: int, value: int) -> None:
        cv2.circle(self.mask, (x, y), int(self.brush_size), int(value), thickness=-1, lineType=cv2.LINE_AA)

    def add_polygon_point(self, x: int, y: int) -> None:
        self.polygon_points.append((int(x), int(y)))

    def pop_polygon_point(self) -> None:
        if self.polygon_points:
            self.polygon_points.pop()

    def clear_polygon_points(self) -> None:
        self.polygon_points.clear()

    def commit_polygon_to_mask(self, value: int = 255) -> bool:
        if len(self.polygon_points) < 3:
            return False
        pts = np.array(self.polygon_points, dtype=np.int32).reshape((-1, 1, 2))
        line_thickness = max(1, int(self.brush_size))
        cv2.fillPoly(self.mask, [pts], int(value), lineType=cv2.LINE_AA)
        # Keep the polygon edge thickness visually consistent with brush mode.
        cv2.polylines(self.mask, [pts], True, int(value), thickness=line_thickness, lineType=cv2.LINE_AA)
        self.clear_polygon_points()
        return True

    def toggle_mode(self) -> None:
        self.mode = "polygon" if self.mode == "brush" else "brush"
        self.drawing = False

    def on_mouse(self, event, x, y, flags, param) -> None:
        self.cursor_xy = (int(x), int(y))
        if self.mode == "polygon":
            if event == cv2.EVENT_LBUTTONDOWN:
                self.add_polygon_point(x, y)
            elif event == cv2.EVENT_RBUTTONDOWN:
                self.pop_polygon_point()
            elif event == cv2.EVENT_LBUTTONDBLCLK:
                self.commit_polygon_to_mask(255)
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.erase_mode = False
            self._draw_at(x, y, 255)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.drawing = True
            self.erase_mode = True
            self._draw_at(x, y, 0)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self._draw_at(x, y, 0 if self.erase_mode else 255)
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            self.drawing = False

    def render(self) -> np.ndarray:
        vis = self.image_bgr.copy()
        if self.show_overlay:
            overlay = vis.copy()
            overlay[self.mask > 0] = (0, 0, 255)
            alpha = float(np.clip(self.opacity, 0, 100)) / 100.0
            vis = cv2.addWeighted(overlay, alpha, vis, 1.0 - alpha, 0.0)

        # contour for masked area
        contours, _ = cv2.findContours((self.mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (0, 255, 255), 1)

        if self.cursor_xy is not None:
            cv2.circle(vis, self.cursor_xy, int(self.brush_size), (255, 255, 255), 1, cv2.LINE_AA)

        if self.polygon_points:
            pts = np.array(self.polygon_points, dtype=np.int32)
            line_thickness = max(1, int(self.brush_size))
            point_radius = max(3, int(self.brush_size // 2))
            cv2.polylines(
                vis,
                [pts.reshape((-1, 1, 2))],
                False,
                (255, 200, 0),
                line_thickness,
                cv2.LINE_AA,
            )
            for px, py in self.polygon_points:
                cv2.circle(vis, (px, py), point_radius, (0, 255, 255), -1, cv2.LINE_AA)
            if self.cursor_xy is not None and self.mode == "polygon":
                cv2.line(
                    vis,
                    self.polygon_points[-1],
                    self.cursor_xy,
                    (120, 220, 255),
                    line_thickness,
                    cv2.LINE_AA,
                )
            if len(self.polygon_points) >= 3:
                cv2.line(
                    vis,
                    self.polygon_points[-1],
                    self.polygon_points[0],
                    (0, 255, 0),
                    line_thickness,
                    cv2.LINE_AA,
                )

        mask_px = int(np.count_nonzero(self.mask))
        h, w = self.mask.shape
        coverage = 100.0 * mask_px / max(1, h * w)
        status = (
            f"Mode:{self.mode}  Brush:{self.brush_size}px  Overlay:{self.opacity}%  "
            f"Mask:{coverage:.1f}%  PolyPts:{len(self.polygon_points)}"
        )
        cv2.rectangle(vis, (8, 8), (min(w - 8, 520), 38), (0, 0, 0), -1)
        cv2.putText(vis, status, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return vis


def run_mask_painter(image_rgb: np.ndarray, asset_name: str, camera: str) -> None:
    painter = MaskPainter(image_rgb)
    win = "Mask Painter"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(1200, image_rgb.shape[1]), min(900, image_rgb.shape[0]))
    cv2.setMouseCallback(win, painter.on_mouse)

    def on_brush(v: int) -> None:
        painter.brush_size = max(1, int(v))

    def on_opacity(v: int) -> None:
        painter.opacity = int(np.clip(v, 0, 100))

    cv2.createTrackbar("Brush", win, painter.brush_size, 200, on_brush)
    cv2.createTrackbar("Overlay%", win, painter.opacity, 100, on_opacity)

    print("Controls:")
    print("  m          : toggle brush/polygon mode")
    print("  Brush mode:")
    print("    Left drag  : paint mask")
    print("    Right drag : erase mask")
    print("  Polygon mode:")
    print("    Left click  : add polygon point")
    print("    Right click : remove last polygon point")
    print("    f or Enter  : fill polygon into mask")
    print("    x           : clear polygon points")
    print("    Double-left : fill polygon into mask")
    print("  [ / ]      : brush size down/up")
    print("  c          : clear mask")
    print("  o          : toggle overlay")
    print("  s          : save asset and exit")
    print("  q or ESC   : quit without saving")

    while True:
        cv2.imshow(win, painter.render())
        key = cv2.waitKey(20) & 0xFF

        if key in (27, ord("q")):
            print("Cancelled; nothing saved.")
            break
        if key == ord("s"):
            saved = save_mask_asset(
                image_rgb=painter.image_rgb,
                mask=painter.mask,
                asset_name=asset_name,
                camera=camera,
            )
            print("Saved mask asset:")
            print(f"  image : {saved['image_path']}")
            print(f"  mask  : {saved['mask_path']}")
            print(f"  masked: {saved['masked_path']}")
            print(f"  meta  : {saved['meta_path']}")
            break
        if key == ord("c"):
            painter.mask.fill(0)
            painter.clear_polygon_points()
        elif key == ord("o"):
            painter.show_overlay = not painter.show_overlay
        elif key == ord("m"):
            painter.toggle_mode()
            print(f"Mode: {painter.mode}")
        elif key == ord("x"):
            painter.clear_polygon_points()
        elif key in (ord("f"), 13):
            ok = painter.commit_polygon_to_mask(255)
            if not ok:
                print("Polygon needs at least 3 points to fill.")
        elif key in (ord("]"), ord("="), ord("+")):
            painter.brush_size = min(200, painter.brush_size + 2)
            cv2.setTrackbarPos("Brush", win, painter.brush_size)
        elif key in (ord("["), ord("-"), ord("_")):
            painter.brush_size = max(1, painter.brush_size - 2)
            cv2.setTrackbarPos("Brush", win, painter.brush_size)

    cv2.destroyWindow(win)


def main() -> None:
    ap = argparse.ArgumentParser(description="Capture image from fixed/arm camera and paint a mask asset.")
    ap.add_argument("--asset-name", required=True, help="Base name for saved files in qbot/assets")
    ap.add_argument("--camera", choices=["fixed", "arm"], default="fixed")
    ap.add_argument("--camera-arm-name", default="camera_drill")
    ap.add_argument("--camera-fixed-name", default="camera_fixed")
    ap.add_argument("--tool-name", default="tcp_drill", help="Tool name for HardwareInitializer")
    ap.add_argument("--warmup-frames", type=int, default=30)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    color_rgb = capture_camera_rgb(
        camera=args.camera,
        camera_arm_name=args.camera_arm_name,
        camera_fixed_name=args.camera_fixed_name,
        tool_name=args.tool_name,
        warmup_frames=args.warmup_frames,
        debug=args.debug,
    )
    run_mask_painter(image_rgb=color_rgb, asset_name=args.asset_name, camera=args.camera)


if __name__ == "__main__":
    main()
