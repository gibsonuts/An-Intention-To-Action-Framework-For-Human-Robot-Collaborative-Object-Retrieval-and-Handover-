#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import threading
import time
from typing import Any, Dict, Optional

import cv2
import numpy as np
import yaml

CONFIG_PATH = ROOT / "hardware" / "config" / "config.yaml"
DEFAULT_WINDOW_NAME = "SAM3 Realtime Detection"
BOX_COLORS = [
    (0, 255, 0),
    (0, 200, 255),
    (255, 180, 0),
    (255, 0, 255),
    (80, 220, 120),
]


def load_hardware_config() -> Dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def make_status_frame(width: int = 960, height: int = 720, message: str = "Waiting for camera frames...") -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(frame, message, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def format_detection_label(detection: Dict[str, Any], fallback_prompt: str) -> str:
    label = str(detection.get("category") or detection.get("phrase") or fallback_prompt)
    score = detection.get("score")
    if score is None:
        return label
    return f"{label} {float(score):.2f}"


def draw_detections(
    frame_bgr: np.ndarray,
    detections: list[Dict[str, Any]],
    prompt: str,
    draw_masks: bool,
    apply_mask_overlay: Any,
) -> np.ndarray:
    frame = frame_bgr.copy()

    for idx, det in enumerate(detections):
        color = BOX_COLORS[idx % len(BOX_COLORS)]
        if draw_masks:
            seg = det.get("segmentation")
            if seg is not None:
                frame = apply_mask_overlay(frame, seg, color, alpha_fg=0.35)

        contour = det.get("contour")
        if contour is not None:
            cv2.drawContours(frame, [contour], -1, color, 2)

        bbox = det.get("bbox")
        if bbox is None or len(bbox) != 4:
            continue
        x, y, w, h = [int(v) for v in bbox]
        x2 = x + max(w, 1)
        y2 = y + max(h, 1)
        cv2.rectangle(frame, (x, y), (x2, y2), color, 2)

        center = det.get("center")
        if center is not None and len(center) == 2:
            cv2.circle(frame, (int(center[0]), int(center[1])), 4, color, -1)

        label = format_detection_label(det, prompt)
        label_y = max(y - 8, 18)
        cv2.putText(frame, label, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, label, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    return frame


def draw_status_panel(
    frame_bgr: np.ndarray,
    camera_name: str,
    prompt: str,
    conf: float,
    busy: bool,
    result: "DetectionResult",
    run_every_ms: int,
) -> np.ndarray:
    frame = frame_bgr.copy()

    age_ms = None
    if result.completed_at is not None:
        age_ms = int((time.time() - result.completed_at) * 1000.0)

    lines = [
        f"{camera_name}  prompt='{prompt}'  conf={conf:.2f}",
        f"status: {'detecting' if busy else 'idle'}  detections={len(result.detections)}  run_every={run_every_ms} ms",
        f"last_inference={result.inference_ms:.0f} ms  age={age_ms if age_ms is not None else 'n/a'} ms  SPACE: refresh  ESC/Q: quit",
    ]

    if result.error:
        lines.append(f"error: {result.error}")

    y = 24
    for line in lines:
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
        y += 28

    return frame


@dataclass
class DetectionResult:
    detections: list[Dict[str, Any]] = field(default_factory=list)
    inference_ms: float = 0.0
    completed_at: Optional[float] = None
    error: Optional[str] = None


class Sam3DetectionWorker:
    def __init__(
        self,
        detector: Any,
        prompt: str,
        conf: float,
        orientation_align: str,
        debug: bool = False,
    ) -> None:
        self.detector = detector
        self.prompt = prompt
        self.conf = conf
        self.orientation_align = orientation_align
        self.debug = debug

        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._pending_frame: Optional[np.ndarray] = None
        self._result = DetectionResult()
        self._busy = False

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def submit(self, frame_bgr: np.ndarray) -> None:
        with self._lock:
            self._pending_frame = frame_bgr.copy()

    def snapshot(self) -> tuple[DetectionResult, bool]:
        with self._lock:
            result = DetectionResult(
                detections=list(self._result.detections),
                inference_ms=float(self._result.inference_ms),
                completed_at=self._result.completed_at,
                error=self._result.error,
            )
            busy = self._busy or self._pending_frame is not None
        return result, busy

    def _run(self) -> None:
        while self._running:
            with self._lock:
                frame = self._pending_frame
                if frame is not None:
                    self._pending_frame = None
                    self._busy = True

            if frame is None:
                time.sleep(0.01)
                continue

            start = time.perf_counter()
            detections: list[Dict[str, Any]] = []
            error: Optional[str] = None

            try:
                detections = self.detector.segment(
                    color_bgr=frame,
                    text_prompt=self.prompt,
                    confidence_threshold=self.conf,
                    category=self.prompt,
                    orientation_align=self.orientation_align,
                )
            except Exception as exc:
                error = str(exc)
                if self.debug:
                    print(f"[ERROR] SAM3 inference failed: {exc}")

            inference_ms = (time.perf_counter() - start) * 1000.0
            with self._lock:
                self._result = DetectionResult(
                    detections=detections,
                    inference_ms=inference_ms,
                    completed_at=time.time(),
                    error=error,
                )
                self._busy = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Realtime RealSense viewer with SAM3 prompt-based detection overlays."
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Text prompt to detect, for example 'screw head' or 'box'.",
    )
    parser.add_argument(
        "--camera-name",
        default="camera_drill",
        help="Camera config name from hardware/config/config.yaml.",
    )
    parser.add_argument(
        "--camera-id",
        default=None,
        help="Optional direct camera serial/topic id override.",
    )
    parser.add_argument(
        "--server-ip",
        default=None,
        help="Optional ZeroMQ camera server IP override.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Optional ZeroMQ camera server port override.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="SAM3 device.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.2,
        help="SAM3 confidence threshold.",
    )
    parser.add_argument(
        "--orientation-align",
        default="long",
        choices=["long", "short"],
        help="Oriented-box alignment mode passed through to SAM3.",
    )
    parser.add_argument(
        "--run-every-ms",
        type=int,
        default=500,
        help="Minimum time between detection submissions.",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=30,
        help="Camera warmup frame count before starting detection.",
    )
    parser.add_argument(
        "--draw-masks",
        action="store_true",
        help="Overlay SAM3 masks in addition to bounding boxes.",
    )
    parser.add_argument(
        "--window-name",
        default=DEFAULT_WINDOW_NAME,
        help="OpenCV window title.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable extra console logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from hardware.camera_rs_client import NetworkRealSenseCamera
    except ModuleNotFoundError as exc:
        if exc.name == "zmq":
            raise SystemExit("[ERROR] Missing dependency 'pyzmq'. Install it to use the RealSense client.") from exc
        raise

    from detectors.sam3_object_detection import Sam3Detector, apply_mask_overlay

    cfg = load_hardware_config()
    camera_cfg = cfg.get(args.camera_name)
    if not isinstance(camera_cfg, dict):
        raise SystemExit(f"[ERROR] Camera config '{args.camera_name}' not found in {CONFIG_PATH}")

    server_ip = args.server_ip or camera_cfg.get("network_camera_ip")
    port = int(args.port or camera_cfg.get("network_camera_port", 5552))
    camera_id = args.camera_id or camera_cfg.get("serial")
    image_mask = NetworkRealSenseCamera.load_mask_from_config(camera_cfg)

    camera = NetworkRealSenseCamera(
        intrinsics=camera_cfg.get("intrinsics"),
        server_ip=server_ip,
        port=port,
        camera_id=camera_id,
        image_mask=image_mask,
    )

    detector = Sam3Detector(
        device=args.device,
        default_conf_threshold=args.conf,
    )
    worker = Sam3DetectionWorker(
        detector=detector,
        prompt=args.prompt,
        conf=args.conf,
        orientation_align=args.orientation_align,
        debug=args.debug,
    )

    print(f"[camera] {args.camera_name} -> tcp://{server_ip}:{port} id={camera_id}")
    print(f"[detect] prompt='{args.prompt}' device={args.device} conf={args.conf:.2f}")

    last_submit_at = 0.0

    try:
        camera.start()
        camera.warmup(args.warmup_frames)
        worker.start()

        while True:
            try:
                _, color_rgb = camera.get_rgbd()
                frame_bgr = color_rgb[:, :, ::-1].copy()
            except Exception as exc:
                frame_bgr = make_status_frame(message=f"{args.camera_name}: {exc}")

            now = time.monotonic()
            if (now - last_submit_at) * 1000.0 >= float(args.run_every_ms):
                worker.submit(frame_bgr)
                last_submit_at = now

            result, busy = worker.snapshot()
            vis = draw_detections(
                frame_bgr=frame_bgr,
                detections=result.detections,
                prompt=args.prompt,
                draw_masks=args.draw_masks,
                apply_mask_overlay=apply_mask_overlay,
            )
            vis = draw_status_panel(
                frame_bgr=vis,
                camera_name=args.camera_name,
                prompt=args.prompt,
                conf=args.conf,
                busy=busy,
                result=result,
                run_every_ms=args.run_every_ms,
            )
            cv2.imshow(args.window_name, vis)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key == ord(" "):
                worker.submit(frame_bgr)
                last_submit_at = time.monotonic()
                if args.debug:
                    print("[detect] manual refresh")
    finally:
        worker.stop()
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
