#!/usr/bin/env python3
import argparse, json, queue, signal, sys, threading, time
from typing import Dict, Any
import yaml
import numpy as np
import zmq
import cv2

# pyrealsense2 is typically installed as: pip install pyrealsense2
import pyrealsense2 as rs

STOP = False

def handle_sigint(signum, frame):
    global STOP
    STOP = True

signal.signal(signal.SIGINT, handle_sigint)
signal.signal(signal.SIGTERM, handle_sigint)


def setup_pipeline(cfg: Dict[str, Any]):
    """Create and start a RealSense pipeline with optional alignment."""
    intr = cfg["intrinsics"]
    width = int(intr["width"])
    height = int(intr["height"])
    fps = int(intr.get("fps", 30))
    serial = cfg.get("serial", None)
    align_to_color = bool(cfg.get("align_depth_to_color", False))
    depth_scale = float(intr.get("depth_scale", 0.001))

    pipeline = rs.pipeline()
    rs_config = rs.config()
    if serial:
        rs_config.enable_device(serial)
    rs_config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    # Use Z16 for depth; we’ll optionally align to color
    rs_config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    profile = pipeline.start(rs_config)

    # Fetch actual depth scale from device if available
    depth_sensor = profile.get_device().first_depth_sensor()
    if depth_sensor:
        try:
            depth_scale_hw = depth_sensor.get_depth_scale()
            if depth_scale_hw > 0:
                depth_scale = depth_scale_hw
        except Exception:
            pass

    align = rs.align(rs.stream.color) if align_to_color else None
    return {
        "camera_id": serial,
        "pipeline": pipeline,
        "align": align,
        "depth_scale": depth_scale,
        "intrinsics": intr,
    }


def encode_frames(color_bgr: np.ndarray, depth_z16: np.ndarray, depth_scale: float):
    """
    - Color: JPEG encoded.
    - Depth: convert meters->millimeters (uint16) and PNG-encode to preserve data.
    """
    # Color -> JPEG
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

    ok_jpg, color_buf = cv2.imencode(".jpg", color_rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok_jpg:
        raise RuntimeError("Failed to JPEG-encode color frame")

    # Depth in meters = depth_z16 * depth_scale. Convert to millimeters and keep uint16.
    depth_mm = (depth_z16.astype(np.float32) * depth_scale * 1000.0).round().astype(np.uint16)
    ok_png, depth_buf = cv2.imencode(".png", depth_mm)  # lossless 16-bit PNG
    if not ok_png:
        raise RuntimeError("Failed to PNG-encode depth frame")

    return color_buf.tobytes(), depth_buf.tobytes(), depth_mm.shape


def publisher_thread(
    cam_ctx: Dict[str, Any],
    topic_prefix: str,
    out_queue: "queue.Queue[list[bytes]]",
    crash_event: threading.Event,
    crash_info: Dict[str, str],
):
    """Grab frames and publish them with a JSON header and encoded payload."""
    id = cam_ctx["camera_id"]
    pipeline: rs.pipeline = cam_ctx["pipeline"]
    align = cam_ctx["align"]
    depth_scale = cam_ctx["depth_scale"]

    topic_color = f"{topic_prefix}/{id}/color".encode("utf-8")
    topic_depth = f"{topic_prefix}/{id}/depth".encode("utf-8")

    try:
        while not STOP:
            frames: rs.composite_frame = pipeline.wait_for_frames(timeout_ms=1000)
            if frames is None:
                continue
            if align is not None:
                frames = align.process(frames)

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            # Convert to numpy
            color = np.asanyarray(color_frame.get_data())  # BGR
            depth = np.asanyarray(depth_frame.get_data())  # uint16 Z16

            server_ts_ms = int(time.time() * 1000)

            color_bytes, depth_bytes, depth_shape = encode_frames(color, depth, depth_scale)

            # Headers
            h_color = {
                "cam": id,
                "stream": "color",
                "encoding": "jpg",
                "shape": [int(color.shape[0]), int(color.shape[1]), 3],
                "dtype": "uint8",
                "server_ts_ms": server_ts_ms,
            }
            h_depth = {
                "cam": id,
                "stream": "depth",
                "encoding": "png16_mm",  # depth in millimeters, uint16
                "shape": [int(depth_shape[0]), int(depth_shape[1])],
                "dtype": "uint16",
                "server_ts_ms": server_ts_ms,
            }

            # Publish multipart: [topic, header_json, payload]
            for msg in (
                [topic_color, json.dumps(h_color).encode("utf-8"), color_bytes],
                [topic_depth, json.dumps(h_depth).encode("utf-8"), depth_bytes],
            ):
                try:
                    out_queue.put_nowait(msg)
                except queue.Full:
                    try:
                        _ = out_queue.get_nowait()
                    except queue.Empty:
                        pass
                    out_queue.put_nowait(msg)
    except Exception as e:
        crash_info["camera_id"] = str(id)
        crash_info["error"] = repr(e)
        crash_event.set()
        print(f"[{id}] publisher thread error:", e, file=sys.stderr)
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass


def run_server_once(args: argparse.Namespace):
    global STOP
    # Load YAML
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    camera_cfg = cfg['realsense']
    listen_port = camera_cfg['listen_port']
    # Prepare ZMQ PUB
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUB)
    sock.setsockopt(zmq.SNDHWM, 2 * 1024)
    sock.bind(f"tcp://{args.bind}:{listen_port}")
    print(f"[server] Publishing on tcp://{args.bind}:{listen_port} with prefix '{args.topic_prefix}'")

    # Setup pipelines for each top-level camera block
    cam_threads = []
    out_queue: "queue.Queue[list[bytes]]" = queue.Queue(maxsize=256)
    crash_event = threading.Event()
    crash_info: Dict[str, str] = {}
    for cam_name, cam_cfg in camera_cfg.items():
        if not isinstance(cam_cfg, dict):
            continue
        if cam_cfg.get("model", "").lower().startswith("realsense"):
            cam_ctx = setup_pipeline( cam_cfg)
            t = threading.Thread(
                target=publisher_thread,
                args=(cam_ctx, args.topic_prefix, out_queue, crash_event, crash_info),
                daemon=True,
            )
            t.start()
            cam_threads.append(t)
            print(f"[server] Started camera '{cam_ctx['camera_id']}'")

    # Keep main thread alive
    try:
        while not STOP:
            try:
                msg = out_queue.get(timeout=0.2)
                sock.send_multipart(msg)
                while True:
                    msg = out_queue.get_nowait()
                    sock.send_multipart(msg)
            except queue.Empty:
                pass
            if crash_event.is_set():
                raise RuntimeError(
                    f"Camera publisher crashed for {crash_info.get('camera_id', 'unknown')}: "
                    f"{crash_info.get('error', 'unknown error')}"
                )
            for t in cam_threads:
                if not t.is_alive() and not crash_event.is_set():
                    raise RuntimeError("A RealSense publisher thread exited unexpectedly.")
            time.sleep(0.2)
    finally:
        print("[server] Shutting down...")
        for t in cam_threads:
            t.join(timeout=2.0)
        sock.close(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RealSense RGB+Depth streaming server")
    parser.add_argument("--config", "-c", default="hardware.yaml", help="YAML config file")
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--topic-prefix", default="realsense")
    parser.add_argument("--restart-delay", type=float, default=2.0, help="Seconds to wait before restarting after a crash.")
    return parser.parse_args()


def main():
    global STOP
    args = parse_args()
    while not STOP:
        try:
            run_server_once(args)
            break
        except KeyboardInterrupt:
            STOP = True
            break
        except Exception as e:
            if STOP:
                break
            print(f"[server] Crash detected: {e}", file=sys.stderr)
            print(f"[server] Restarting in {args.restart_delay:.1f}s...", file=sys.stderr)
            time.sleep(max(0.1, float(args.restart_delay)))

if __name__ == "__main__":
    main()
