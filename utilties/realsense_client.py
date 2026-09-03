#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, time
import numpy as np
import cv2

def visualize_depth(depth_mm_u16: np.ndarray) -> np.ndarray:
    depth_m = depth_mm_u16.astype(np.float32) / 1000.0
    vis = np.clip((depth_m - 0.2) / (3.0 - 0.2), 0, 1)
    vis_u8 = (vis * 255).astype(np.uint8)
    return cv2.applyColorMap(vis_u8, cv2.COLORMAP_JET)

def overlay_delay(img: np.ndarray, delay_ms: float, label: str):
    txt = f"{label}  delay: {int(delay_ms)} ms"
    cv2.rectangle(img, (5, 5), (5 + 260, 35), (0, 0, 0), -1)
    cv2.putText(img, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

def main():
    parser = argparse.ArgumentParser(description="RealSense stream client")
    parser.add_argument("--server-ip", type=str, default='192.168.10.100')
    parser.add_argument("--port", type=int, default=5552)
    parser.add_argument("--topic-prefix", default="realsense")
    parser.add_argument(
        "--id",
        type=str,
        default=None,
        help="Optional camera id/serial to filter. Omit it to show all cameras.",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    try:
        import zmq
    except ImportError:
        raise SystemExit(
            "[ERROR] Missing dependency 'pyzmq'. Install it to use the RealSense client."
        )

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.RCVHWM, 2 * 1024)
    # Optional: don't block forever
    sock.setsockopt(zmq.RCVTIMEO, 2000)  # 2s receive timeout
    sock.connect(f"tcp://{args.server_ip}:{args.port}")
    # Subscribe to one camera if requested, otherwise subscribe to the full prefix.
    if args.id:
        topic = f"{args.topic_prefix}/{args.id}/"
    else:
        topic = f"{args.topic_prefix}/"
    print(topic)
    sock.setsockopt_string(zmq.SUBSCRIBE, topic)

    print('starting topic_prefix ', f"tcp://{args.server_ip}:{args.port}")

    print(f"[client] Connected to tcp://{args.server_ip}:{args.port}")

    while True:
        try:
            frames = sock.recv_multipart()  # tolerate any N>=3 frames
        except zmq.Again:
            # timeout just to keep UI responsive
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue
        except KeyboardInterrupt:
            break

        if len(frames) < 3:
            if args.debug:
                print(f"[client] Ignoring short message with {len(frames)} frames")
            continue

        # Robust unpack: topic, header, and the rest is payload (concat)
        topic_b = frames[0]
        header_b = frames[1]
        payload_b = b"".join(frames[2:])

        try:
            topic = topic_b.decode("utf-8", errors="ignore")
            header = json.loads(header_b.decode("utf-8"))
        except Exception as e:
            if args.debug:
                print(f"[client] Failed to parse header/topic: {e}, topic raw={topic_b[:32]!r}")
            continue

        cam = header.get("cam", "unknown")
        stream = header.get("stream", "unknown")
        encoding = header.get("encoding", "")
        now_ms = int(time.time() * 1000)
        server_ts_ms = int(header.get("server_ts_ms", now_ms))
        delay_ms = now_ms - server_ts_ms

        if args.debug:
            print(f"[client] {topic} frames={len(frames)} stream={stream} enc={encoding} delay={delay_ms}ms")

        if stream == "color" and encoding == "jpg":
            buf = np.frombuffer(payload_b, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is None:
                if args.debug: print("[client] JPEG decode failed")
                continue
            overlay_delay(img, delay_ms, f"{cam} RGB")
            cv2.imshow(f"{cam} - RGB", img)

        elif stream == "depth" and encoding.startswith("png16"):
            buf = np.frombuffer(payload_b, dtype=np.uint8)
            depth_mm = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
            if depth_mm is None:
                if args.debug: print("[client] PNG decode failed")
                continue
            if depth_mm.dtype != np.uint16:
                depth_mm = depth_mm.astype(np.uint16, copy=False)
            vis = visualize_depth(depth_mm)
            overlay_delay(vis, delay_ms, f"{cam} DEPTH")
            cv2.imshow(f"{cam} - DEPTH", vis)

        # ESC to quit
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
