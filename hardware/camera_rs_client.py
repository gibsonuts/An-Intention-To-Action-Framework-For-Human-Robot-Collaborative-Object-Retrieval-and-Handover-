#!/usr/bin/env python3
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import threading, time, json, argparse
import numpy as np
import zmq
import cv2
import sys 

# ====== Helpers for visualization ======
def _visualize_depth_mm(depth_mm_u16: np.ndarray) -> np.ndarray:
    """Convert uint16 depth in millimeters to a viewable BGR heatmap (JET)."""
    depth_m = depth_mm_u16.astype(np.float32) / 1000.0
    vis = np.clip((depth_m - 0.2) / (3.0 - 0.2), 0, 1)
    vis_u8 = (vis * 255).astype(np.uint8)
    return cv2.applyColorMap(vis_u8, cv2.COLORMAP_JET)


def _overlay_delay(img, delay_ms, label: str) -> None:
    # Ensure cv::Mat-compatible buffer
    if img.dtype != np.uint8:
        img = img.astype(np.uint8, copy=False)
    img = np.ascontiguousarray(img)

    txt = f"{label}  delay: {int(delay_ms)} ms"
    cv2.rectangle(img, (5, 5), (5 + 300, 35), (0, 0, 0), -1)
    cv2.putText(img, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)

def _check_server_reachable(server_ip: str, port: int, topic_prefix: str, timeout_ms: int = 2000) -> bool:
    """
    Try to connect to the PUB server and receive a single multipart message
    within timeout_ms. Returns True if something is received, False otherwise.
    """
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    try:
        sock.setsockopt(zmq.RCVHWM, 1)
        sock.setsockopt(zmq.RCVTIMEO, timeout_ms)

        # Subscribe to all cameras under this prefix
        topic = f"{topic_prefix}/"
        sock.setsockopt_string(zmq.SUBSCRIBE, topic)

        sock.connect(f"tcp://{server_ip}:{port}")
        # Will raise zmq.Again if no message arrives within timeout_ms
        _ = sock.recv_multipart()
        return True
    except zmq.Again:
        # Timed out waiting for any data; treat as unreachable
        return False
    except Exception as e:
        print(f"Error while checking server reachability: {e}", file=sys.stderr)
        return False
    finally:
        try:
            sock.close(0)
        except Exception:
            pass

# ====== Network camera subscriber ======
class NetworkRealSenseCamera:
    """
    Network-backed "camera" that receives frames from a ZeroMQ PUB server.

    Matches the spirit of your RealSenseCamera API:
      - start(), stop(), warmup()
      - get_rgbd() -> (depth_uint16_mm, color_rgb_uint8)
      - get_rgb_jpeg()

    Configure with camera_id (e.g., "123134654") to filter topics to that cam.
    """
    def __init__(
        self,
        intrinsics,
        server_ip: str,
        port: int = 5555,
        topic_prefix: str = "realsense",
        camera_id: Optional[str] = None,   # set to "camera_arm" or "camera_fixed" to filter
        recv_timeout_ms: int = 2000,
        image_mask: Optional[np.ndarray] = None,

    ):  
        
        if not _check_server_reachable(server_ip, port, topic_prefix):
            print(
                f"ERROR: Could not receive any data from tcp://{server_ip}:{port} "
                f"with topic prefix '{topic_prefix}/'. Is the realsense camera server running and publishing?"
            )
            sys.exit(1)

        print('starting realsense server_ip ', server_ip)
        print('starting camera_id ', camera_id)
        print('starting topic_prefix ', topic_prefix)
        self.server_ip = server_ip
        self.port = port
        self.topic_prefix = topic_prefix.rstrip("/")
        self.camera_id = camera_id
        self.recv_timeout_ms = recv_timeout_ms

        self._ctx: Optional[zmq.Context] = None
        self._sock: Optional[zmq.Socket] = None

        self._thread: Optional[threading.Thread] = None
        self._running: bool = False
        self._lock = threading.Lock()
        self.intrinsics = intrinsics
        # Latest frames + timestamps
        self._latest_depth_mm: Optional[np.ndarray] = None  # uint16 millimeters
        self._latest_color_rgb: Optional[np.ndarray] = None
        self._latest_server_ts_ms_color: Optional[int] = None
        self._latest_server_ts_ms_depth: Optional[int] = None
        shape_hw = None
        if isinstance(intrinsics, dict):
            width = intrinsics.get("width")
            height = intrinsics.get("height")
            if width and height:
                shape_hw = (int(height), int(width))
        self._image_mask = self._normalize_mask(image_mask, shape_hw)

    @staticmethod
    def _normalize_mask(mask: Optional[np.ndarray], shape_hw: Optional[Tuple[int, int]]) -> Optional[np.ndarray]:
        if mask is None:
            return None
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if shape_hw is not None and mask.shape != shape_hw:
            mask = cv2.resize(mask, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
        if mask.dtype != np.uint8:
            mask = np.clip(mask, 0, 255).astype(np.uint8)
        return np.where(mask > 127, 255, 0).astype(np.uint8)

    @classmethod
    def load_mask_from_config(cls, cam_cfg: dict) -> Optional[np.ndarray]:
        mask_path = cam_cfg.get("image_mask_path")
        if not mask_path:
            return None

        mask_file = Path(mask_path)
        if not mask_file.is_absolute():
            mask_file = Path(__file__).resolve().parent / "config" / mask_file

        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Failed to load camera image mask: {mask_file}")
        return mask

    def _apply_image_mask(self, depth: np.ndarray, color_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self._image_mask is None:
            return depth, color_rgb
        if self._image_mask.shape != depth.shape:
            mask = self._normalize_mask(self._image_mask, depth.shape)
        else:
            mask = self._image_mask
        masked_depth = np.where(mask > 0, depth, 0).astype(depth.dtype, copy=False)
        masked_color = cv2.bitwise_and(color_rgb, color_rgb, mask=mask)
        return masked_depth, masked_color

    def start(self) -> None:
        if self._running:
            return
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.RCVHWM, 2048)
        self._sock.setsockopt(zmq.RCVTIMEO, self.recv_timeout_ms)

        self._sock.connect(f"tcp://{self.server_ip}:{self.port}")
        print('self.camera_id',self.camera_id)
        # Topic subscription
        if self.camera_id:
            topic = f"{self.topic_prefix}/{self.camera_id}/"
        else:
            topic = f"{self.topic_prefix}/"
        self._sock.setsockopt_string(zmq.SUBSCRIBE, topic)

        self._running = True
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._sock is not None:
            try:
                self._sock.close(0)
            except Exception:
                pass
            self._sock = None

    def warmup(self, n_frames: int = 60) -> None:
        # Just wait ~n_frames/fps so the receive loop has time to populate latest frames
        time.sleep(n_frames * 0.033)

    def _recv_loop(self):
        last_warn = 0
        while self._running:
            try:
                frames = self._sock.recv_multipart()
                
            except zmq.Again:
                now = time.time()
                if now - last_warn > 2.0:
                    print(self.camera_id, "fail to read")
                    last_warn = now
                continue
            except Exception as e:
                # Avoid hot-looping on errors
                time.sleep(0.01)
                print(e)
                continue

            if len(frames) < 3:
                continue

            topic_b = frames[0]
            header_b = frames[1]
            payload_b = b"".join(frames[2:])  # robust to extra envelope frames

            # Parse header
            try:
                header = json.loads(header_b.decode("utf-8"))
            except Exception:
                continue

            # Optionally enforce camera filter at the header level
            cam = header.get("cam")
            if self.camera_id and cam != self.camera_id:
                continue

            stream = header.get("stream")
            encoding = header.get("encoding")
            server_ts_ms = int(header.get("server_ts_ms", time.time() * 1000))

            if stream == "color" and encoding == "jpg":
                buf = np.frombuffer(payload_b, dtype=np.uint8)
                img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if img_bgr is None:
                    continue
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                with self._lock:
                    self._latest_color_rgb = img_rgb
                    self._latest_server_ts_ms_color = server_ts_ms

            elif stream == "depth" and encoding.startswith("png16"):
                buf = np.frombuffer(payload_b, dtype=np.uint8)
                depth_mm = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
                if depth_mm is None:
                    continue
                if depth_mm.dtype != np.uint16:
                    depth_mm = depth_mm.astype(np.uint16, copy=False)
                with self._lock:
                    self._latest_depth_mm = depth_mm
                    self._latest_server_ts_ms_depth = server_ts_ms

    # --- API mirror ---
    def get_rgbd(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (depth_mm: HxW uint16, color_rgb: HxWx3 uint8) from the latest received frames.
        """
        with self._lock:
            if self._latest_depth_mm is None or self._latest_color_rgb is None:
                raise RuntimeError("No frames received yet.")
            depth = self._latest_depth_mm.copy()
            color = self._latest_color_rgb.copy()
        return self._apply_image_mask(depth, color)

    def get_rgb_jpeg(self, quality: int = 90) -> bytes:
        _, color_rgb = self.get_rgbd()
        # color_rgb is already RGB
        ok, buf = cv2.imencode(".jpg", color_rgb[:, :, ::-1],  # convert back to BGR for cv2.imencode
                               [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            raise RuntimeError("JPEG encode failed")
        return bytes(buf)

    # Convenience for latency if you want it programmatically
    def get_delays_ms(self) -> Dict[str, Optional[int]]:
        now_ms = int(time.time() * 1000)
        with self._lock:
            c = now_ms - self._latest_server_ts_ms_color if self._latest_server_ts_ms_color else None
            d = now_ms - self._latest_server_ts_ms_depth if self._latest_server_ts_ms_depth else None
        return {"color_ms": c, "depth_ms": d}

# ====== Multi-camera rig that feels like your RealSenseRig ======
class NetworkRealSenseRig:
    """
    Manage multiple network cameras (identified by camera_id, e.g., 'camera_arm', 'camera_fixed').
    """
    def __init__(self, server_ip: str, cam_names: List[str], port: int = 5555, topic_prefix: str = "realsense"):
        self.cams: Dict[str, NetworkRealSenseCamera] = {
            name: NetworkRealSenseCamera(intrinsics=None, server_ip=server_ip, port=port, topic_prefix=topic_prefix, camera_id=name)
            for name in cam_names
        }

    def start_all(self) -> None:
        for c in self.cams.values():
            c.start()

    def stop_all(self) -> None:
        for c in self.cams.values():
            c.stop()

    def get_rgbd(self, cam_name: str) -> Tuple[np.ndarray, np.ndarray]:
        return self.cams[cam_name].get_rgbd()

    def get_delays_ms(self, cam_name: str) -> Dict[str, Optional[int]]:
        return self.cams[cam_name].get_delays_ms()

# ====== Optional demo GUI ======
def _demo_gui(server_ip: str, port: int, topic_prefix: str):
    # Subscribe to all cameras (leave camera_id=None) and visualize streams with delay
    sub_all = NetworkRealSenseCamera(server_ip, port, topic_prefix, camera_id=None)
    sub_all.start()
    print(f"[client] Connected to tcp://{server_ip}:{port}  (prefix='{topic_prefix}/')")

    try:
        while True:
            # Pull the latest frames if available and show them.
            # Since we don't know the names ahead of time here, we’ll just try to render
            # the last received frames (if any). For a known set, use NetworkRealSenseRig.
            with sub_all._lock:
                color = sub_all._latest_color_rgb.copy() if sub_all._latest_color_rgb is not None else None
                depth = sub_all._latest_depth_mm.copy() if sub_all._latest_depth_mm is not None else None
                c_ts = sub_all._latest_server_ts_ms_color
                d_ts = sub_all._latest_server_ts_ms_depth

            now_ms = int(time.time() * 1000)
            if color is not None:
                # color_bgr = color[:, :, ::-1]
                color_bgr = color[:, :, ::-1].copy()
                delay_c = now_ms - c_ts if c_ts else 0
                _overlay_delay(color_bgr, delay_c, "RGB")
                cv2.imshow("RGB (latest of any cam)", color_bgr)

            if depth is not None:
                vis = _visualize_depth_mm(depth)
                delay_d = now_ms - d_ts if d_ts else 0
                _overlay_delay(vis, delay_d, "DEPTH")
                cv2.imshow("DEPTH (latest of any cam)", vis)

            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        sub_all.stop()
        cv2.destroyAllWindows()

# ====== CLI ======
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Network RealSense client (subscriber)")
    parser.add_argument("--server-ip", default=  '192.168.10.100')
    parser.add_argument("--port", type=int, default=5552)
    parser.add_argument("--topic-prefix", default="realsense")
    parser.add_argument("--cams", nargs="*", default=[], help="Optional camera names (e.g. camera_drill camera_fixed camera_gripper)")
    args = parser.parse_args()

  
    if not args.cams:
        print("No --demo and no --cams provided. Example:")
        print("  python3 network_realsense_client.py --server-ip 192.168.10.100 --cams camera_arm camera_fixed")
        exit(0)

    rig = NetworkRealSenseRig(args.server_ip, args.cams, args.port, args.topic_prefix)
    rig.start_all()
    print(f"[client] Started for cams: {args.cams}")
    try:
        while True:
            for name in args.cams:
                try:
                    depth_mm, color_rgb = rig.get_rgbd(name)
                    delays = rig.get_delays_ms(name)
                    # Quick preview (press ESC to exit)
                    color_bgr = color_rgb[:, :, ::-1]
                    _overlay_delay(color_bgr, delays.get("color_ms") or 0, f"{name} RGB")
                    vis = _visualize_depth_mm(depth_mm)
                    _overlay_delay(vis, delays.get("depth_ms") or 0, f"{name} DEPTH")
                    cv2.imshow(f"{name} - RGB", color_bgr)
                    cv2.imshow(f"{name} - DEPTH", vis)
                except Exception:
                    pass
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        rig.stop_all()
        cv2.destroyAllWindows()
