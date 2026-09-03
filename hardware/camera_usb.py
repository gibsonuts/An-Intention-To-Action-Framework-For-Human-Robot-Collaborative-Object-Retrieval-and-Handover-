#!/usr/bin/env python3
from typing import Tuple, Optional, List, Dict, Any
import os
import platform
import time

import cv2
import numpy as np
from PIL import Image

# ---------- Discovery ----------

def _default_backend_for_os() -> int:
    """
    Choose a sane default backend per OS to improve device stability.
    """
    sys = platform.system()
    if sys == "Windows":
        # Media Foundation is modern & stable on Windows 10/11
        return cv2.CAP_MSMF
    elif sys == "Darwin":
        # AVFoundation is default on macOS
        return cv2.CAP_AVFOUNDATION
    else:
        # v4l2 on Linux
        return cv2.CAP_V4L2


def _backend_name(backend: int) -> str:
    names = {
        cv2.CAP_ANY: "ANY",
        getattr(cv2, "CAP_V4L2", -1): "V4L2",
        getattr(cv2, "CAP_DSHOW", -1): "DSHOW",
        getattr(cv2, "CAP_MSMF", -1): "MSMF",
        getattr(cv2, "CAP_AVFOUNDATION", -1): "AVFOUNDATION",
        getattr(cv2, "CAP_VFW", -1): "VFW",
        getattr(cv2, "CAP_GSTREAMER", -1): "GSTREAMER",
        getattr(cv2, "CAP_QT", -1): "QUICKTIME",
        getattr(cv2, "CAP_FFMPEG", -1): "FFMPEG",
    }
    return names.get(backend, str(backend))


def list_usb_cameras(max_indices: int = 10, backend: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Best-effort enumeration of webcam-like devices by probing indices [0..max_indices-1].

    Returns a list of dicts with keys:
      - index: int
      - backend: str (name)
      - name: Optional[str] (Linux only, if discoverable)
      - opened: bool (True if a capture stream opened successfully)

    Notes:
      * Device "names" are not portably available via OpenCV. On Linux, we try sysfs.
      * For robust/authoritative enumeration with real names/paths, consider platform APIs.
    """
    backend = backend if backend is not None else _default_backend_for_os()
    out = []
    for idx in range(max_indices):
        cap = cv2.VideoCapture(idx, backend)
        opened = cap.isOpened()
        name = None
        if platform.system() == "Linux":
            # Try to read the UVC device name from sysfs if present
            sysfs_name = f"/sys/class/video4linux/video{idx}/name"
            if os.path.exists(sysfs_name):
                try:
                    with open(sysfs_name, "r", encoding="utf-8") as f:
                        name = f.read().strip()
                except Exception:
                    name = None
        out.append({
            "index": idx,
            "backend": _backend_name(backend),
            "name": name,
            "opened": bool(opened),
        })
        if opened:
            cap.release()
    return out


# ---------- Camera ----------

class USBCamera:
    """
    Minimal, robust wrapper around cv2.VideoCapture for a single USB/UVC webcam.

    Rough parity with your RealSenseCamera:
      - start(), stop(), warmup()
      - get_rgb() -> HxWx3 uint8
      - from_config()
      - get_intrinsics_and_scale(intrinsics_dict)  <-- best-effort; uses supplied calibration if present.
    """
    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        device_index: Optional[int] = 0,
        backend: Optional[int] = None,
        fourcc: Optional[str] = None,
        convert_to_rgb: bool = True,
        calibration: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            width, height, fps: Desired capture configuration.
            device_index: OpenCV device index (0,1,2,...). If None, we'll try 0.
            backend: Optional cv2 backend (e.g., cv2.CAP_MSMF, cv2.CAP_V4L2, etc.).
            fourcc: Optional FOURCC (e.g., 'MJPG', 'YUY2'). Many webcams prefer 'MJPG'.
            convert_to_rgb: If True, convert BGR->RGB on read.
            calibration: Optional dict with {fx, fy, cx, cy} from prior calibration.
        """
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.device_index = 0 if device_index is None else int(device_index)
        self.backend = backend if backend is not None else _default_backend_for_os()
        self.fourcc = fourcc
        self.convert_to_rgb = bool(convert_to_rgb)

        self._cap: Optional[cv2.VideoCapture] = None
        self._calib = calibration or {}

    @classmethod
    def from_config(cls, cam_cfg: dict) -> "USBCamera":
        intr = cam_cfg.get("intrinsics", {})
        width = int(intr.get("width", cam_cfg.get("width", 1280)))
        height = int(intr.get("height", cam_cfg.get("height", 720)))
        fps = int(cam_cfg.get("fps", 30))
        idx = cam_cfg.get("index") or cam_cfg.get("device_index") or 0

        backend_name = (cam_cfg.get("backend") or "").upper()
        backend_map = {
            "ANY": cv2.CAP_ANY,
            "V4L2": getattr(cv2, "CAP_V4L2", cv2.CAP_ANY),
            "DSHOW": getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY),
            "MSMF": getattr(cv2, "CAP_MSMF", cv2.CAP_ANY),
            "AVFOUNDATION": getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY),
            "GSTREAMER": getattr(cv2, "CAP_GSTREAMER", cv2.CAP_ANY),
        }
        backend = backend_map.get(backend_name) if backend_name else None

        fourcc = cam_cfg.get("fourcc")
        convert_to_rgb = bool(cam_cfg.get("convert_to_rgb", True))

        # If intrinsics are supplied, pass them as calibration so get_intrinsics_and_scale can fill correctly.
        calibration = {}
        for k in ("fx", "fy", "cx", "cy"):
            if k in intr:
                calibration[k] = float(intr[k])

        return cls(
            width=width,
            height=height,
            fps=fps,
            device_index=idx,
            backend=backend,
            fourcc=fourcc,
            convert_to_rgb=convert_to_rgb,
            calibration=calibration or None,
        )

    def start(self) -> None:
        if self._cap is not None:
            return  # already started

        self._cap = cv2.VideoCapture(self.device_index, self.backend)
        if not self._cap.isOpened():
            self._cap.release()
            self._cap = None
            raise RuntimeError(f"Failed to open camera index {self.device_index} "
                               f"with backend {_backend_name(self.backend)}.")

        # Configure stream properties
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
        self._cap.set(cv2.CAP_PROP_FPS, float(self.fps))

        if self.fourcc:
            fourcc_val = cv2.VideoWriter_fourcc(*self.fourcc)
            self._cap.set(cv2.CAP_PROP_FOURCC, float(fourcc_val))

        # Some backends need a small delay to apply settings
        time.sleep(0.05)

        # Verify negotiated settings (not all requests are honored)
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f = int(round(self._cap.get(cv2.CAP_PROP_FPS) or self.fps))
        if (w != self.width) or (h != self.height):
            # Not fatal—just warn via print so you can see what's actually running
            print(f"[USBCamera] Requested {self.width}x{self.height}@{self.fps}, "
                  f"got {w}x{h}@{f}")

        self.width, self.height, self.fps = w, h, f

    def stop(self) -> None:
        try:
            if self._cap is not None:
                self._cap.release()
        except Exception:
            pass
        finally:
            self._cap = None

    def warmup(self, n_frames: int = 30) -> None:
        if self._cap is None:
            raise RuntimeError("Camera not started. Call start() first.")
        for _ in range(max(1, n_frames)):
            self._cap.read()

    def _read_raw(self) -> np.ndarray:
        if self._cap is None:
            raise RuntimeError("Camera not started. Call start() first.")
        ok, frame_bgr = self._cap.read()
        if not ok or frame_bgr is None:
            raise RuntimeError("Failed to grab frame from camera.")
        return frame_bgr

    def get_rgb(self) -> np.ndarray:
        """
        Returns color image as HxWx3 uint8 (RGB).
        """
        frame_bgr = self._read_raw()
        if self.convert_to_rgb:
            return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # If conversion disabled, still ensure output is RGB by swapping channels
        return frame_bgr[:, :, ::-1].copy()

    def get_rgb_file(self) -> np.ndarray:
        frame = self.get_rgb()
        tmp_img_path = "/tmp/frame_llm.jpg"
        Image.fromarray(frame).save(tmp_img_path, "JPEG", quality=95)
        return tmp_img_path

    def get_rgb_jpeg(self,quality=90) -> np.ndarray:
        """
        Returns color image as HxWx3 uint8 (RGB).
        """
        frame_bgr = self._read_raw()
        if self.convert_to_rgb:
            frame =  cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # If conversion disabled, still ensure output is RGB by swapping channels
        ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
             raise RuntimeError("JPEG encode failed")  
        return bytes(buf)

    def get_bgr(self) -> np.ndarray:
        """
        Returns color image as HxWx3 uint8 (BGR) for OpenCV-native processing.
        """
        return self._read_raw()

    def get_intrinsics_and_scale(self, intrinsics: Dict[str, float]) -> None:
        """
        Best-effort population of intrinsics into provided dict.

        For USB webcams, OpenCV does not expose camera intrinsics. This method will:
          * Use any supplied calibration values (fx, fy, cx, cy) passed in via `calibration`
            (e.g., loaded from a checkerboard calibration you ran previously).
          * Otherwise, assume principal point is the image centre and leave fx/fy unset.
          * Set a dummy 'depth_scale' to 1.0 to keep parity with code paths that expect it.

        Side-effects:
            intrinsics['width'], intrinsics['height'] are set.
            intrinsics['cx'], intrinsics['cy'] are set to centre if not provided.
            intrinsics['fx'], intrinsics['fy'] kept if provided, else left unchanged if present,
            else omitted.
            intrinsics['depth_scale'] = 1.0
        """
        intrinsics['width'] = float(self.width)
        intrinsics['height'] = float(self.height)
        if 'cx' not in self._calib:
            intrinsics['cx'] = float(self.width) / 2.0
        if 'cy' not in self._calib:
            intrinsics['cy'] = float(self.height) / 2.0

        # Apply provided calibration (if any)
        for k in ('fx', 'fy', 'cx', 'cy'):
            if k in self._calib:
                intrinsics[k] = float(self._calib[k])

        # No depth on a regular webcam, but some pipelines expect a value
        intrinsics['depth_scale'] = 1.0

    # ---- Optional helpers for common UVC controls (best-effort; backend dependent) ----

    def set_auto_exposure(self, enable: bool) -> bool:
        """
        Toggle auto exposure if the backend supports it.
        Returns True if the underlying set() appeared to succeed.
        """
        if self._cap is None:
            raise RuntimeError("Camera not started.")
        # OpenCV semantics vary by backend; this is best-effort.
        return bool(self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75 if enable else 0.25))

    def set_exposure(self, value: float) -> bool:
        if self._cap is None:
            raise RuntimeError("Camera not started.")
        return bool(self._cap.set(cv2.CAP_PROP_EXPOSURE, float(value)))

    def set_auto_focus(self, enable: bool) -> bool:
        if self._cap is None:
            raise RuntimeError("Camera not started.")
        # Some backends: 0=manual, 1=auto
        return bool(self._cap.set(cv2.CAP_PROP_AUTOFOCUS, 1.0 if enable else 0.0))

    def set_focus(self, value: float) -> bool:
        if self._cap is None:
            raise RuntimeError("Camera not started.")
        return bool(self._cap.set(cv2.CAP_PROP_FOCUS, float(value)))


# ---------- Multi-Cam Rig ----------

class USBRig:
    """
    Optional helper to manage multiple webcams by index.
    API mirrors your RealSenseRig shape where sensible.
    """
    def __init__(
        self,
        indices: List[int],
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        backend: Optional[int] = None,
        align_depth_to_color: Optional[bool] = None,  # kept for API parity; unused
        per_camera_calibration: Optional[Dict[int, Dict[str, float]]] = None,
        fourcc: Optional[str] = None,
        convert_to_rgb: bool = True,
    ):
        self.backend = backend if backend is not None else _default_backend_for_os()
        self.cams: Dict[int, USBCamera] = {
            i: USBCamera(
                width=width,
                height=height,
                fps=fps,
                device_index=i,
                backend=self.backend,
                fourcc=fourcc,
                convert_to_rgb=convert_to_rgb,
                calibration=(per_camera_calibration or {}).get(i),
            )
            for i in indices
        }

    @staticmethod
    def discover_indices(max_indices: int = 10) -> List[int]:
        return [d["index"] for d in list_usb_cameras(max_indices=max_indices) if d["opened"]]

    def start_all(self) -> None:
        for cam in self.cams.values():
            cam.start()

    def stop_all(self) -> None:
        for cam in self.cams.values():
            cam.stop()

    def get_rgb(self, index: int) -> np.ndarray:
        return self.cams[index].get_rgb()

    def get_bgr(self, index: int) -> np.ndarray:
        return self.cams[index].get_bgr()

    def get_intrinsics_and_scale(self, index: int, intrinsics: Dict[str, float]) -> None:
        self.cams[index].get_intrinsics_and_scale(intrinsics)


# ---------- Convenience: estimating intrinsics from a known FOV (optional) ----------

def intrinsics_from_fov(width: int, height: int, fov_x_deg: Optional[float] = None, fov_y_deg: Optional[float] = None) -> Dict[str, float]:
    """
    If you know your lens horizontal or vertical FOV, you can estimate (fx, fy, cx, cy).
    Provide either fov_x_deg or fov_y_deg (or both). Assumes square pixels if only one is given.

    Returns: dict with fx, fy, cx, cy
    """
    cx, cy = width / 2.0, height / 2.0

    fx = fy = None
    if fov_x_deg is not None:
        fx = width / (2.0 * np.tan(np.deg2rad(fov_x_deg) / 2.0))
    if fov_y_deg is not None:
        fy = height / (2.0 * np.tan(np.deg2rad(fov_y_deg) / 2.0))

    if fx is None and fy is not None:
        fx = fy  # assume square pixels
    if fy is None and fx is not None:
        fy = fx

    if fx is None or fy is None:
        raise ValueError("Provide at least one of fov_x_deg or fov_y_deg.")

    return {"fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy)}


# ---------- Example usage (comment out in production) ----------
if __name__ == "__main__":
    cams = list_usb_cameras()
    print("Detected cameras:", cams)

    cam = USBCamera(width=1280, height=720, fps=30, device_index=0, fourcc="MJPG")
    cam.start()
    cam.warmup(15)

    intr = {}
    cam.get_intrinsics_and_scale(intr)
    print("Intrinsics (best-effort):", intr)

    img = cam.get_rgb()
    print("Frame:", img.shape, img.dtype)

    cam.stop()
