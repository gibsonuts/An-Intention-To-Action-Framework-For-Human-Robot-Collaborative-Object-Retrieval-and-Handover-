# #!/usr/bin/env python3
# from typing import Tuple, Optional, List, Dict
# import numpy as np
# import pyrealsense2 as rs
# from PIL import Image
# import cv2

# def list_realsense_devices() -> List[Dict[str, str]]:
#     """
#     Return a list of connected RealSense devices with basic info.
#     """
#     ctx = rs.context()
#     out = []
#     for dev in ctx.query_devices():
#         d = {
#             "name": dev.get_info(rs.camera_info.name),
#             "serial": dev.get_info(rs.camera_info.serial_number),
#             "usb_type": dev.get_info(rs.camera_info.usb_type_descriptor)
#         }
#         out.append(d)
#     return out


# class RealSenseCamera:
#     def __init__(
#         self,
#         width: int=1280,
#         height: int=720,
#         fps: int=30,
#         align_depth_to_color: bool = True,
#         device_serial: Optional[str] = None,
#     ):
#         """
#         If multiple cameras are connected, pass `device_serial` to bind this instance
#         to one specific camera. You can get serials from `list_realsense_devices()`.
#         """
#         self.width = width
#         self.height = height
#         self.align_depth_to_color = align_depth_to_color
#         self.fps = fps
#         self.device_serial = device_serial

#         self.pipeline: Optional[rs.pipeline] = None
#         self.profile: Optional[rs.pipeline_profile] = None
#         self._align: Optional[rs.align] = None

#     @classmethod
#     def from_config(cls, cam_cfg: dict) -> "RealSenseCamera":
#         intr = cam_cfg.get("intrinsics", {})
#         width = intr.get("width", 1280)
#         height = intr.get("height", 720)
#         fps = intr.get("fps", 30)
#         align = bool(cam_cfg.get("align_depth_to_color", False))
#         serial = cam_cfg.get("serial") or cam_cfg.get("device_serial")
#         fps = intr.get("fps", 30)
#         return cls(width=width, height=height,fps=fps, align_depth_to_color=align, device_serial=serial)

#     def start(self) -> None:
#         if self.pipeline is not None:
#             return  # already started
#         self.pipeline = rs.pipeline()
#         rs_cfg = rs.config()

#         # If a specific device is requested, bind the pipeline to it
#         if self.device_serial:
#             print('starting cam',self.device_serial)
#             rs_cfg.enable_device(self.device_serial)
#         else:
#             print('starting rs camera auto')

#         rs_cfg.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
#         rs_cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)

#         self.profile = self.pipeline.start(rs_cfg)
#         self._align = rs.align(rs.stream.color) if self.align_depth_to_color else None
  
#     def stop(self) -> None:
#         try:
#             if self.pipeline is not None:
#                 self.pipeline.stop()
#         except Exception:
#             pass
#         finally:
#             self.pipeline = None
#             self.profile = None
#             self._align = None

#     def warmup(self, n_frames: int = 60) -> None:
#         if not self.pipeline:
#             raise RuntimeError("Camera not started. Call start() first.")
#         for _ in range(n_frames):
#             self.pipeline.wait_for_frames()

#     def get_intrinsics_and_scale(self,intrinsics):
#         if not self.profile:
#             raise RuntimeError("Camera not started. Call start() first.")
#         s = self.profile.get_stream(rs.stream.color)
#         intr = s.as_video_stream_profile().get_intrinsics()
#         cx, cy, fx, fy = intr.ppx, intr.ppy, intr.fx, intr.fy
#         depth_sensor = self.profile.get_device().first_depth_sensor()
#         scale = float(depth_sensor.get_depth_scale())
#         intrinsics['fx'] = fx
#         intrinsics['fy'] = fy
#         intrinsics['cx'] = cx
#         intrinsics['cy'] = cy        
#         intrinsics['depth_scale'] = scale        
#         return
 
#     def get_rgbd(self) -> Tuple[np.ndarray, np.ndarray]:
#         """
#         Returns (depth: HxW uint16, color_rgb: HxWx3 uint8)
#         """
#         if not self.pipeline:
#             raise RuntimeError("Camera not started. Call start() first.")
#         frameset = self.pipeline.wait_for_frames()
#         if self._align:
#             frameset = self._align.process(frameset)

#         depth_frame = frameset.get_depth_frame()
#         color_frame = frameset.get_color_frame()
#         if not depth_frame or not color_frame:
#             raise RuntimeError("Failed to acquire frames from camera.")
#         else:
#             print('frames exist')
#         depth = np.asanyarray(depth_frame.get_data())
#         color_bgr = np.asanyarray(color_frame.get_data())
#         color_rgb = color_bgr[:, :, ::-1].copy()
#         return depth, color_rgb


#     def get_rgb_file(self) -> np.ndarray:
#         depth,color_rgb = self.get_rgbd()
#         tmp_img_path = "/tmp/frame_llm.jpg"
#         Image.fromarray(color_rgb).save(tmp_img_path, "JPEG", quality=95)
#         return tmp_img_path

#     def get_rgb_jpeg(self,quality=90) -> np.ndarray:
#         """
#         Returns color image as HxWx3 uint8 (RGB).
#         """
#         depth,color_rgb = self.get_rgbd()
#         # If conversion disabled, still ensure output is RGB by swapping channels
#         ok, buf = cv2.imencode(".jpg", color_rgb, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
#         if not ok:
#              raise RuntimeError("JPEG encode failed")  
#         return bytes(buf)

# class RealSenseRig:
#     """
#     Optional helper to manage multiple cameras simultaneously.
#     Create once with a list of serials, then get frames/params by serial.
#     """
#     def __init__(self, serials: List[str], width: int = 1280, height: int = 720, fps: int = 30, align_depth_to_color: bool = True):
#         self.cams: Dict[str, RealSenseCamera] = {
#             s: RealSenseCamera(width=width, height=height, fps=fps, align_depth_to_color=align_depth_to_color, device_serial=s)
#             for s in serials
#         }

#     @staticmethod
#     def discover_serials() -> List[str]:
#         return [d["serial"] for d in list_realsense_devices()]

#     def start_all(self) -> None:
#         for cam in self.cams.values():
#             cam.start()

#     def stop_all(self) -> None:
#         for cam in self.cams.values():
#             cam.stop()

#     def get_rgbd(self, serial: str) -> Tuple[np.ndarray, np.ndarray]:
#         return self.cams[serial].get_rgbd()

#     def get_intrinsics_and_scale(self, serial: str) -> Tuple[float, float, float, float, float]:
#         return self.cams[serial].get_intrinsics_and_scale()
#!/usr/bin/env python3
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import numpy as np
import pyrealsense2 as rs
from PIL import Image
import cv2
import threading
import time


def list_realsense_devices() -> List[Dict[str, str]]:
    """
    Return a list of connected RealSense devices with basic info.
    """
    ctx = rs.context()
    out = []
    for dev in ctx.query_devices():
        d = {
            "name": dev.get_info(rs.camera_info.name),
            "serial": dev.get_info(rs.camera_info.serial_number),
            "usb_type": dev.get_info(rs.camera_info.usb_type_descriptor),
        }
        out.append(d)
    return out


class RealSenseCamera:
    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        align_depth_to_color: bool = True,
        device_serial: Optional[str] = None,
        image_mask: Optional[np.ndarray] = None,
    ):
        """
        If multiple cameras are connected, pass `device_serial` to bind this instance
        to one specific camera. You can get serials from `list_realsense_devices()`.
        """
        self.width = width
        self.height = height
        self.align_depth_to_color = align_depth_to_color
        self.fps = fps
        self.device_serial = device_serial

        self.pipeline: Optional[rs.pipeline] = None
        self.profile: Optional[rs.pipeline_profile] = None
        self._align: Optional[rs.align] = None

        # Threading
        self._thread: Optional[threading.Thread] = None
        self._running: bool = False
        self._lock = threading.Lock()
        self._latest_depth: Optional[np.ndarray] = None
        self._latest_color: Optional[np.ndarray] = None
        self._image_mask = self._normalize_mask(image_mask, (self.height, self.width))

    @classmethod
    def from_config(cls, cam_cfg: dict) -> "RealSenseCamera":
        intr = cam_cfg.get("intrinsics", {})
        width = intr.get("width", 1280)
        height = intr.get("height", 720)
        fps = intr.get("fps", 30)
        align = bool(cam_cfg.get("align_depth_to_color", False))
        serial = cam_cfg.get("serial") or cam_cfg.get("device_serial")
        mask = cls._load_mask_from_config(cam_cfg, (height, width))
        return cls(width=width, height=height, fps=fps,
                   align_depth_to_color=align, device_serial=serial, image_mask=mask)

    @staticmethod
    def _normalize_mask(mask: Optional[np.ndarray], shape_hw: Tuple[int, int]) -> Optional[np.ndarray]:
        if mask is None:
            return None
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if mask.shape != shape_hw:
            mask = cv2.resize(mask, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
        if mask.dtype != np.uint8:
            mask = np.clip(mask, 0, 255).astype(np.uint8)
        return np.where(mask > 127, 255, 0).astype(np.uint8)

    @classmethod
    def _load_mask_from_config(cls, cam_cfg: dict, shape_hw: Tuple[int, int]) -> Optional[np.ndarray]:
        mask_path = cam_cfg.get("image_mask_path")
        if not mask_path:
            return None

        mask_file = Path(mask_path)
        if not mask_file.is_absolute():
            mask_file = Path(__file__).resolve().parent / "config" / mask_file

        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Failed to load camera image mask: {mask_file}")
        return cls._normalize_mask(mask, shape_hw)

    def _apply_image_mask(self, depth: np.ndarray, color_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self._image_mask is None:
            return depth, color_rgb
        masked_depth = np.where(self._image_mask > 0, depth, 0).astype(depth.dtype, copy=False)
        masked_color = cv2.bitwise_and(color_rgb, color_rgb, mask=self._image_mask)
        return masked_depth, masked_color

    def start(self) -> None:
        if self.pipeline is not None:
            return  # already started

        self.pipeline = rs.pipeline()
        rs_cfg = rs.config()

        # Bind to specific device if requested
        if self.device_serial:
            print("Starting RealSense cam", self.device_serial)
            rs_cfg.enable_device(self.device_serial)
        else:
            print("Starting RealSense cam (auto)")

        rs_cfg.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        rs_cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)

        self.profile = self.pipeline.start(rs_cfg)
        self._align = rs.align(rs.stream.color) if self.align_depth_to_color else None

        # Start capture thread
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self):
        while self._running:
            try:
                frameset = self.pipeline.wait_for_frames()
                if self._align:
                    frameset = self._align.process(frameset)

                depth_frame = frameset.get_depth_frame()
                color_frame = frameset.get_color_frame()
                if depth_frame and color_frame:
                    depth = np.asanyarray(depth_frame.get_data())
                    color_bgr = np.asanyarray(color_frame.get_data())
                    color_rgb = color_bgr[:, :, ::-1].copy()

                    with self._lock:
                        self._latest_depth = depth
                        self._latest_color = color_rgb
            except Exception as e:
                print("Capture loop error:", e)
                time.sleep(0.01)

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        try:
            if self.pipeline is not None:
                self.pipeline.stop()
        except Exception:
            pass
        finally:
            self.pipeline = None
            self.profile = None
            self._align = None

    def warmup(self, n_frames: int = 60) -> None:
        print(f"Warming up {n_frames} frames...")
        for _ in range(n_frames):
            if not self._running:
                raise RuntimeError("Camera not started. Call start() first.")
            time.sleep(1.0 / self.fps)

    # def get_intrinsics_and_scale(self) -> Dict[str, float]:
    #     if not self.profile:
    #         raise RuntimeError("Camera not started. Call start() first.")
    #     s = self.profile.get_stream(rs.stream.color)
    #     intr = s.as_video_stream_profile().get_intrinsics()
    #     depth_sensor = self.profile.get_device().first_depth_sensor()
    #     return {
    #         "fx": intr.fx,
    #         "fy": intr.fy,
    #         "cx": intr.ppx,
    #         "cy": intr.ppy,
    #         "depth_scale": float(depth_sensor.get_depth_scale()),
    #     }
    def get_intrinsics_and_scale(self,intrinsics):
        if not self.profile:
            raise RuntimeError("Camera not started. Call start() first.")
        s = self.profile.get_stream(rs.stream.color)
        intr = s.as_video_stream_profile().get_intrinsics()
        cx, cy, fx, fy = intr.ppx, intr.ppy, intr.fx, intr.fy
        depth_sensor = self.profile.get_device().first_depth_sensor()
        scale = float(depth_sensor.get_depth_scale())
        intrinsics['fx'] = fx
        intrinsics['fy'] = fy
        intrinsics['cx'] = cx
        intrinsics['cy'] = cy        
        intrinsics['depth_scale'] = scale        
        return
 
    def get_rgbd(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (depth: HxW uint16, color_rgb: HxWx3 uint8) of the latest frame.
        """
        with self._lock:
            if self._latest_depth is None or self._latest_color is None:
                raise RuntimeError("No frames captured yet. Try warmup or wait a bit.")
            depth = self._latest_depth.copy()
            color = self._latest_color.copy()
        return self._apply_image_mask(depth, color)

    def get_rgb_file(self) -> str:
        _, color_rgb = self.get_rgbd()
        tmp_img_path = "/tmp/frame_llm.jpg"
        Image.fromarray(color_rgb).save(tmp_img_path, "JPEG", quality=95)
        return tmp_img_path

    def get_rgb_jpeg(self, quality: int = 90) -> bytes:
        """
        Returns color image as JPEG bytes.
        """
        _, color_rgb = self.get_rgbd()
        ok, buf = cv2.imencode(".jpg", color_rgb, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            raise RuntimeError("JPEG encode failed")
        return bytes(buf)


class RealSenseRig:
    """
    Helper to manage multiple cameras simultaneously.
    """
    def __init__(self, serials: List[str], width: int = 1280, height: int = 720, fps: int = 30,
                 align_depth_to_color: bool = True):
        self.cams: Dict[str, RealSenseCamera] = {
            s: RealSenseCamera(width=width, height=height, fps=fps,
                               align_depth_to_color=align_depth_to_color,
                               device_serial=s)
            for s in serials
        }

    @staticmethod
    def discover_serials() -> List[str]:
        return [d["serial"] for d in list_realsense_devices()]

    def start_all(self) -> None:
        for cam in self.cams.values():
            cam.start()

    def stop_all(self) -> None:
        for cam in self.cams.values():
            cam.stop()

    def get_rgbd(self, serial: str) -> Tuple[np.ndarray, np.ndarray]:
        return self.cams[serial].get_rgbd()

    def get_intrinsics_and_scale(self, serial: str) -> Dict[str, float]:
        return self.cams[serial].get_intrinsics_and_scale()
