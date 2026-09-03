import time
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Literal

import numpy as np
import commons.grasp_utils as utils
import cv2
import threading
# --------- Data logging ---------
@dataclass
class LoggedFrame:
    depth: np.ndarray      # HxW, uint16 or float32
    color: np.ndarray      # HxWx3, uint8
    T_base_cam: np.ndarray # 4x4


class Scanner:
    """
    Auto-centers a scanning dome on a table point estimated from the depth image center.

    Workflow:
      1) Grab an RGB-D frame while looking straight down.
      2) Back-project the *center pixel* (robust windowed median) to CAM frame, then to BASE:
         => this 3D point is the dome center on the table.
      3) Create poses so the CAMERA rides a small spherical dome (radius = your standoff),
         the camera always *looks at* the center, and the TCP **roll** stays locked.
      4) Execute a smooth path with blends and a decisive settle; log frames throughout.

    Notes:
      • The pole axis is BASE +Z (table assumed horizontal; +Z upward).
      • If you truly want the TCP (not the camera) on the sphere, set center_to='tcp'.
      • Requires RealSense intrinsics & depth scale via `cam.profile` (pyrealsense2).

    """

    def _apply_smooth_tilt_along_arch(self,
                                      arch: List[np.ndarray],
                                      arch_center: np.ndarray,
                                      R_tcp_ref: np.ndarray,
                                      tilt_max: float = 0.3) -> List[np.ndarray]:
        """
        Apply a gradually increasing tilt toward `arch_center` near the ends
        of the arch, with 0 tilt in the middle.

        This avoids a sharp orientation change at a single waypoint, so the
        robot doesn't stop and rotate at the extremes.
        """
        n = len(arch)
        if n <= 2:
            return arch

        new_arch: List[np.ndarray] = []

        for i, T in enumerate(arch):
            # normalized position along arch: 0 at start, 1 at end
            t = i / (n - 1)

            # distance to nearest end (0.5 in the middle, 1.0 at the ends)
            d_end = max(t, 1.0 - t)  # 0.5 -> middle, 1.0 -> ends

            # map [0.5, 1.0] -> [0.0, 1.0], clamp below 0.5 to 0 (no tilt in middle)
            s = (d_end - 0.5) / 0.5
            s = max(0.0, min(1.0, s))

            # smoothstep to make it soft: 3s^2 - 2s^3
            s_smooth = (3.0 * s * s) - (2.0 * s * s * s)

            if s_smooth <= 0.0:
                # no tilt in the central region
                new_arch.append(T)
            else:
                # gradually ramp up tilt towards `tilt_max` near the ends
                T_tilt = self._tilt_tcp_towards_point(
                    T, arch_center, R_tcp_ref, tilt_scale=tilt_max * s_smooth
                )
                new_arch.append(T_tilt)

        return new_arch

    @staticmethod
    def _tilt_tcp_towards_point(T_tcp: np.ndarray,
                                center: np.ndarray,
                                R_tcp_ref: np.ndarray,
                                tilt_scale: float = 0.3) -> np.ndarray:
        """
        Tilt the TCP slightly toward `center`.

        tilt_scale in [0, 1]:
          0.0 -> no tilt
          1.0 -> fully point Z at center
        """
        T_out = T_tcp.copy()
        R_cur = T_out[:3, :3]
        p = T_out[:3, 3]

        # Ideal Z-axis: from TCP position toward the center point
        k_full = center - p
        nk = np.linalg.norm(k_full)
        if nk < 1e-9:
            return T_out
        k_full = k_full / nk

        # Current TCP Z-axis
        k_cur = R_cur[:, 2]

        # Blend between current Z and ideal Z so tilt is smaller
        alpha = max(0.0, min(1.0, tilt_scale))
        k = (1.0 - alpha) * k_cur + alpha * k_full
        nk2 = np.linalg.norm(k)
        if nk2 < 1e-9:
            return T_out
        k = k / nk2

        # Project reference X into plane orthogonal to k to define roll
        x_ref = R_tcp_ref[:, 0]
        x_proj = x_ref - np.dot(x_ref, k) * k
        n = np.linalg.norm(x_proj)
        if n < 1e-9:
            # Fallback: try Y axis
            y_ref = R_tcp_ref[:, 1]
            x_proj = y_ref - np.dot(y_ref, k) * k
            n = np.linalg.norm(x_proj)
            if n < 1e-9:
                return T_out

        x_new = x_proj / n
        y_new = np.cross(k, x_new)
        y_new /= max(1e-9, np.linalg.norm(y_new))

        R_tilt = np.column_stack((x_new, y_new, k))
        T_out[:3, :3] = R_tilt
        return T_out

    def __init__(self,
                 arm: "URArm",
                 cam: "RealSenseCamera",
                 T_tcp_cam: np.ndarray,
                 debug: bool = True,
                ):
        self.debug = debug
        self.arm = arm
        self.cam = cam
        self.T_tcp_cam = T_tcp_cam
        self._R_cam_of_tcp = T_tcp_cam[:3, :3].copy()
        self._t_cam_of_tcp = T_tcp_cam[:3, 3].copy()
        self._t = None
        
    def _build_arch_waypoints_axis(self,
                                   T_base_tcp_start: np.ndarray,
                                   R_tcp_ref: np.ndarray,
                                   radius_m: float,
                                   angle_min_deg: float,
                                   angle_max_deg: float,
                                   n_points: int,
                                   axis: Literal["x", "y"] = "y",
                                   ) -> List[np.ndarray]:
        """
        Build TCP poses along a vertical circular arch in the given axis plane.

        The arch is a circle of radius `radius_m` in either:
            - X–Z plane (axis="x")
            - Y–Z plane (axis="y")

        The current TCP position is the *top* of the arch. The circle center is
        directly below the current pose along BASE +Z by `radius_m`.

        Parametrization (BASE frame):
            center = p0 - radius * ez
            p(theta) = center + radius * (sin(theta) * axis_dir + cos(theta) * ez)

        where:
            p0       = current TCP position
            ez       = BASE Z-axis (0,0,1)
            axis_dir = (1,0,0) for X-arch or (0,1,0) for Y-arch
        """
        p0 = T_base_tcp_start[:3, 3].copy()
        ez = np.array([0.0, 0.0, 1.0], dtype=float)

        if axis == "x":
            axis_dir = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            axis_dir = np.array([0.0, 1.0, 0.0], dtype=float)

        radius = max(1e-4, float(radius_m))
        center = p0 - radius * ez

        angle_min = float(angle_min_deg) * math.pi / 180.0
        angle_max = float(angle_max_deg) * math.pi / 180.0
        if n_points <= 1:
            thetas = [0.0]
        else:
            thetas = np.linspace(angle_min, angle_max, n_points)

        waypoints: List[np.ndarray] = []
        for th in thetas:
            s = math.sin(th)
            c = math.cos(th)

            # point on the circular arch
            offset = radius * (s * axis_dir + c * ez)
            p = center + offset

            T = T_base_tcp_start.copy()
            T[:3, 3] = p
            T = self._enforce_no_roll(T, R_tcp_ref)
            waypoints.append(T)

        return waypoints
    
    def scan_arch_xy(self,
                     radius_m: float = 0.10,
                     angle_min_deg: float = -45.0,
                     angle_max_deg: float = 45.0,
                     n_points: int = 25,
                     speed: float = 0.10,
                     accel: float = 1.00,
                     log_interval_s: float = 0.7
                     ) -> Tuple[List[LoggedFrame], np.ndarray]:
        """
        Perform an X-arch and a Y-arch scan around the *current* TCP pose.

        Each arch is a vertical circular arc (in X–Z or Y–Z) of radius `radius_m`,
        with the current TCP position at the top of the arch.

        Path sequence (conceptually):
            start
              → X-arch out-and-back
              → Y-arch out-and-back
              → back to start (handled by path builder)

        Args:
            radius_m:      Radius of the circular arch in meters.
            angle_min_deg: Minimum angle of the arch (deg), typically negative.
            angle_max_deg: Maximum angle of the arch (deg), typically positive.
            n_points:      Number of points per arch (per direction).
            speed:         Nominal path speed.
            accel:         Nominal path accel.
        """
        frames: List[LoggedFrame] = []

        # --- capture start state
        depth0, color0 = self.cam.get_rgbd()
        T_base_tcp_start = self.arm.get_T_base_tcp()
        T_base_cam_start = T_base_tcp_start @ self.T_tcp_cam
        frames.append(LoggedFrame(depth=depth0, color=color0, T_base_cam=T_base_cam_start))

        R_tcp_ref = T_base_tcp_start[:3, :3].copy()

        # --- build arches in TCP space
        n_points = max(3, int(n_points))
        angle_min = float(angle_min_deg)
        angle_max = float(angle_max_deg)
        if angle_max < angle_min:
            angle_min, angle_max = angle_max, angle_min  # swap if user inverted

        waypoints_tcp: List[np.ndarray] = []

        # Circle center used by both X/Y arches: directly below start along +Z
        ez = np.array([0.0, 0.0, 1.0], dtype=float)
        radius = max(1e-4, float(radius_m))
        arch_center = T_base_tcp_start[:3, 3] - radius * ez

        tilt_max = 0.3  # smaller tilt; tweak 0.2–0.4 as you like

        # X-arch (X-Z plane)
        # arch_x = self._build_arch_waypoints_axis(
        #     T_base_tcp_start=T_base_tcp_start,
        #     R_tcp_ref=R_tcp_ref,
        #     radius_m=radius_m*3,
        #     angle_min_deg=10,
        #     angle_max_deg=angle_max,
        #     n_points=n_points,
        #     axis="x",
        # )
        # # Smooth tilt near ends of X arch
        # arch_x = self._apply_smooth_tilt_along_arch(
        #     arch_x, arch_center, R_tcp_ref, tilt_max=tilt_max
        # )

        # waypoints_tcp.extend(arch_x)
        # return along the same arch (excluding endpoints to avoid duplicates)
        # if len(arch_x) > 2:
        #     waypoints_tcp.extend(reversed(arch_x[1:-1]))

        # Y-arch (Y-Z plane)
        arch_y = self._build_arch_waypoints_axis(
            T_base_tcp_start=T_base_tcp_start,
            R_tcp_ref=R_tcp_ref,
            radius_m=radius_m,
            angle_min_deg=angle_min,
            angle_max_deg=angle_max,
            n_points=n_points,
            axis="y",
        )
        # Smooth tilt near ends of Y arch
        arch_y = self._apply_smooth_tilt_along_arch(
            arch_y, arch_center, R_tcp_ref, tilt_max=tilt_max
        )

        waypoints_tcp.extend(arch_y)
        # arch_x = self._build_arch_waypoints_axis(
        #     T_base_tcp_start=arch_y[-1],
        #     R_tcp_ref=R_tcp_ref,
        #     radius_m=radius_m,
        #     angle_min_deg=0,
        #     angle_max_deg=angle_max,
        #     n_points=n_points,
        #     axis="x",
        # )
        # arch_x = self._apply_smooth_tilt_along_arch(
        #     arch_x, arch_center, R_tcp_ref, tilt_max=tilt_max
        # )
        # waypoints_tcp.extend(arch_x)


     
        # --- convert to controller path (fast, constant speed)
        all_wps = self._build_fast_path(waypoints_tcp, T_base_tcp_start, speed, accel)

        # --- execute + log frames
        self._t = threading.Thread(
            target=self._move_and_log,
            args=(all_wps, log_interval_s, frames,),
            name="movelog_arch_xy",
            daemon=True
        )
        self._t.start()
        self._t.join()

        return frames, T_base_tcp_start

    def scan_hemisphere(self,
             radius_m: Optional[float] = None,
             pitch_min_deg: float = -10.0,
             pitch_max_deg: float = 10.0,
             yaw_min_deg: float = -45.0,
             yaw_max_deg: float = 45.0,
             n_pitch: int = 3,
             n_yaw: int = 24,
             scan_centre_from_current_z_m: float = 0.2,
             center_to: Literal["camera", "tcp"] = "tcp",
             center_window_px: int = 15,
             speed: float = 0.10,
             accel: float = 0.50,
             log_interval_s: float = 0.75
             ) -> Tuple[List[LoggedFrame], np.ndarray]:
        """
        Execute a hemisphere scan around the *current* pose.

        Angles are defined relative to the current view direction:

            - pitch = 0°  : along current center→(cam/tcp) direction
            - yaw   = 0°  : an arbitrary orthogonal direction in that dome frame

        Args:
            radius_m:      Sphere radius. If None, use distance from center to current cam/tcp.
            pitch_min_deg: Min pitch offset from current direction (deg).
            pitch_max_deg: Max pitch offset from current direction (deg).
            yaw_min_deg:   Min yaw around the current direction (deg).
            yaw_max_deg:   Max yaw around the current direction (deg).
            n_pitch:       Samples between pitch_min_deg and pitch_max_deg.
            n_yaw:         Samples between yaw_min_deg and yaw_max_deg.
        """
        frames: List[LoggedFrame] = []

        # --- capture start state
        depth0, color0 = self.cam.get_rgbd()
        T_base_tcp_start = self.arm.get_T_base_tcp()
        R_tcp_ref = T_base_tcp_start[:3, :3].copy()  # roll reference
        T_base_cam_start = T_base_tcp_start @ self.T_tcp_cam
        frames.append(LoggedFrame(depth=depth0, color=color0, T_base_cam=T_base_cam_start))

        # --- define center point by dropping down in BASE Z from current camera origin
        center_base = T_base_cam_start[:3, 3].copy()
        center_base[2] -= scan_centre_from_current_z_m  # e.g. 0.2 m down

        # If you want depth-based center instead, re-enable this:
        # center_base = self._center_point_from_depth(depth0, T_base_cam_start, center_window_px)
        # if center_base is None:
        #     z_table = 0.0
        #     center_base = np.array([T_base_cam_start[0, 3],
        #                             T_base_cam_start[1, 3],
        #                             z_table], dtype=float)

        # --- radius: distance from center to current cam/tcp if not provided
        if radius_m is None:
            radius_m = self._auto_radius(center_base, T_base_tcp_start, T_base_cam_start, center_to)
            radius_m = float(max(1e-3, radius_m))

        # --- build waypoints on a dome **around the current pose direction**
        waypoints_tcp = self._plan_dome_waypoints(
            center=center_base,
            radius=radius_m,
            pitch_min_deg=pitch_min_deg,
            pitch_max_deg=pitch_max_deg,
            yaw_min_deg=yaw_min_deg,
            yaw_max_deg=yaw_max_deg,
            n_pitch=n_pitch,
            n_yaw=n_yaw,
            center_to=center_to,
            R_tcp_ref=R_tcp_ref,
            T_base_tcp_start=T_base_tcp_start,
            T_base_cam_start=T_base_cam_start,
        )

        if not waypoints_tcp:
            waypoints_tcp = self._fallback_ring(T_base_tcp_start, R_tcp_ref, max(0.03, 0.5 * radius_m))

        # --- convert to controller path with blends + settle
        all_wps = self._build_tapered_path(waypoints_tcp, T_base_tcp_start, speed, accel)
        print('waypoints', all_wps)

        # --- execute + log frames
        print('start log frame')
        self._t = threading.Thread(
            target=self._move_and_log,
            args=(all_wps, log_interval_s, frames,),
            name="movelog",
            daemon=True
        )
        self._t.start()
        self._t.join()

        return frames, T_base_tcp_start

    # ------------------ Public API ------------------
    def scan_line(self,
                  min_offset_m: float = -0.10,
                  max_offset_m: float = 0.10,
                  n_points: int = 25,
                  axis: Literal["x", "y"] = "y",
                  speed: float = 0.10,
                  accel: float = 1.00,
                  log_interval_s: float = 0.7
                  ) -> Tuple[List[LoggedFrame], np.ndarray]:
        """
        Simple linear scan around the *current* TCP pose.

        Path:
            start → (min_offset along axis) → … → (max_offset along axis) → back to start
        The return to start is handled by `_build_tapered_path`.

        Args:
            min_offset_m:  Signed minimum offset from current pose along chosen axis.
            max_offset_m:  Signed maximum offset from current pose along chosen axis.
                           Typically min < 0, max > 0 for left/right around start.
            n_points:      Number of points between min_offset and max_offset.
            axis:          'y' → move along BASE Y, 'x' → along BASE X.
            speed:         Nominal path speed.
            accel:         Nominal path accel.
        """
        frames: List[LoggedFrame] = []

        # --- capture start state
        depth0, color0 = self.cam.get_rgbd()
        T_base_tcp_start = self.arm.get_T_base_tcp()
        T_base_cam_start = T_base_tcp_start @ self.T_tcp_cam
        frames.append(LoggedFrame(depth=depth0, color=color0, T_base_cam=T_base_cam_start))

        R_tcp_ref = T_base_tcp_start[:3, :3].copy()
        p0 = T_base_tcp_start[:3, 3].copy()

        # --- sanity on offsets and point count
        n_points = max(2, int(n_points))
        min_offset = float(min_offset_m)
        max_offset = float(max_offset_m)

        # If user accidentally swaps them, fix it
        if max_offset < min_offset:
            min_offset, max_offset = max_offset, min_offset

        # --- choose axis in BASE frame
        if axis == "x":
            dir_vec = np.array([1.0, 0.0, 0.0], dtype=float)
        else:  # "y" by default
            dir_vec = np.array([0.0, 1.0, 0.0], dtype=float)

        # --- build straight-line waypoints: min → max (one sweep only)
        offsets = np.linspace(min_offset, max_offset, n_points)
        waypoints_tcp: List[np.ndarray] = []
        for d in offsets:
            T = T_base_tcp_start.copy()
            T[:3, 3] = p0 + d * dir_vec
            T = self._enforce_no_roll(T, R_tcp_ref)
            waypoints_tcp.append(T)

        # NOTE: no "there and back" duplication here.
        # `_build_tapered_path` will automatically bring you back to the start pose.

        # --- convert to controller path with blends + settle
        # all_wps = self._build_tapered_path(waypoints_tcp, T_base_tcp_start, speed, accel)
        all_wps = self._build_fast_path(waypoints_tcp, T_base_tcp_start, speed, accel)

        # --- execute + log frames
        self._t = threading.Thread(
            target=self._move_and_log,
            args=(all_wps, log_interval_s, frames,),
            name="movelog_line",
            daemon=True
        )
        self._t.start()
        self._t.join()

        return frames, T_base_tcp_start

    # ------------------ Depth → BASE center estimation ------------------
    def _center_point_from_depth(self,
                                 depth_img: np.ndarray,
                                 T_base_cam: np.ndarray,
                                 window_px: int) -> Optional[np.ndarray]:
        """
        Robustly back-project a center pixel neighborhood to get a 3D point in BASE.
        Uses median of valid depths in a square window around the image center.
        """
        try:
            import pyrealsense2 as rs  # type: ignore
        except ImportError:
            return None
        if getattr(self.cam, "profile", None) is None:
            return None

        H, W = depth_img.shape
        cx, cy = W // 2, H // 2
        w = max(4, int(window_px))
        x0, x1 = max(0, cx - w), min(W, cx + w + 1)
        y0, y1 = max(0, cy - w), min(H, cy + w + 1)
        patch = depth_img[y0:y1, x0:x1]

        # valid mask
        if patch.dtype == np.uint16:
            mask = patch > 0
        else:
            mask = patch > 1e-6

        if not np.any(mask):
            return None

        # robust depth (median of valid)
        d_vals = patch[mask].astype(np.float64)
        sensor = self.cam.profile.get_device().first_depth_sensor()
        depth_scale = float(sensor.get_depth_scale())
        depth_m = float(np.median(d_vals)) * depth_scale
        if depth_m <= 0.0 or not np.isfinite(depth_m):
            return None

        # Use the exact center pixel for the ray; median only for range
        intr = self.cam.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        Xc = (float(cx) - float(intr.ppx)) / float(intr.fx) * depth_m
        Yc = (float(cy) - float(intr.ppy)) / float(intr.fy) * depth_m
        Zc = depth_m

        p_cam = np.array([Xc, Yc, Zc, 1.0], dtype=float)
        p_base = (T_base_cam @ p_cam)[:3]
        return p_base

    # ------------------ Geometry helpers ------------------
    @staticmethod
    def _enforce_no_roll(T_tcp_desired: np.ndarray, R_tcp_ref: np.ndarray) -> np.ndarray:
        """
        Preserve position and TCP Z-axis (approach axis).
        Rotate around that Z so the TCP X-axis matches the reference roll.
        """
        T_out = T_tcp_desired.copy()
        R_des = T_out[:3, :3]

        k = R_des[:, 2]
        nk = np.linalg.norm(k)
        if nk < 1e-9:
            return T_out
        k = k / nk

        x_ref = R_tcp_ref[:, 0]
        x_proj = x_ref - np.dot(x_ref, k) * k
        n = np.linalg.norm(x_proj)
        if n < 1e-9:
            y_ref = R_tcp_ref[:, 1]
            x_proj = y_ref - np.dot(y_ref, k) * k
            n = np.linalg.norm(x_proj)
            if n < 1e-9:
                return T_out

        x_new = x_proj / n
        y_new = np.cross(k, x_new); y_new /= max(1e-9, np.linalg.norm(y_new))
        R_no_roll = np.column_stack((x_new, y_new, k))
        T_out[:3, :3] = R_no_roll
        return T_out

    @staticmethod
    def _look_at_T(camera_pos: np.ndarray, target: np.ndarray) -> Optional[np.ndarray]:
        """Return camera pose that looks at `target`, preferring +Z up."""
        forward = target - camera_pos
        n = np.linalg.norm(forward)
        if n < 1e-9:
            return None
        forward /= n

        up = np.array([0.0, 0.0, 1.0], dtype=float)
        if abs(float(np.dot(forward, up))) > 0.98:
            up = np.array([0.0, 1.0, 0.0], dtype=float)

        right = np.cross(up, forward); right /= max(1e-9, np.linalg.norm(right))
        down  = np.cross(forward, right); down  /= max(1e-9, np.linalg.norm(down))

        T = np.eye(4, dtype=float)
        T[:3, :3] = np.column_stack((right, down, forward))
        T[:3, 3]  = camera_pos
        return T

    def _plan_dome_waypoints(self,
                             center: np.ndarray,
                             radius: float,
                             pitch_min_deg: float,
                             pitch_max_deg: float,
                             yaw_min_deg: float,
                             yaw_max_deg: float,
                             n_pitch: int,
                             n_yaw: int,
                             center_to: str,
                             R_tcp_ref: np.ndarray,
                             T_base_tcp_start: np.ndarray,
                             T_base_cam_start: np.ndarray,
                             ) -> List[np.ndarray]:
        """
        Build TCP poses so CAMERA/TCP rides a spherical dome centered at `center`,
        oriented around the *current* center→(cam/tcp) direction.

        pitch, yaw are defined in that local dome frame:

            - pitch = 0° : along current direction (center→cam/tcp)
            - yaw   = 0° : arbitrary, but fixed, direction orthogonal to that.
        """
        n_yaw = max(4, int(n_yaw))
        n_pitch = max(1, int(n_pitch))

        # --- choose which point defines the pole (camera or tcp)
        if center_to == "camera":
            start_pos = T_base_cam_start[:3, 3]
        else:  # "tcp"
            start_pos = T_base_tcp_start[:3, 3]

        pole = start_pos - center
        norm_pole = np.linalg.norm(pole)
        if norm_pole < 1e-9:
            # Fallback: arbitrary pole
            pole = np.array([0.0, 0.0, 1.0], dtype=float)
        else:
            pole = pole / norm_pole

        # --- build an orthonormal basis: {pole, t0, t1}
        world_up = np.array([0.0, 0.0, 1.0], dtype=float)
        if abs(float(np.dot(world_up, pole))) > 0.99:
            world_up = np.array([0.0, 1.0, 0.0], dtype=float)

        t0 = world_up - np.dot(world_up, pole) * pole
        t0_norm = np.linalg.norm(t0)
        if t0_norm < 1e-9:
            t0 = np.array([1.0, 0.0, 0.0], dtype=float)
            t0_norm = 1.0
        t0 /= t0_norm

        t1 = np.cross(pole, t0)
        t1 /= max(1e-9, np.linalg.norm(t1))

        # --- angle grids
        pitch_min = float(pitch_min_deg) * math.pi / 180.0
        pitch_max = float(pitch_max_deg) * math.pi / 180.0
        if n_pitch == 1:
            pitch_levels = [0.5 * (pitch_min + pitch_max)]
        else:
            pitch_levels = list(np.linspace(pitch_min, pitch_max, n_pitch))

        yaw_min = float(yaw_min_deg) * math.pi / 180.0
        yaw_max = float(yaw_max_deg) * math.pi / 180.0
        if yaw_max < yaw_min:
            yaw_min, yaw_max = yaw_max, yaw_min  # swap

        cam_to_tcp = np.linalg.inv(self.T_tcp_cam)
        waypoints: List[np.ndarray] = []

        for i_p, pitch in enumerate(pitch_levels):

            # boustrophedon in yaw for nicer paths
            if i_p % 2 == 0:
                yaw_vals = np.linspace(yaw_min, yaw_max, n_yaw, endpoint=True)
            else:
                yaw_vals = np.linspace(yaw_max, yaw_min, n_yaw, endpoint=True)

            for yaw in yaw_vals:
                cp, sp = math.cos(pitch), math.sin(pitch)
                cy, sy = math.cos(yaw), math.sin(yaw)

                # direction on sphere
                dir_vec = cp * pole + sp * (cy * t0 + sy * t1)
                dir_vec /= max(1e-9, np.linalg.norm(dir_vec))

                if center_to == "camera":
                    cam_pos = center + radius * dir_vec
                    T_cam = self._look_at_T(cam_pos, center)
                    if T_cam is None:
                        continue
                    T_tcp = self._enforce_no_roll(T_cam @ cam_to_tcp, R_tcp_ref)

                else:  # "tcp"
                    tcp_pos = center + radius * dir_vec
                    # approximate orientation from "camera looking at center" then back to tcp
                    T_cam_approx = self._look_at_T(tcp_pos, center)
                    if T_cam_approx is None:
                        continue
                    R_cam_des = T_cam_approx[:3, :3]
                    R_tcp = R_cam_des @ np.linalg.inv(self._R_cam_of_tcp)
                    cam_pos = tcp_pos + R_tcp @ self._t_cam_of_tcp
                    T_cam_true = self._look_at_T(cam_pos, center)
                    if T_cam_true is None:
                        continue
                    T_tcp = T_cam_true @ cam_to_tcp
                    T_tcp[:3, 3] = tcp_pos
                    T_tcp = self._enforce_no_roll(T_tcp, R_tcp_ref)

                waypoints.append(T_tcp)

        return waypoints
    
    def _auto_radius(self,
                    center_base: np.ndarray,
                    T_base_tcp_start: np.ndarray,
                    T_base_cam_start: np.ndarray,
                    center_to: str) -> float:
        """Compute standoff radius from the current start pose."""
        if center_to == "camera":
            p = T_base_cam_start[:3, 3]
        else:  # "tcp"
            p = T_base_tcp_start[:3, 3]
        return float(np.linalg.norm(p - center_base))

    @staticmethod
    def _fallback_ring(T_base_tcp_start: np.ndarray,
                       R_tcp_ref: np.ndarray,
                       radius_m: float) -> List[np.ndarray]:
        """Small planar ring around start if dome cannot be built."""
        n_ring = 16
        angles = np.linspace(0.0, 2.0 * math.pi, num=n_ring, endpoint=False)
        p0 = T_base_tcp_start[:3, 3].copy()
        waypoints = []
        for th in angles:
            T = T_base_tcp_start.copy()
            T[:3, 3] = p0 + radius_m * np.array([math.cos(th), math.sin(th), 0.0])
            T = Scanner._enforce_no_roll(T, R_tcp_ref)
            waypoints.append(T)
        return waypoints
    def _build_fast_path(self,
                         waypoints_tcp: List[np.ndarray],
                         T_base_tcp_start: np.ndarray,
                         speed: float,
                         accel: float,
                         blend: float = 0.003) -> List[List[float]]:
        """
        Simple, constant-speed path:
          start → small +Z lead → all waypoints → small +Z pre-brake → exact start

        No easing, no speed modulation – uses the requested speed/accel everywhere.
        """
        all_waypoints: List[List[float]] = []

        # --- lead-in from start (optional small lift)
        lead_T = T_base_tcp_start.copy()
        lead_T[:3, 3] += np.array([0.0, 0.0, 0.003])  # 3 mm lift
        p_lead = utils.pose_from_T(lead_T)
        all_waypoints.append([*p_lead, speed, accel, blend])

        # --- main path (constant speed/accel)
        for T in waypoints_tcp:
            p = utils.pose_from_T(T)
            all_waypoints.append([*p, speed, accel, blend])

        # --- pre-brake + final settle at the exact start pose
        # pre_T = T_base_tcp_start.copy()
        # pre_T[:3, 3] += np.array([0.0, 0.0, 0.003])
        # p_pre   = utils.pose_from_T(pre_T)
        # p_final = utils.pose_from_T(T_base_tcp_start)

        # all_waypoints.append([*p_pre,   speed, accel, blend])
        # all_waypoints.append([*p_final, speed, accel, 0.0])

        return all_waypoints

    def _build_tapered_path(self,
                            waypoints_tcp: List[np.ndarray],
                            T_base_tcp_start: np.ndarray,
                            speed: float,
                            accel: float) -> List[List[float]]:
        """Convert TCP T's to moveLPath points with good blends + sharp settle, with a soft start."""
        all_waypoints: List[List[float]] = []

        # --- soft lead-in from the exact start pose (short +Z lift)
        lead_T = T_base_tcp_start.copy()
        lead_T[:3, 3] += np.array([0.0, 0.0, 0.006])  # 6 mm lift
        p_lead = utils.pose_from_T(lead_T)
        # very slow + tiny blend to avoid a jerk off the line
        all_waypoints.append([*p_lead, 0.03, 0.20, 0.001])

        # --- main path with an ease-in/ease-out speed profile
        N = len(waypoints_tcp)
        for i, T in enumerate(waypoints_tcp):
            p = utils.pose_from_T(T)
            # normalized progress along path
            t = i / max(1, N - 1)

            # cubic ease-in/ease-out (smoothstep-like) for speed+accel scaling
            # bias the first ~20% to be noticeably slower
            def smoothstep(x):  # 3x^2 - 2x^3
                x = max(0.0, min(1.0, x))
                return (3.0 * x * x) - (2.0 * x * x * x)

            # map t into two regions: slow start (0..0.2) then ramp to cruise
            if t <= 0.3:
                # s = smoothstep(t / 0.2) * 0.5  # 0 -> 0.
                s = 0.5 + 0.25 * smoothstep((t - 0.2) / 0.8)  # 0.5 -
            else:
                s = 0.5 + 0.5 * smoothstep((t - 0.2) / 0.8)  # 0.5 -> 1.0

            v_i = max(0.02, (0.15 + 0.85 * s) * speed)   # ~0.03–1.0×speed
            a_i = max(0.10, (0.20 + 0.80 * s) * accel)   # ~0.2–1.0×accel

            # larger blends early, taper to your nominal by mid-path
            # bend_i = (1.0 - s) * 0.020 + s * 0.008       # 0.020 → 0.008
            bend_i = 0.003
            all_waypoints.append([*p, v_i, a_i, bend_i])
            # all_waypoints.append([*p, speed, accel, 0.01])

        # --- pre-brake & final settle near the starting pose (short +Z lift)
        final_T = T_base_tcp_start.copy()
        pre_T   = final_T.copy()
        pre_T[:3, 3] += np.array([0.0, 0.0, 0.004])  # 4 mm lift

        p_pre   = utils.pose_from_T(pre_T)
        p_final = utils.pose_from_T(final_T)

        all_waypoints.append([*p_pre,   0.06, 1.00, 0.001])
        all_waypoints.append([*p_final, 0.03, 1.20, 0.0])
        return all_waypoints

    def _move_and_log(self,
                      all_waypoints: List[List[float]],
                      log_interval_s: float,
                      frames_out: List[LoggedFrame]) -> None:

        self._record_frame(frames_out)
        """Execute the path and log frames while motion is in-flight."""
        self.arm.moveLPath(all_waypoints, async_=True)

        # Periodic logs while controller is active
        while True:
            time.sleep(max(0.02, log_interval_s))
            self._record_frame(frames_out)
            if not self.arm.getIsMoving():
                print('finished')
                break

        # Final log
        print('record completed')
        self._record_frame(frames_out)

    def _draw_overlay(self, color_img: np.ndarray, T_base_tcp: np.ndarray, T_base_cam: np.ndarray) -> np.ndarray:
        """
        Draw a lightweight HUD: TCP pose, camera z-height, and a center reticle.
        """
        vis = color_img.copy()
        h, w = vis.shape[:2]

        # Text lines
        line1, _ = self._format_pose(T_base_tcp)
        cam_z_mm = 1000.0 * float(T_base_cam[2, 3])
        line2 = f"CAM: Z={cam_z_mm:+.1f} mm   (Press 'q' to close windows)"

        # Background plate for readability
        pad = 6
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 1
        (tw1, th1), _ = cv2.getTextSize(line1, font, scale, thickness)
        (tw2, th2), _ = cv2.getTextSize(line2, font, scale, thickness)
        box_w = int(max(tw1, tw2) + 2 * pad)
        box_h = int(th1 + th2 + 3 * pad)
        cv2.rectangle(vis, (10, 10), (10 + box_w, 10 + box_h), (0, 0, 0), -1)
        cv2.addWeighted(vis[10:10+box_h, 10:10+box_w], 0.6,
                        np.zeros((box_h, box_w, 3), dtype=vis.dtype), 0.4, 0, vis[10:10+box_h, 10:10+box_w])

        # Draw text
        y0 = 10 + pad + th1
        cv2.putText(vis, line1, (10 + pad, y0), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(vis, line2, (10 + pad, y0 + th2 + pad), font, scale, (200, 200, 200), thickness, cv2.LINE_AA)

        # Center reticle
        cx, cy = w // 2, h // 2
        cv2.drawMarker(vis, (cx, cy), (0, 255, 0), markerType=cv2.MARKER_TARGET, markerSize=16, thickness=1)
        cv2.circle(vis, (cx, cy), 22, (0, 255, 0), 1, cv2.LINE_AA)

        # Optional downscale for speed
        if self._window_scale != 1.0:
            vis = cv2.resize(vis, None, fx=self._window_scale, fy=self._window_scale, interpolation=cv2.INTER_AREA)
        return vis
        
    def _record_frame(self, frames: List[LoggedFrame]) -> LoggedFrame:
        try:
            depth_img, color_img = self.cam.get_rgbd()           
        except RuntimeError as e:
            print('failed to get rgbd',e)
            return None
            
        T_now_tcp = self.arm.get_T_base_tcp()
        T_now_cam = T_now_tcp @ self.T_tcp_cam
        frame = LoggedFrame(depth=depth_img, color=color_img, T_base_cam=T_now_cam)
        frames.append(frame)
        return frame
