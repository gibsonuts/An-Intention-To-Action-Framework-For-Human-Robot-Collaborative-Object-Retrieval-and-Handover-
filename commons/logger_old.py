import time
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Literal

import numpy as np
import commons.grasp_utils as utils


# --------- Data logging ---------
@dataclass
class LoggedFrame:
    depth: np.ndarray      # HxW, uint16 or float32
    color: np.ndarray      # HxWx3, uint8
    T_base_cam: np.ndarray # 4x4


class HemisphereScanner:
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

    def __init__(self,
                 arm: "URArm",
                 cam: "RealSenseCamera",
                 T_tcp_cam: np.ndarray):
        self.arm = arm
        self.cam = cam
        self.T_tcp_cam = T_tcp_cam
        self._R_cam_of_tcp = T_tcp_cam[:3, :3].copy()
        self._t_cam_of_tcp = T_tcp_cam[:3, 3].copy()

    # ------------------ Public API ------------------
    def scan(self,
             radius_m: Optional[float] = None,
             tilt_max_deg: float = 5.0,   # ± around top view
             n_tilt: int = 3,             # e.g., [0°, mid, max]
             n_az: int = 24,              # azimuth samples per ring
             center_to: Literal["camera", "tcp"] = "tcp",
             center_window_px: int = 15,  # robust depth window around image center
             speed: float = 0.10,
             accel: float = 0.50,
             log_interval_s: float = 0.75) -> Tuple[List[LoggedFrame], np.ndarray]:
        """
        Execute the dome scan with the center auto-estimated from the current depth frame.

        Returns:
            (logged_frames, T_base_tcp_start)
        """
        frames: List[LoggedFrame] = []

        # --- capture start state
        depth0, color0 = self.cam.get_rgbd()
        T_base_tcp_start = self.arm.get_T_base_tcp()
        R_tcp_ref = T_base_tcp_start[:3, :3].copy()  # roll reference
        T_base_cam_start = T_base_tcp_start @ self.T_tcp_cam
        frames.append(LoggedFrame(depth=depth0, color=color0, T_base_cam=T_base_cam_start))

        # --- estimate table center directly under the optical axis
        center_base = self._center_point_from_depth(depth0, T_base_cam_start, center_window_px)
        if center_base is None:
            # Fallback: use vertical drop from camera origin onto a nominal table z
            # (You can replace z_table with your known table height.)
            z_table = 0.0
            center_base = np.array([T_base_cam_start[0, 3],
                                    T_base_cam_start[1, 3],
                                    z_table], dtype=float)
        if radius_m is None:
            radius_m = self._auto_radius(center_base, T_base_tcp_start, T_base_cam_start, center_to)
            # optional: enforce a minimum to avoid grazing the table
            radius_m = float(max(1e-3, radius_m))

        # --- build waypoints on a spherical cap around top view
        waypoints_tcp = self._plan_dome_waypoints(
            center=center_base,
            radius=radius_m,
            tilt_max_deg=tilt_max_deg,
            n_tilt=n_tilt,
            n_az=n_az,
            center_to=center_to,
            R_tcp_ref=R_tcp_ref,
        )

        if not waypoints_tcp:
            waypoints_tcp = self._fallback_ring(T_base_tcp_start, R_tcp_ref, max(0.03, 0.5 * radius_m))

        # --- convert to controller path with good blends + sharp settle
        all_wps = self._build_tapered_path(waypoints_tcp, T_base_tcp_start, speed, accel)

        # --- execute + log frames
        self._move_and_log(all_wps, log_interval_s, frames)

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
                             tilt_max_deg: float,
                             n_tilt: int,
                             n_az: int,
                             center_to: str,
                             R_tcp_ref: np.ndarray
                             ) -> List[np.ndarray]:
        """
        Build TCP poses so CAMERA (default) sits on a spherical cap centered at `center`.
        Top view = along BASE +Z from center; tilt increases toward equator.
        """
        n_az = max(4, int(n_az))
        n_tilt = max(1, int(n_tilt))

        pole = np.array([0.0, 0.0, 1.0], dtype=float)  # up from center
        east = np.array([1.0, 0.0, 0.0], dtype=float)
        north = np.cross(pole, east); north /= max(1e-9, np.linalg.norm(north))
        east = np.cross(north, pole); east /= max(1e-9, np.linalg.norm(east))

        tilt_max = abs(float(tilt_max_deg)) * math.pi / 180.0
        tilt_levels = [0.0] if n_tilt == 1 else list(np.linspace(0.0, tilt_max, n_tilt))
    
        cam_to_tcp = np.linalg.inv(self.T_tcp_cam)
        waypoints: List[np.ndarray] = []

        for i_t, tilt in enumerate(tilt_levels):

            # # --- normal rings (one pass each, boustrophedon)
            az_vals = (np.linspace(0.0, 2.0 * math.pi, n_az, endpoint=False)
                    if i_t % 2 == 0
                    else np.linspace(2.0 * math.pi, 0.0, n_az, endpoint=False))

            for az in az_vals:
                ca, sa = math.cos(az), math.sin(az)
                horiz = east * ca + north * sa
                dir_vec = (math.cos(tilt) * pole) + (math.sin(tilt) * horiz)
                dir_vec /= max(1e-9, np.linalg.norm(dir_vec))
                if center_to == "camera":
                    cam_pos = center + radius * dir_vec
                    T_cam = self._look_at_T(cam_pos, center)
                    if T_cam is None:
                        continue
                    T_tcp = self._enforce_no_roll(T_cam @ cam_to_tcp, R_tcp_ref)
                else:
                    tcp_pos = center + radius * dir_vec
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

        # Ensure first pose is exactly the top view
        # if waypoints:
        #     cam_top = center + radius * pole
        #     T_cam_top = self._look_at_T(cam_top, center)
        #     if T_cam_top is not None:
        #         T_tcp_top = T_cam_top @ cam_to_tcp
        #         T_tcp_top = self._enforce_no_roll(T_tcp_top, R_tcp_ref)
        #         waypoints.insert(0, T_tcp_top)

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
            T = HemisphereScanner._enforce_no_roll(T, R_tcp_ref)
            waypoints.append(T)
        return waypoints

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
                s = smoothstep(t / 0.2) * 0.5  # 0 -> 0.5
            else:
                s = 0.5 + 0.5 * smoothstep((t - 0.2) / 0.8)  # 0.5 -> 1.0

            v_i = max(0.02, (0.15 + 0.85 * s) * speed)   # ~0.03–1.0×speed
            a_i = max(0.10, (0.20 + 0.80 * s) * accel)   # ~0.2–1.0×accel

            # larger blends early, taper to your nominal by mid-path
            bend_i = (1.0 - s) * 0.020 + s * 0.008       # 0.020 → 0.008

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
        """Execute the path and log frames while motion is in-flight."""
        self.arm.moveLPath(all_waypoints, async_=True)

        # Initial log
        self._record_frame(frames_out)

        # Periodic logs while controller is active
        while True:
            time.sleep(max(0.02, log_interval_s))
            self._record_frame(frames_out)
            if not self.arm.running():
                break

        # Final log
        self._record_frame(frames_out)

    def _record_frame(self, frames: List[LoggedFrame]) -> LoggedFrame:
        depth_img, color_img = self.cam.get_rgbd()
        T_now_tcp = self.arm.get_T_base_tcp()
        T_now_cam = T_now_tcp @ self.T_tcp_cam
        frame = LoggedFrame(depth=depth_img, color=color_img, T_base_cam=T_now_cam)
        frames.append(frame)
        return frame
