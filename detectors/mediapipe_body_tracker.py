import cv2
import copy
import time
import threading
import queue
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import mediapipe as mp

# ========== MediaPipe shorthands ==========
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands


# ========== Data types ==========
@dataclass
class Landmark3D:
    idx: int
    X: float  # meters, camera coordinates (+X right, +Y down, +Z forward)
    Y: float
    Z: float
    vis: float = 1.0


# ========== Tracker ==========
class RealSenseMediapipeTracker:
    """
    Streams RGBD frames from a *pre-initialized* camera (e.g., hw.cam_arm),
    runs MediaPipe Pose + Hands, and exposes thread-safe getters for the latest
    3D body and hand landmarks.

    IMPORTANT:
      - No pyrealsense2 pipeline here.
      - Pass in a camera-like object with .get_rgbd() -> (depth, color_rgb or bgr).
      - Intrinsics (fx, fy, cx, cy) + depth scale are taken from cam_cfg['intrinsics'].
      - GUI is handled ONLY on the main thread via DisplayHub; workers just push frames.

    Coordinate frames:
      - Input camera frame assumes +X right, +Y down, +Z forward (RealSense-like).
      - Map to another frame by passing a 4x4 transform into the getters.
    """

    def __init__(
        self,
        camera_obj: Any,
        cam_cfg: Any,
        # MediaPipe options
        enable_pose: bool = True,
        enable_hands: bool = True,
        max_num_hands: int = 2,
        min_det_conf: float = 0.5,
        min_trk_conf: float = 0.5,
        # Visualization
        debug: bool = False,  # if True, frames are published to a queue for DisplayHub
        preview_window_name: str = "Camera + MediaPipe",
        # If your camera returns RGB images already, set this True.
        # If it returns BGR, set False so we convert correctly for MediaPipe.
        color_is_rgb: bool = True,
        # ---- Filtering config ----
        min_visible_kps: int = 15,
        min_shoulder_width_m: float = 0.18,
        max_shoulder_width_m: float = 0.80,
        min_torso_len_m: float = 0.25,
        max_torso_len_m: float = 1.10,
        min_body_depth_m: float = 0.30,
        max_body_depth_m: float = 6.00,
        keep_last_good_secs: float = 0.25,
        # ---- Gaze / facing thresholds ----
        max_face_yaw_deg: float = 20.0,
        max_face_pitch_deg: float = 60.0,
        min_nose_ahead_m: float = 0.01,  # nose must be at least 2cm closer than head center
        # ---- Face-only mode ----
        face_only: bool = False,
    ):
        # --- Camera wiring ---
        if camera_obj is None:
            raise ValueError("Provide camera_obj and a config with valid intrinsics.")
        if cam_cfg is None or "intrinsics" not in cam_cfg:
            raise ValueError("cam_cfg must include an 'intrinsics' dict.")
        self.cam = camera_obj
        self._color_intrinsics: Optional[Dict[str, float]] = None
        self._depth_scale_m: float = 0.001  # fallback default
        self.color_is_rgb = color_is_rgb
        self._set_intrinsics_from_dict(cam_cfg["intrinsics"])

        # --- Modes & options ---
        self.face_only = bool(face_only)
        # If face_only and hands not explicitly True, disable hands for speed
        if self.face_only and enable_hands is True:
            pass
        elif self.face_only:
            enable_hands = False

        # --- MediaPipe flags (actual objects are created in the worker thread) ---
        self.enable_pose = enable_pose
        self.enable_hands = enable_hands
        self.max_num_hands = max_num_hands
        self.min_det_conf = min_det_conf
        self.min_trk_conf = min_trk_conf

        # --- Visualization (no imshow in worker threads!) ---
        serial = getattr(camera_obj, "device_serial", "unknown")
        self.preview_window_name = f'[{serial}] {preview_window_name}'
        self.debug = bool(debug)  # if True, publish preview frames into _preview_queue
        self._preview_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=1)

        # MediaPipe objects (owned by worker thread). We keep refs for getters but
        # create/close them inside the worker thread.
        self.pose = None
        self.hands = None
        self._pose_model_complexity = 0 if self.face_only else 1

        # Threading / state
        self._t: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest_pose3d: Dict[int, Landmark3D] = {}
        self._latest_hands3d: List[Dict[int, Landmark3D]] = []
        self._last_update_s: float = 0.0

        # --- Filtering config ---
        self.min_visible_kps = int(min_visible_kps)
        self.min_shoulder_width_m = float(min_shoulder_width_m)
        self.max_shoulder_width_m = float(max_shoulder_width_m)
        self.min_torso_len_m = float(min_torso_len_m)
        self.max_torso_len_m = float(max_torso_len_m)
        self.min_body_depth_m = float(min_body_depth_m)
        self.max_body_depth_m = float(max_body_depth_m)
        self.keep_last_good_secs = float(keep_last_good_secs)

        # Gaze thresholds
        self.max_face_yaw_deg = float(max_face_yaw_deg)
        self.max_face_pitch_deg = float(max_face_pitch_deg)
        self.min_nose_ahead_m = float(min_nose_ahead_m)

        # Last good pose cache
        self._last_good_pose3d: Dict[int, Landmark3D] = {}
        self._last_good_ts: float = 0.0

        self._depth_template = None
        self._depth_template_valid = None
        
        # FPS / gaze helper
        self._last_frame_ts: Optional[float] = None
        self._last_gaze: Optional[Tuple[Optional[bool], Dict[str, float]]] = None  # (is_looking, metrics)

    def _depth_image_to_meters(self, depth_image: np.ndarray) -> np.ndarray:
        """
        Convert a full depth image to meters using self._depth_scale_m.

        - If depth_image is floating-point, it is assumed to already be in meters.
        - If integer, it is scaled by depth_scale_m.
        """
        if depth_image is None:
            raise ValueError("depth_image is None")

        depth = depth_image.astype(np.float32)
        if not np.issubdtype(depth_image.dtype, np.floating):
            depth *= self._depth_scale_m
        return depth
    
    # --------- Depth template helpers ---------
    def build_depth_template(
        self,
        num_frames: int = 30,
        sleep_s: float = 0.02,
    ) -> Optional[np.ndarray]:
        """
        Build a static depth template from multiple frames.

        Args:
            num_frames: number of frames to accumulate (more = smoother, slower).
            use_camera_direct: if True, call self.cam.get_rgbd() directly.
                               False is safer if the camera object isn't thread-safe.
            sleep_s: small delay between samples.

        Returns:
            The depth template in meters (H x W, float32), or None if no frames.
        """
        frames_m = []

        for _ in range(num_frames):
            depth, _ = self.cam.get_rgbd()
   
            if depth is None:
                time.sleep(sleep_s)
                continue

            depth_m = self._depth_image_to_meters(depth)
            frames_m.append(depth_m)
            time.sleep(sleep_s)

        if not frames_m:
            print("[Tracker] build_depth_template: no depth frames collected.")
            return None

        stack = np.stack(frames_m, axis=0)  # (N, H, W)
        template = np.median(stack, axis=0).astype(np.float32)

        with self._lock:
            self._depth_template = template
            self._depth_template_valid = template > 0.0

        return template

    def set_depth_template(self, template_m: np.ndarray) -> None:
        """
        Manually set the depth template (meters).
        """
        if template_m is None:
            raise ValueError("template_m is None")
        if not isinstance(template_m, np.ndarray):
            raise TypeError("template_m must be a numpy array")

        template = template_m.astype(np.float32)
        with self._lock:
            self._depth_template = template
            self._depth_template_valid = template > 0.0

    def get_depth_template(self) -> Optional[np.ndarray]:
        """
        Get a copy of the current depth template (meters), or None.
        """
        with self._lock:
            if self._depth_template is None:
                return None
            return self._depth_template.copy()
        
    def get_depth_debug_images(
        self,
        min_diff_m: float = 0.05,
        roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Returns (template_vis_bgr, delta_vis_bgr) for debugging.

        - template_vis_bgr: colormap of the template depth.
        - delta_vis_bgr: colormap of (template - current_depth) where positive
                         means 'closer than template', with foreground overlay.
        """

        # Grab template safely
        with self._lock:
            template = None if self._depth_template is None else self._depth_template.copy()
            valid_mask = None if self._depth_template_valid is None else self._depth_template_valid.copy()

        if template is None or valid_mask is None:
            return None, None

        depth_raw, _ = self.cam.get_rgbd()
        if depth_raw is None:
            return None, None

        depth_m = self._depth_image_to_meters(depth_raw)

        # Align ROI if requested
        if roi is not None:
            x, y, w, h = roi
            x2, y2 = x + w, y + h
            template = template[y:y2, x:x2]
            depth_m = depth_m[y:y2, x:x2]
            valid_mask = valid_mask[y:y2, x:x2]

        valid = (template > 0.0) & (depth_m > 0.0) & valid_mask
        if not np.any(valid):
            return None, None

        # --- Template visualization ---
        tmpl = template.copy()
        tmpl[~valid] = 0.0
        t_min, t_max = np.percentile(tmpl[valid], [5, 95])
        if t_max <= t_min:
            t_max = t_min + 1e-3
        tmpl_norm = np.clip((tmpl - t_min) / (t_max - t_min), 0.0, 1.0)
        tmpl_u8 = (tmpl_norm * 255).astype(np.uint8)
        tmpl_color = cv2.applyColorMap(tmpl_u8, cv2.COLORMAP_JET)

        # --- Delta visualization (template - depth) ---
        delta = template - depth_m
        delta[~valid] = 0.0
        delta_pos = np.clip(delta, 0.0, 0.3)  # clamp to 30cm for visual clarity
        d_min, d_max = 0.0, 0.3
        delta_norm = np.clip((delta_pos - d_min) / (d_max - d_min), 0.0, 1.0)
        delta_u8 = (delta_norm * 255).astype(np.uint8)
        delta_color = cv2.applyColorMap(delta_u8, cv2.COLORMAP_HOT)

        # Highlight strong foreground (e.g. > min_diff_m) in bright color
        fg = (delta > min_diff_m) & valid
        delta_color[fg] = (0, 255, 0)  # bright green means "intrusion"

        return tmpl_color, delta_color

    def check_depth_intrusion(
        self,
        max_depth_m: float = 0.2,
        min_diff_m: float = 0.001,
        min_area_ratio: float = 0.,
        roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Compare the current depth image to the depth template to detect intrusions.

        max_depth_m:
            Ignore any pixels where either template or current depth is farther
            than this distance (in meters).
        """

        # Grab template safely
        with self._lock:
            template = None if self._depth_template is None else self._depth_template.copy()
            valid_mask = None if self._depth_template_valid is None else self._depth_template_valid.copy()

        if template is None or valid_mask is None:
            return False, {
                "area_ratio": 0.0,
                "mean_delta_m": 0.0,
                "n_fg_pixels": 0,
                "n_valid_pixels": 0,
            }

        # Depth frame
        depth_raw, _ = self.cam.get_rgbd()
        if depth_raw is None:
            return False, {
                "area_ratio": 0.0,
                "mean_delta_m": 0.0,
                "n_fg_pixels": 0,
                "n_valid_pixels": 0,
            }

        depth_m = self._depth_image_to_meters(depth_raw)

        # Align to ROI if requested
        if roi is not None:
            x, y, w, h = roi
            x2, y2 = x + w, y + h
            template = template[y:y2, x:x2]
            depth_m = depth_m[y:y2, x:x2]
            valid_mask = valid_mask[y:y2, x:x2]

        # --- Build per-image masks first ---
        # Valid template pixels: >0, within depth range, and marked valid
        if max_depth_m is not None:
            mask_tmpl = (template > 0.0) & (template <= max_depth_m) & valid_mask
            mask_curr = (depth_m > 0.0) & (depth_m <= max_depth_m) & valid_mask
        else:
            mask_tmpl = (template > 0.0) & valid_mask
            mask_curr = (depth_m > 0.0) & valid_mask

        # Intersection: pixels valid in *both* template and current depth
        valid = mask_tmpl & mask_curr

        n_valid = int(np.count_nonzero(valid))
        if n_valid == 0:
            # No overlapping valid pixels within range → nothing to compute
            return False, {
                "area_ratio": 0.0,
                "mean_delta_m": 0.0,
                "n_fg_pixels": 0,
                "n_valid_pixels": 0,
            }

        # Flattened valid depths for stats if you want them
        tmpl_valid = template[valid]
        curr_valid = depth_m[valid]

        # Positive when something is closer than the template
        delta = tmpl_valid - curr_valid
        fg = delta > min_diff_m

        n_fg = int(np.count_nonzero(fg))
        area_ratio = float(n_fg) / float(n_valid) if n_valid > 0 else 0.0
        mean_delta = float(delta[fg].mean()) if n_fg > 0 else 0.0

        intrusion = (n_fg > 0) and (area_ratio >= min_area_ratio)

        # Optional debug print
        print(
            "[debug] template depth stats (m):",
            "name", self.preview_window_name,
            "n_valid_template", int(np.count_nonzero(mask_tmpl)),
            "n_valid_current", int(np.count_nonzero(mask_curr)),
            "n_valid", n_valid,
            "n_fg", n_fg,
            "mean_delta", mean_delta,
            "ratio", area_ratio,
        )

        metrics = {
            # If you actually want millimetres, keep the *1000; if not, drop it.
            "area_ratio": area_ratio ,
            "mean_delta_m": mean_delta,
            "n_fg_pixels": n_fg,
            "n_valid_pixels": n_valid,
        }
        return intrusion, metrics

    # --------- Public toggles ---------
    def enable_face_only(self, enabled: bool = True):
        was_enabled = self.face_only
        self.face_only = bool(enabled)
        new_complexity = 0 if self.face_only else 1
        if new_complexity != self._pose_model_complexity or (self.face_only != was_enabled):
            # Restart worker thread with updated complexity
            self.stop()
            self._pose_model_complexity = new_complexity
            self.start_thread()

    # --------- Lifecycle ---------
    def _set_intrinsics_from_dict(self, intr: Dict[str, float]):
        required = ("fx", "fy", "cx", "cy")
        if not all(k in intr for k in required):
            raise ValueError(f"Intrinsics dict must contain {required}")
        self._color_intrinsics = {
            "fx": float(intr["fx"]),
            "fy": float(intr["fy"]),
            "cx": float(intr["cx"]),
            "cy": float(intr["cy"]),
        }
        if "depth_scale_m" in intr:
            self._depth_scale_m = float(intr["depth_scale_m"])
        elif "scale" in intr:
            self._depth_scale_m = float(intr["scale"])

    def start_thread(self):
        """Start the capture/process loop in a background thread."""
        if self._t and self._t.is_alive():
            return
        self._stop.clear()
        self._t = threading.Thread(target=self._loop, name=f"MP-Tracker-{id(self)}", daemon=True)
        self._t.start()

    def stop(self):
        """Stop the thread and clean up (does NOT stop the shared camera)."""
        self._stop.set()
        if self._t and self._t.is_alive():
            self._t.join(timeout=2.0)
        self._t = None
        # pose/hands are closed in the worker thread's finally block

    # --------- Public, thread-safe getters ---------
    def get_body_positions(
        self,
        transform_4x4: Optional[np.ndarray] = None,
        filter_visible: bool = True,
        visibility_threshold: float = 0.5,
    ) -> Dict[int, np.ndarray]:
        with self._lock:
            lm = copy.deepcopy(self._latest_pose3d)
        return self._landmark_dict_to_xyz(lm, transform_4x4, filter_visible, visibility_threshold)

    def get_hand_positions(
        self,
        hand_index: int = 0,
        transform_4x4: Optional[np.ndarray] = None,
        filter_visible: bool = False,  # MediaPipe Hands has no visibility; keep False
    ) -> Dict[int, np.ndarray]:
        with self._lock:
            if hand_index < 0 or hand_index >= len(self._latest_hands3d):
                return {}
            lm = copy.deepcopy(self._latest_hands3d[hand_index])
        return self._landmark_dict_to_xyz(lm, transform_4x4, filter_visible, 0.0)

    def get_all_hands_positions(
        self, transform_4x4: Optional[np.ndarray] = None
    ) -> List[Dict[int, np.ndarray]]:
        with self._lock:
            hands = copy.deepcopy(self._latest_hands3d)

        # h = [self._landmark_dict_to_xyz(h, transform_4x4, False, 0.0) for h in hands]
        h = [self._landmark_dict_to_xyz(h, transform_4x4, False, 0.0) for h in hands]
        # print(transform_4x4,h)
        return h

    def last_update_time(self) -> float:
        with self._lock:
            return self._last_update_s

    def is_looking_at_camera(self) -> Tuple[Optional[bool], Dict[str, float]]:
        """
        Returns (is_looking, metrics) where:
          - is_looking is True/False if computable, otherwise None.
          - metrics = {'yaw_deg','pitch_deg','nose_ahead_m'} when available (NaN otherwise).
        """
        with self._lock:
            lm = copy.deepcopy(self._latest_pose3d)
        if not lm:
            return None, {"yaw_deg": np.nan, "pitch_deg": np.nan, "nose_ahead_m": np.nan}
        ok, metrics = self._compute_gaze_metrics(lm)
        return ok, metrics

    def get_preview_frame_nowait(self) -> Optional[np.ndarray]:
        """
        Non-blocking: returns the most recent preview frame (BGR) if available, else None.
        """
        try:
            # Drain older frame if multiple pending; keep the newest
            frame = None
            while True:
                frame = self._preview_queue.get_nowait()
        except queue.Empty:
            pass
        return frame

    def get_current_rgb(self):
        depth_image, color_img = self.cam.get_rgbd()
        if depth_image is None or color_img is None:
                time.sleep(0.005)
                return None
        color_for_draw = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
        return color_for_draw

    # --------- Internal capture/processing loop ---------
    def _loop(self):
        pose = None
        hands = None
        
        try:
            # Build graphs on THIS thread
            if self.enable_pose:
                pose = mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=self._pose_model_complexity,
                    enable_segmentation=False,
                    min_detection_confidence=self.min_det_conf,
                    min_tracking_confidence=self.min_trk_conf,
                    smooth_landmarks=True,
                )
                self.pose = pose

            if self.enable_hands:
                hands = mp_hands.Hands(
                    static_image_mode=False,
                    model_complexity=1,
                    max_num_hands=self.max_num_hands,
                    min_detection_confidence=self.min_det_conf,
                    min_tracking_confidence=self.min_trk_conf,
                )
                self.hands = hands

            # Processing loop
            while not self._stop.is_set():
                depth_image, color_img = self.cam.get_rgbd()
                if depth_image is None or color_img is None:
                    time.sleep(0.005)
                    continue

                # Prepare images for MediaPipe / drawing
                if self.color_is_rgb:
                    rgb = color_img
                    color_for_draw = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
                else:
                    color_for_draw = color_img
                    rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

                pose3d: Dict[int, Landmark3D] = {}
                hands3d: List[Dict[int, Landmark3D]] = []

                ok_flag: Optional[bool] = None
                valid_count = 0
                shoulder_w = float("nan")
                torso_len = float("nan")
                med_z = float("nan")
                hands_count = 0

                # ---- Pose ----
                if pose is not None:
                    res = pose.process(rgb)
                    if res.pose_landmarks is not None:
                        pose3d = self._normalized_to_3d(
                            res.pose_landmarks.landmark, color_for_draw.shape, depth_image
                        )
                        valid_count, shoulder_w, torso_len, med_z = self._pose_metrics(pose3d)
                        ok_flag = self._is_pose_plausible(pose3d)

                    # (Optional) draw landmarks on color_for_draw; actual display is main thread only
                    if res.pose_landmarks is not None:
                        mp_drawing.draw_landmarks(
                            color_for_draw, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
                        )

                # ---- Hands ----
                if hands is not None:
                    res_h = hands.process(rgb)
                    if res_h.multi_hand_landmarks:
                        for hand_lms in res_h.multi_hand_landmarks:
                            hands3d.append(
                                self._normalized_to_3d(hand_lms.landmark, color_for_draw.shape, depth_image)
                            )
                            mp_drawing.draw_landmarks(
                                color_for_draw, hand_lms, mp_hands.HAND_CONNECTIONS,
                                mp_drawing.DrawingSpec(thickness=1, circle_radius=2),
                                mp_drawing.DrawingSpec(thickness=2, circle_radius=2)
                            )
                    hands_count = len(hands3d)

                # FPS estimate
                now_ts = time.time()
                fps_est = None if self._last_frame_ts is None else max(1e-6, now_ts - self._last_frame_ts)
                fps_est = None if fps_est is None else 1.0 / fps_est
                self._last_frame_ts = now_ts

                # Gaze metrics
                gaze_ok, metrics = (None, {"yaw_deg": np.nan, "pitch_deg": np.nan, "nose_ahead_m": np.nan})
                if pose3d:
                    gaze_ok, metrics = self._compute_gaze_metrics(pose3d)
                    self._last_gaze = (gaze_ok, metrics)
                else:
                    self._last_gaze = (None, metrics)

                # Publish landmark results (with plausibility gating unless face_only)
                with self._lock:
                    if self.face_only:
                        self._latest_pose3d = pose3d if pose3d else {}
                    else:
                        if pose3d and self._is_pose_plausible(pose3d):
                            self._latest_pose3d = pose3d
                            self._last_good_pose3d = pose3d
                            self._last_good_ts = time.time()
                        else:
                            now = time.time()
                            if self._last_good_pose3d and (now - self._last_good_ts) <= self.keep_last_good_secs:
                                self._latest_pose3d = self._last_good_pose3d
                            else:
                                self._latest_pose3d = {}

                    self._latest_hands3d = hands3d
                    self._last_update_s = time.time()

                # Draw HUD panel onto color_for_draw (for the main-thread viewer)
                self._draw_status(
                    color_for_draw,
                    ok=(ok_flag if not self.face_only else True if pose3d else None),
                    valid_count=valid_count,
                    shoulder_w=shoulder_w,
                    torso_len=torso_len,
                    med_z=med_z,
                    hands_count=hands_count,
                    fps=fps_est,
                    look=gaze_ok,
                    yaw_deg=metrics.get("yaw_deg", np.nan),
                    pitch_deg=metrics.get("pitch_deg", np.nan),
                    nose_ahead_m=metrics.get("nose_ahead_m", np.nan),
                    pose3d_for_overlay=pose3d,
                )

                # Mode badge
                mode_label = "MODE: FACE ONLY" if self.face_only else "MODE: FULL BODY"
                cv2.putText(color_for_draw, mode_label, (10, color_for_draw.shape[0]-12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

                # Publish preview frame (BGR) to queue for DisplayHub
                if self.debug:
                    try:
                        if self._preview_queue.full():
                            _ = self._preview_queue.get_nowait()
                        self._preview_queue.put_nowait(color_for_draw)
                    except queue.Full:
                        pass

                # Tiny sleep helps responsiveness without starving other threads
                time.sleep(0.001)

        except Exception as e:
            print(f"[Tracker] Loop error: {e}")
        finally:
            # Close models on the same thread they were created
            if pose is not None:
                try:
                    pose.close()
                except Exception:
                    pass
            if hands is not None:
                try:
                    hands.close()
                except Exception:
                    pass
            self.pose = None
            self.hands = None

    # --------- Helpers ---------
    @staticmethod
    def _clamp(v: int, lo: int, hi: int) -> int:
        return max(lo, min(hi, v))

    def _depth_at(self, depth_image: np.ndarray, u: int, v: int) -> float:
        d = self._read_depth(depth_image, u, v)
        if d > 0:
            return d
        max_r = 3
        for r in range(1, max_r + 1):
            for dv in range(-r, r + 1):
                for du in range(-r, r + 1):
                    uu = self._clamp(u + du, 0, depth_image.shape[1] - 1)
                    vv = self._clamp(v + dv, 0, depth_image.shape[0] - 1)
                    d = self._read_depth(depth_image, uu, vv)
                    if d > 0:
                        return d
        return 0.0

    def _read_depth(self, depth_image: np.ndarray, u: int, v: int) -> float:
        raw = float(depth_image[v, u])
        if raw <= 0:
            return 0.0
        if np.issubdtype(depth_image.dtype, np.floating):
            return raw
        return raw * self._depth_scale_m

    def _normalized_to_3d(self, lms, image_shape, depth_image) -> Dict[int, Landmark3D]:
        H, W = image_shape[0], image_shape[1]
        out: Dict[int, Landmark3D] = {}
        intr = self._color_intrinsics
        if intr is None:
            return out

        fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]

        for i, lm in enumerate(lms):
            u = self._clamp(int(round(lm.x * W)), 0, W - 1)
            v = self._clamp(int(round(lm.y * H)), 0, H - 1)
            Z = self._depth_at(depth_image, u, v)  # meters
            if Z <= 0:
                out[i] = Landmark3D(i, np.nan, np.nan, np.nan, float(getattr(lm, "visibility", 1.0)))
                continue
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            out[i] = Landmark3D(i, float(X), float(Y), float(Z), float(getattr(lm, "visibility", 1.0)))
        return out

    @staticmethod
    def _apply_transform(XYZ: np.ndarray, T: Optional[np.ndarray]) -> np.ndarray:
        if T is None:
            return XYZ
        if T.shape != (4, 4):
            raise ValueError("transform_4x4 must be 4x4")
        out = XYZ.copy()
        mask = ~np.any(np.isnan(out), axis=1)
        if not np.any(mask):
            return out
        homo = np.ones((out.shape[0], 4), dtype=np.float64)
        homo[:, :3] = out
        homo[mask] = (T @ homo[mask].T).T
        out[mask] = homo[mask, :3]
        return out

    def _landmark_dict_to_xyz(
        self,
        lm_dict: Dict[int, Landmark3D],
        T: Optional[np.ndarray],
        filter_visible: bool,
        vis_thr: float,
    ) -> Dict[int, np.ndarray]:
        if not lm_dict:
            return {}
        idxs = sorted(lm_dict.keys())
        arr = np.array([[lm_dict[i].X, lm_dict[i].Y, lm_dict[i].Z] for i in idxs], dtype=np.float64)
        if filter_visible:
            vis = np.array([lm_dict[i].vis for i in idxs], dtype=np.float32)
        else:
            vis = np.ones((len(idxs),), dtype=np.float32)

        arr = self._apply_transform(arr, T)

        result: Dict[int, np.ndarray] = {}
        for i, idx in enumerate(idxs):
            if filter_visible and not np.isnan(arr[i]).any() and vis[i] < vis_thr:
                continue
            result[idx] = arr[i]
        return result

    # --------- Pose filtering helpers ---------
    @staticmethod
    def _nanfree(arr: np.ndarray) -> np.ndarray:
        return arr[~np.isnan(arr).any(axis=1)]

    def _pose_has_core_torso(self, lm: Dict[int, Landmark3D]) -> bool:
        need = [11, 12, 23, 24]
        for i in need:
            if i not in lm:
                return False
            if np.isnan([lm[i].X, lm[i].Y, lm[i].Z]).any():
                return False
        return True

    def _pose_metrics(self, lm: Dict[int, Landmark3D]) -> Tuple[int, float, float, float]:
        if not lm:
            return 0, np.nan, np.nan, np.nan

        pts = []
        for _, L in lm.items():
            xyz = np.array([L.X, L.Y, L.Z], dtype=np.float64)
            if not np.isnan(xyz).any():
                pts.append(xyz)
        valid_count = len(pts)
        if valid_count == 0:
            return 0, np.nan, np.nan, np.nan
        pts = np.vstack(pts)
        median_depth = float(np.median(pts[:, 2]))

        # Shoulders (11,12)
        if 11 in lm and 12 in lm:
            sL = np.array([lm[11].X, lm[11].Y, lm[11].Z])
            sR = np.array([lm[12].X, lm[12].Y, lm[12].Z])
            if not (np.isnan(sL).any() or np.isnan(sR).any()):
                shoulder_width = float(np.linalg.norm(sL - sR))
                shoulder_mid = (sL + sR) / 2.0
            else:
                shoulder_width = np.nan
                shoulder_mid = np.array([np.nan, np.nan, np.nan])
        else:
            shoulder_width = np.nan
            shoulder_mid = np.array([np.nan, np.nan, np.nan])

        # Hips (23,24)
        if 23 in lm and 24 in lm:
            hL = np.array([lm[23].X, lm[23].Y, lm[23].Z])
            hR = np.array([lm[24].X, lm[24].Y, lm[24].Z])
            if not (np.isnan(hL).any() or np.isnan(hR).any()):
                hip_mid = (hL + hR) / 2.0
            else:
                hip_mid = np.array([np.nan, np.nan, np.nan])
        else:
            hip_mid = np.array([np.nan, np.nan, np.nan])

        torso_len = float(np.linalg.norm(shoulder_mid - hip_mid)) if not (
            np.isnan(shoulder_mid).any() or np.isnan(hip_mid).any()
        ) else np.nan

        return valid_count, shoulder_width, torso_len, median_depth

    def _is_pose_plausible(self, lm: Dict[int, Landmark3D]) -> bool:
        if not self._pose_has_core_torso(lm):
            return False
        valid_count, shoulder_w, torso_len, med_z = self._pose_metrics(lm)
        if valid_count < self.min_visible_kps:
            return False
        if np.isnan(shoulder_w) or not (self.min_shoulder_width_m <= shoulder_w <= self.max_shoulder_width_m):
            return False
        if np.isnan(torso_len) or not (self.min_torso_len_m <= torso_len <= self.max_torso_len_m):
            return False
        if np.isnan(med_z) or not (self.min_body_depth_m <= med_z <= self.max_body_depth_m):
            return False
        return True

    # --------- Head gaze helpers ---------
    def _compute_gaze_metrics(self, lm: Dict[int, Landmark3D]) -> Tuple[Optional[bool], Dict[str, float]]:
        # Need nose and a head-center proxy
        if 0 not in lm or any(np.isnan([lm[0].X, lm[0].Y, lm[0].Z])):
            return None, {"yaw_deg": np.nan, "pitch_deg": np.nan, "nose_ahead_m": np.nan}

        centers = []
        for idx in (7, 8):  # ears
            if idx in lm and not np.isnan([lm[idx].X, lm[idx].Y, lm[idx].Z]).any():
                centers.append(np.array([lm[idx].X, lm[idx].Y, lm[idx].Z], dtype=np.float64))
        if len(centers) < 2:
            centers = []
            for idx in (2, 5):  # eyes
                if idx in lm and not np.isnan([lm[idx].X, lm[idx].Y, lm[idx].Z]).any():
                    centers.append(np.array([lm[idx].X, lm[idx].Y, lm[idx].Z], dtype=np.float64))
        if len(centers) < 2:
            return None, {"yaw_deg": np.nan, "pitch_deg": np.nan, "nose_ahead_m": np.nan}

        head_center = (centers[0] + centers[1]) / 2.0
        nose = np.array([lm[0].X, lm[0].Y, lm[0].Z], dtype=np.float64)

        v = nose - head_center  # camera coords: +Z forward; facing camera => v ≈ [0, 0, negative]
        if np.isnan(v).any() or abs(v[2]) < 1e-6:
            return None, {"yaw_deg": np.nan, "pitch_deg": np.nan, "nose_ahead_m": np.nan}

        yaw_rad = np.arctan2(v[0], -v[2])
        pitch_rad = np.arctan2(v[1], -v[2])
        yaw_deg = float(np.degrees(yaw_rad))
        pitch_deg = float(np.degrees(pitch_rad))
        nose_ahead_m = float(head_center[2] - nose[2])  # positive if nose closer to camera

        is_looking = (
            (abs(yaw_deg) <= self.max_face_yaw_deg) and
            (abs(pitch_deg) <= self.max_face_pitch_deg) and
            (nose_ahead_m >= self.min_nose_ahead_m)
        )
        return is_looking, {"yaw_deg": yaw_deg, "pitch_deg": pitch_deg, "nose_ahead_m": nose_ahead_m}

    def _project_point(self, XYZ: np.ndarray) -> Optional[Tuple[int, int]]:
        intr = self._color_intrinsics
        if intr is None or np.isnan(XYZ).any() or abs(XYZ[2]) < 1e-6:
            return None
        fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
        u = int(round(fx * XYZ[0] / XYZ[2] + cx))
        v = int(round(fy * XYZ[1] / XYZ[2] + cy))
        return u, v

    def _draw_status(self, img_bgr: np.ndarray, ok: Optional[bool],
                     valid_count: int = 0,
                     shoulder_w: float = float("nan"),
                     torso_len: float = float("nan"),
                     med_z: float = float("nan"),
                     hands_count: int = 0,
                     fps: Optional[float] = None,
                     look: Optional[bool] = None,
                     yaw_deg: float = float("nan"),
                     pitch_deg: float = float("nan"),
                     nose_ahead_m: float = float("nan"),
                     pose3d_for_overlay: Optional[Dict[int, Landmark3D]] = None) -> None:
        h, w = img_bgr.shape[:2]
        pad = 8
        line_h = 22
        lines = []

        if ok is None:
            header = "POSE: NO POSE"
            color = (50, 180, 255)   # orange (BGR)
        elif ok:
            header = "POSE: GOOD"
            color = (60, 200, 60)    # green
        else:
            header = "POSE: BAD"
            color = (60, 60, 220)    # red

        if look is None:
            gaze_str = "LOOK: N/A"
            gaze_color = (128, 128, 128)
        elif look:
            gaze_str = "LOOK: YES"
            gaze_color = (60, 200, 60)
        else:
            gaze_str = "LOOK: NO"
            gaze_color = (60, 60, 220)

        lines.append(header)
        lines.append(f"valid:{valid_count:2d}  SW:{shoulder_w:0.2f}m  TL:{torso_len:0.2f}m")
        lines.append(f"Zmed:{med_z:0.2f}m  hands:{hands_count}")
        lines.append(f"{gaze_str}  yaw:{yaw_deg:0.1f}°  pitch:{pitch_deg:0.1f}°  noseΔ:{nose_ahead_m:0.2f}m")
        if fps is not None:
            lines.append(f"FPS:{fps:4.1f}")

        # Panel background
        box_w = 0
        for s in lines:
            sz = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            box_w = max(box_w, sz[0])
        box_h = line_h * len(lines) + pad
        box_w = box_w + 2 * pad

        overlay = img_bgr.copy()
        cv2.rectangle(overlay, (pad, pad), (pad + box_w, pad + box_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.35, img_bgr, 0.65, 0, img_bgr)

        cv2.rectangle(img_bgr, (pad, pad), (pad + 8, pad + box_h), color, -1)
        cv2.rectangle(img_bgr, (pad, pad), (pad + 8, pad + 8), gaze_color, -1)

        y = pad + 18
        for i, s in enumerate(lines):
            c = color if i == 0 else (240, 240, 240)
            cv2.putText(img_bgr, s, (pad + 16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2, cv2.LINE_AA)
            y += line_h

        # Visualize head-center -> nose vector (if available)
        if pose3d_for_overlay and (0 in pose3d_for_overlay):
            pts = []
            for idx in (7, 8):  # ears
                if idx in pose3d_for_overlay:
                    L = pose3d_for_overlay[idx]
                    if not np.isnan([L.X, L.Y, L.Z]).any():
                        pts.append(np.array([L.X, L.Y, L.Z]))
            if len(pts) < 2:
                pts = []
                for idx in (2, 5):  # eyes
                    if idx in pose3d_for_overlay:
                        L = pose3d_for_overlay[idx]
                        if not np.isnan([L.X, L.Y, L.Z]).any():
                            pts.append(np.array([L.X, L.Y, L.Z]))
            if len(pts) >= 2:
                head_center = (pts[0] + pts[1]) / 2.0
                nose = pose3d_for_overlay[0]
                hc_uv = self._project_point(head_center)
                nose_uv = self._project_point(np.array([nose.X, nose.Y, nose.Z]))
                if hc_uv and nose_uv:
                    cv2.circle(img_bgr, hc_uv, 4, (0, 255, 255), -1)
                    cv2.circle(img_bgr, nose_uv, 4, (255, 255, 0), -1)
                    cv2.arrowedLine(img_bgr, hc_uv, nose_uv, (0, 255, 0), 2, tipLength=0.3)


# ---------- Convenience: build a 4x4 from R (3x3) and t (3,) ----------
def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3,)
    return T


# ---------- Example “camera -> ROS REP-103” rotation ----------
def camera_to_ros_T() -> np.ndarray:
    """
    RealSense camera frame: +X right, +Y down, +Z forward
    ROS REP-103 (base_link): +X forward, +Y left, +Z up
    """
    R = np.array([[0,  0, 1],
                  [-1, 0, 0],
                  [0, -1, 0]], dtype=np.float64)
    return make_T(R, np.zeros(3))


# ========== Simple main-thread display hub ==========
# class DisplayHub:
#     def __init__(self, trackers):
#         self.trackers = trackers
#         self._want_quit = False
#         self.show_depth_debug = False  # toggle with 'd'

#     def tick(self, poll_ms: int = 1) -> bool:
#         """Run one non-blocking UI tick. Return True if user requested quit."""
#         for trk in self.trackers:
#             frame = trk.get_preview_frame_nowait()
#             if frame is not None:
#                 cv2.imshow(trk.preview_window_name, frame)
#             else:
#                 frame = trk.get_current_rgb()
#                 if frame is not None:
#                      cv2.imshow(trk.preview_window_name, frame)

#             if self.show_depth_debug:
#                 tmpl, delta = trk.get_depth_debug_images()
#                 if tmpl is not None:
#                     cv2.imshow(trk.preview_window_name + " [Depth Template]", tmpl)
#                 if delta is not None:
#                     cv2.imshow(trk.preview_window_name + " [Depth Delta]", delta)

#         # keep UI responsive even if no frames
#         key = cv2.waitKey(poll_ms) & 0xFF
#         if key == 27:  # ESC
#             self._want_quit = True
#         elif key == ord('d'):
#             # toggle depth debug
#             self.show_depth_debug = not self.show_depth_debug
#             if not self.show_depth_debug:
#                 # close extra windows when turning off
#                 for trk in self.trackers:
#                     cv2.destroyWindow(trk.preview_window_name + " [Depth Template]")
#                     cv2.destroyWindow(trk.preview_window_name + " [Depth Delta]")

#         return self._want_quit

#     def close(self):
#         try:
#             cv2.destroyAllWindows()
#         except Exception:
#             pass

class DisplayHub:
    def __init__(self, trackers):
        self.trackers = trackers
        self._want_quit = False
        self.show_depth_debug = False  # toggle with 'd'

    def tick(self, poll_ms: int = 1) -> bool:
        """Run one non-blocking UI tick. Return True if user requested quit."""
        for trk in self.trackers:
            # Main RGB preview
            frame = trk.get_preview_frame_nowait()
            if frame is not None:
                cv2.imshow(trk.preview_window_name, frame)
            else:
                frame = trk.get_current_rgb()
                if frame is not None:
                    cv2.imshow(trk.preview_window_name, frame)

            # Optional depth debug views
            if self.show_depth_debug:
                tmpl, delta = trk.get_depth_debug_images()
                if tmpl is not None:
                    cv2.imshow(trk.preview_window_name + " [Depth Template]", tmpl)

                if delta is not None:
                    # Get intrusion info and overlay it on the delta image
                    intrusion, metrics = trk.check_depth_intrusion()
                    delta_with_hud = delta.copy()

                    area_ratio = metrics.get("area_ratio", 0.0)
                    mean_delta = metrics.get("mean_delta_m", 0.0)
                    n_fg = metrics.get("n_fg_pixels", 0)
                    n_valid = metrics.get("n_valid_pixels", 0)

                    lines = [
                        f"INTRUSION: {'YES' if intrusion else 'NO'}",
                        f"area_ratio: {area_ratio:.3f}",
                        f"mean_delta: {mean_delta:.3f} m",
                        f"fg_pixels: {n_fg}/{n_valid}",
                    ]

                    pad = 8
                    line_h = 22
                    box_w = 0
                    for s in lines:
                        sz = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        box_w = max(box_w, sz[0])
                    box_h = line_h * len(lines) + pad
                    box_w = box_w + 2 * pad

                    # Panel background
                    overlay = delta_with_hud.copy()
                    cv2.rectangle(
                        overlay,
                        (pad, pad),
                        (pad + box_w, pad + box_h),
                        (0, 0, 0),
                        -1,
                    )
                    cv2.addWeighted(overlay, 0.35, delta_with_hud, 0.65, 0, delta_with_hud)

                    # Left strip: green when intrusion, red when none
                    strip_color = (0, 255, 0) if intrusion else (0, 0, 255)
                    cv2.rectangle(delta_with_hud, (pad, pad), (pad + 8, pad + box_h), strip_color, -1)

                    # Text
                    y = pad + 18
                    for i, s in enumerate(lines):
                        c = strip_color if i == 0 else (240, 240, 240)
                        cv2.putText(
                            delta_with_hud,
                            s,
                            (pad + 16, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            c,
                            2,
                            cv2.LINE_AA,
                        )
                        y += line_h

                    cv2.imshow(trk.preview_window_name + " [Depth Delta]", delta_with_hud)

        # keep UI responsive even if no frames
        key = cv2.waitKey(poll_ms) & 0xFF
        if key == 27:  # ESC
            self._want_quit = True

        elif key == ord('d'):
            # toggle depth debug
            self.show_depth_debug = not self.show_depth_debug
            if not self.show_depth_debug:
                # close extra windows when turning off
                for trk in self.trackers:
                    cv2.destroyWindow(trk.preview_window_name + " [Depth Template]")
                    cv2.destroyWindow(trk.preview_window_name + " [Depth Delta]")

        elif key == ord('t'):
            # Capture/build depth template for each tracker
            print("[DisplayHub] Building depth template(s)...")
            for trk in self.trackers:
                try:
                    trk.build_depth_template()
                    print(f"[DisplayHub] Depth template built for {trk.preview_window_name}")
                except Exception as e:
                    print(f"[DisplayHub] Error building depth template for {trk.preview_window_name}: {e}")
            # After capturing, automatically show debug windows so user sees the result
            self.show_depth_debug = True

        return self._want_quit

    def close(self):
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

# ========== Example usage ==========
"""
from hardware_init import HardwareInitializer, PipelineConfig

if __name__ == "__main__":
    # Optional: make OpenCV single-threaded to avoid rare UI contention
    cv2.setNumThreads(1)

    cfg = PipelineConfig(...)           # your config
    hw = HardwareInitializer(cfg).initialize()

    face_tracker = RealSenseMediapipeTracker(
        camera_obj=hw.cam_arm,
        cam_cfg=hw.cam_arm_cfg,
        face_only=True,
        enable_hands=False,
        debug=True,  # publish preview frames for DisplayHub
        preview_window_name="Arm Cam + MediaPipe",
        color_is_rgb=True,  # set accordingly
    )

    face2_tracker = RealSenseMediapipeTracker(
        camera_obj=hw.cam_fixed,
        cam_cfg=hw.cam_fixed_cfg,
        face_only=False,
        enable_hands=True,
        debug=True,  # publish preview frames
        preview_window_name="Fixed Cam + MediaPipe",
        color_is_rgb=True,  # set accordingly
    )

    face_tracker.start_thread()
    face2_tracker.start_thread()

    # (Optional) read landmarks in your app loop on a timer/thread
    # body_xyz = face_tracker.get_body_positions()

    # All GUI happens here on the MAIN thread:
    DisplayHub([face_tracker, face2_tracker]).loop(poll_ms=10)

    # Cleanup
    face_tracker.stop()
    face2_tracker.stop()
"""
