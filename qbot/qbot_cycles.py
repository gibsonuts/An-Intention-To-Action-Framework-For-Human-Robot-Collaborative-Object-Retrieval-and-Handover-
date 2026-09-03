#!/usr/bin/env python3
"""
Refactored screw detection and installation system with YAML configuration.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
import math
import re
import time
import yaml
from typing import List, Dict, Any, Tuple, Optional, Callable
import numpy as np
import select
import cv2
import commons.utils as utils
# from your_module import ScrewdriverClient, OpenAIRealtimeClient  # adjust imports as needed


from detectors.sam3_object_detection import (
    Sam3Detector,
    filter_masks_by_overlap,
    filter_masks_by_non_overlap,
    draw_mask_debug,
    mask_centroid,
    masks_overlap,
)
from hardware.hardware_init import HardwareInitializer
from hardware.screwdriver_client import ScrewdriverClient
from llm.openai_query_client import OpenAiQueryClient
from llm.openai_realtime_client import OpenAIRealtimeClient
import commons.utils as utils


CFG_PATH = 'config/cycles.yaml'


def confirm_active_tcp(handles, tool_name: str = "tcp_drill", *, context: str = "planning") -> List[float]:
    """Apply the configured flange-to-TCP pose before frame-sensitive work."""
    tool_transforms = getattr(handles, "tool_tcp_transforms", {}) or {}
    if tool_name not in tool_transforms:
        raise RuntimeError(
            f"Missing {tool_name} configuration in hardware/config/arm_control.yaml"
        )

    tcp_pose = [
        float(value)
        for value in utils.matrix_to_ur_pose(
            np.asarray(tool_transforms[tool_name], dtype=float)
        )
    ]
    response = handles.arm.set_tcp(tcp_pose)
    if isinstance(response, dict) and response.get("ok") is False:
        raise RuntimeError(
            f"Failed to confirm {tool_name} before {context}: {response!r}"
        )

    handles.tool_name = tool_name
    handles.arm.tcp_pose = list(tcp_pose)
    actual_pose = [float(value) for value in handles.arm.get_tcp_pose_axis_angle()]
    print(
        f"[TCP] Confirmed {tool_name} before {context}; "
        f"configured_tcp={tcp_pose}, actual_base_tcp={actual_pose}."
    )
    return tcp_pose


def apply_mask_to_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply a binary mask to an RGB image.

    Thin wrapper so mask asset tooling is callable from this module without
    duplicating the implementation.
    """
    if image.ndim == 2:
        from qbot.mask_asset_tools import apply_binary_mask_depth
        return apply_binary_mask_depth(image, mask)

    from qbot.mask_asset_tools import apply_binary_mask

    return apply_binary_mask(image, mask)


def compare_image_with_saved_mask_asset(
    image: np.ndarray,
    asset_name: str,
    assets_dir: Optional[str] = None,
    resize_input: bool = True,
) -> Dict[str, Any]:
    """
    Mask `image` with the saved asset mask and compare it to the saved masked image.

    Returns a dict containing `similarity_score` (0..1) plus diagnostic metrics.
    """
    if image.ndim == 2:
        from qbot.mask_asset_tools import compare_depth_to_saved_mask_asset

        return compare_depth_to_saved_mask_asset(
            depth_u16=image,
            asset_name=asset_name,
            assets_dir=assets_dir,
            resize_input=resize_input,
        )

    from qbot.mask_asset_tools import compare_image_to_saved_mask_asset

    return compare_image_to_saved_mask_asset(
        image_rgb=image,
        asset_name=asset_name,
        assets_dir=assets_dir,
        resize_input=resize_input,
    )

class Config:
    """Configuration container loaded from YAML."""
    
    def __init__(self, config_path: str = "screw_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def get(self, *keys, default=None):
        """Get nested config value using dot notation."""
        value = self.config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return default
            if value is None:
                return default
        return value
  
    @property
    def screw_length(self):
        return self.get('motion', 'screw_length')
    
    @property
    def default_speed(self):
        return self.get('motion', 'default_speed')
    
    @property
    def prompts(self):
        return self.get('prompts', default={})
    
    @property
    def offsets(self):
        return self.get('motion', 'offsets')

    @property
    def constraints(self):
        return self.get('motion', 'constraints')

    @property
    def confidence(self):
        # Backward compatibility: older configs used `confidence`, newer configs use
        # camera-specific `confidence_fixed` / `confidence_arm`.
        return (
            self.get('confidence')
            or self.get('confidence_fixed')
            or self.get('confidence_arm')
            or {}
        )

    def confidence_for_camera(self, camera_type: Optional[str] = None) -> Dict[str, Any]:
        """Return confidence thresholds for the requested camera with sensible fallbacks."""
        cam = (camera_type or "").strip().lower()
        if cam in (
            "arm", "camera_arm", "drill", "gripper", "camera_drill",
            "camera_gripper", "tcp_drill", "tcp_gripper",
        ):
            primary = self.get('confidence_arm', default={}) or {}
            fallback = self.get('confidence_fixed', default={}) or {}
        else:
            primary = self.get('confidence_fixed', default={}) or {}
            fallback = self.get('confidence_arm', default={}) or {}

        legacy = self.get('confidence', default={}) or {}
        merged = {}
        merged.update(legacy)
        merged.update(fallback)
        merged.update(primary)
        return merged

    @property
    def servo(self):
        return self.get('motion','servo')
    
    @property
    def filtering(self):
        return self.get('filtering', default={})


class DetectionResult:
    """Container for detection results."""
    def __init__(self, targets: List[Dict[str, Any]], moved: bool = False):
        self.targets = targets
        self.moved = moved
        self.success = len(targets) > 0


class CameraHelper:
    """Helper for camera operations."""
    
    def __init__(self, handles, config: Config):
        self.handles = handles
        self.config = config
    
    def get_rgbd_and_intrinsics(self, camera_type: str):
        """Get RGBD image and intrinsics for specified camera."""
        camera_type = str(camera_type).strip().lower()
        if camera_type == "fixed":
            depth, color = self.handles.cam_fixed.get_rgbd()
            intr = self.handles.cam_fixed.intrinsics
            T_cam = self.handles.T_base_fixed_camera
        elif camera_type == "arm":
            depth, color = self.handles.cam_arm.get_rgbd()
            intr = self.handles.cam_arm.intrinsics
            T_cam = self.handles.T_tcp_cam
        elif camera_type in (getattr(self.handles, "tool_cameras", {}) or {}):
            camera = self.handles.tool_cameras[camera_type]
            depth, color = camera.get_rgbd()
            intr = camera.intrinsics
            T_cam = self.handles.tool_camera_transforms[camera_type]
        else:
            raise ValueError(f"Unknown camera_type: {camera_type}")
        
        return depth, color, intr, T_cam
    
    def project_tcp_to_fixed_cam(self) -> Tuple[Optional[float], Optional[float]]:
        """Project current TCP position to fixed camera image coordinates."""
        T_base_tcp = self.handles.arm.get_T_base_tcp()
        p_base_tcp = T_base_tcp[:3, 3].reshape(3,)
        
        T_base_cam = self.handles.T_base_fixed_camera
        T_cam_base = np.linalg.inv(T_base_cam)
        
        p_base_h = np.concatenate([p_base_tcp, [1.0]])
        p_cam_h = T_cam_base @ p_base_h
        Xc, Yc, Zc, _ = p_cam_h
        
        if Zc <= 1e-6:
            print("[WARN] TCP behind camera or too close")
            return None, None
        
        intr = self.handles.cam_fixed.intrinsics
        fx, fy, cx, cy = utils._get_intrinsics_fx_fy_cx_cy(intr)
        
        u = fx * (Xc / Zc) + cx
        v = fy * (Yc / Zc) + cy
        
        return float(u), float(v)

    def project_base_point_to_image(
        self,
        point_base: Any,
        *,
        camera_type: str,
        T_base_tcp: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Project a base-frame 3D point into the requested camera image."""
        p_base = np.asarray(point_base, dtype=float).reshape(-1)
        if p_base.size < 3:
            raise ValueError("point_base must contain at least 3 values")
        p_base = p_base[:3]

        cam = str(camera_type or "").strip().lower()
        if cam == "fixed":
            T_base_cam = self.handles.T_base_fixed_camera
            intr = self.handles.cam_fixed.intrinsics
        elif cam == "arm":
            if T_base_tcp is None:
                T_base_tcp = self.handles.arm.get_T_base_tcp()
            T_base_cam = np.asarray(T_base_tcp, dtype=float) @ np.asarray(self.handles.T_tcp_cam, dtype=float)
            intr = self.handles.cam_arm.intrinsics
        else:
            raise ValueError(f"Unknown camera_type: {camera_type}")

        T_cam_base = np.linalg.inv(T_base_cam)
        p_cam_h = T_cam_base @ np.concatenate([p_base, [1.0]])
        Xc, Yc, Zc, _ = p_cam_h
        if Zc <= 1e-6:
            return {
                "ok": False,
                "reason": "point behind camera",
                "point_cam": [float(Xc), float(Yc), float(Zc)],
            }

        fx, fy, cx, cy = utils._get_intrinsics_fx_fy_cx_cy(intr)
        u = float(fx) * (float(Xc) / float(Zc)) + float(cx)
        v = float(fy) * (float(Yc) / float(Zc)) + float(cy)
        return {
            "ok": True,
            "pixel": (u, v),
            "depth_m": float(Zc),
            "point_cam": [float(Xc), float(Yc), float(Zc)],
            "T_base_cam": np.asarray(T_base_cam, dtype=float),
        }


class MotionPlanner:
    """Handles motion planning and execution."""
    
    def __init__(self, handles, config: Config):
        self.handles = handles
        self.config = config

    def _orient_tcp_y_to_base(
        self,
        approach: np.ndarray,
        R_reference: np.ndarray,
        orientation_profile: str = "target",
    ) -> np.ndarray:
        """Apply the configured target or pickup TCP orientation."""
        orientation_cfg = self.config.get(
            'motion',
            f'{orientation_profile}_orientation',
            default=self.config.get('motion', 'target_orientation', default={}) or {},
        ) or {}
        R_reference = np.asarray(R_reference, dtype=float)
        z_axis = np.asarray(R_reference[:, 2], dtype=float).reshape(3,)
        z_norm = float(np.linalg.norm(z_axis))
        if z_norm < 1e-8:
            return R_reference.copy()
        z_axis = z_axis / z_norm

        if bool(orientation_cfg.get('straight_down', False)):
            z_axis = np.array([0.0, 0.0, -1.0], dtype=float)
            x_axis = np.asarray(R_reference[:, 0], dtype=float).reshape(3,)
            x_axis -= z_axis * float(np.dot(x_axis, z_axis))
            if np.linalg.norm(x_axis) < 1e-8:
                x_axis = np.asarray(R_reference[:, 1], dtype=float).reshape(3,)
                x_axis -= z_axis * float(np.dot(x_axis, z_axis))
            if np.linalg.norm(x_axis) < 1e-8:
                x_axis = np.array([1.0, 0.0, 0.0], dtype=float)
            x_axis = utils.normalize(x_axis)
            y_axis = utils.normalize(np.cross(z_axis, x_axis))
            return np.column_stack((x_axis, y_axis, z_axis))

        axis_to_base = str(orientation_cfg.get('tcp_axis_to_base', '')).strip().lower()
        if axis_to_base in ('+x', '-x', '+y', '-y'):
            base_point = np.asarray(
                orientation_cfg.get('base_point', [0.0, 0.0, 0.0]),
                dtype=float,
            ).reshape(3,)
            toward_base = base_point - np.asarray(approach, dtype=float).reshape(3,)
            toward_base -= z_axis * float(np.dot(toward_base, z_axis))
            if np.linalg.norm(toward_base) >= 1e-8:
                toward_base = utils.normalize(toward_base)
                if axis_to_base in ('+x', '-x'):
                    x_axis = toward_base if axis_to_base == '+x' else -toward_base
                    y_axis = np.cross(z_axis, x_axis)
                else:
                    y_axis = toward_base if axis_to_base == '+y' else -toward_base
                    x_axis = np.cross(y_axis, z_axis)
                x_axis = utils.normalize(x_axis)
                y_axis = utils.normalize(y_axis)
                return np.column_stack((x_axis, y_axis, z_axis))

        y_direction_base = orientation_cfg.get('tcp_y_direction_base')
        if y_direction_base is not None:
            y_hint = np.asarray(y_direction_base, dtype=float).reshape(3,)
        elif bool(orientation_cfg.get('tcp_y_to_base', True)):
            base_point = np.asarray(
                orientation_cfg.get('base_point', [0.0, 0.0, 0.0]),
                dtype=float,
            ).reshape(3,)
            y_hint = base_point - np.asarray(approach, dtype=float).reshape(3,)
        else:
            return R_reference.copy()

        # Keep TCP +Y perpendicular to the fixed TCP +Z approach axis.
        y_hint = y_hint - z_axis * float(np.dot(y_hint, z_axis))

        if np.linalg.norm(y_hint) < 1e-8:
            y_hint = np.asarray(R_reference[:, 1], dtype=float).reshape(3,)
            y_hint = y_hint - z_axis * float(np.dot(y_hint, z_axis))

        if np.linalg.norm(y_hint) < 1e-8:
            fallback = np.array([1.0, 0.0, 0.0], dtype=float)
            if abs(float(np.dot(fallback, z_axis))) > 0.9:
                fallback = np.array([0.0, 1.0, 0.0], dtype=float)
            y_hint = np.cross(z_axis, fallback)

        y_axis = utils.normalize(y_hint)
        x_axis = np.cross(y_axis, z_axis)
        x_norm = float(np.linalg.norm(x_axis))
        if x_norm < 1e-8:
            return R_reference.copy()
        x_axis = x_axis / x_norm
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / float(np.linalg.norm(y_axis))

        return np.column_stack((x_axis, y_axis, z_axis))

    def _gripper_load_stem_axis(self) -> str:
        orientation_cfg = self.config.get(
            "motion", "gripper_load_orientation", default={}
        ) or {}
        axis = str(orientation_cfg.get("tcp_axis_along_stem", "+x")).strip().lower()
        if axis not in ("+x", "-x", "+y", "-y"):
            print(
                f"[WARN] Invalid gripper-load tcp_axis_along_stem {axis!r}; "
                "using '+x'."
            )
            return "+x"
        return axis

    def _orient_tcp_axis_to_stem_vertical_down(
        self,
        stem_angle_deg_cam: Optional[float],
        stem_direction_cam: Optional[Any],
        T_base_cam: np.ndarray,
        R_fallback: np.ndarray,
        *,
        orientation_config_name: str = "gripper_load_orientation",
        axis_config_name: str = "tcp_axis_along_stem",
    ) -> np.ndarray:
        """Set TCP +Z down and align a configured signed X/Y axis with the stem."""
        orientation_cfg = self.config.get(
            "motion", orientation_config_name, default={}
        ) or {}
        z_axis = np.asarray(
            orientation_cfg.get("tcp_z_direction_base", [0.0, 0.0, -1.0]),
            dtype=float,
        ).reshape(3,)
        z_norm = float(np.linalg.norm(z_axis))
        if z_norm < 1e-8:
            z_axis = np.array([0.0, 0.0, -1.0], dtype=float)
        else:
            z_axis /= z_norm

        stem_axis = None
        if stem_direction_cam is not None:
            direction = np.asarray(stem_direction_cam, dtype=float).reshape(-1)
            if direction.size >= 2:
                stem_axis_cam = np.array([direction[0], direction[1], 0.0], dtype=float)
                stem_axis_base = np.asarray(T_base_cam, dtype=float)[:3, :3] @ stem_axis_cam
                stem_axis_base -= z_axis * float(np.dot(stem_axis_base, z_axis))
                if float(np.linalg.norm(stem_axis_base)) >= 1e-8:
                    stem_axis = stem_axis_base
        if stem_axis is None and stem_angle_deg_cam is not None:
            angle_rad = math.radians(float(stem_angle_deg_cam))
            stem_axis_cam = np.array(
                [math.cos(angle_rad), math.sin(angle_rad), 0.0], dtype=float
            )
            stem_axis_base = np.asarray(T_base_cam, dtype=float)[:3, :3] @ stem_axis_cam
            stem_axis_base -= z_axis * float(np.dot(stem_axis_base, z_axis))
            if float(np.linalg.norm(stem_axis_base)) >= 1e-8:
                stem_axis = stem_axis_base

        if stem_axis is None:
            stem_axis = np.asarray(R_fallback, dtype=float)[:3, 0].copy()
            stem_axis -= z_axis * float(np.dot(stem_axis, z_axis))
        if float(np.linalg.norm(stem_axis)) < 1e-8:
            stem_axis = np.array([1.0, 0.0, 0.0], dtype=float)
            stem_axis -= z_axis * float(np.dot(stem_axis, z_axis))
        if float(np.linalg.norm(stem_axis)) < 1e-8:
            stem_axis = np.array([0.0, 1.0, 0.0], dtype=float)
            stem_axis -= z_axis * float(np.dot(stem_axis, z_axis))
        stem_norm = float(np.linalg.norm(stem_axis))
        if stem_norm < 1e-8:
            return R_fallback.copy()
        stem_axis /= stem_norm

        selected_axis = str(
            orientation_cfg.get(axis_config_name, "+x")
        ).strip().lower()
        if selected_axis not in ("+x", "-x", "+y", "-y"):
            print(
                f"[WARN] Invalid {orientation_config_name}.{axis_config_name} "
                f"{selected_axis!r}; using '+x'."
            )
            selected_axis = "+x"
        if selected_axis in ("+x", "-x"):
            x_axis = stem_axis if selected_axis == "+x" else -stem_axis
            y_axis = np.cross(z_axis, x_axis)
            y_axis /= float(np.linalg.norm(y_axis))
        else:
            y_axis = stem_axis if selected_axis == "+y" else -stem_axis
            x_axis = np.cross(y_axis, z_axis)
            x_axis /= float(np.linalg.norm(x_axis))
        return np.column_stack((x_axis, y_axis, z_axis))

    def _orient_gripper_drop_tcp(
        self,
        approach: np.ndarray,
        R_fallback: np.ndarray,
    ) -> np.ndarray:
        """Orient a drop pose with a selected TCP axis up and -Z toward base."""
        orientation_cfg = self.config.get(
            "motion", "gripper_drop_orientation", default={}
        ) or {}
        up_axis_name = str(
            orientation_cfg.get("tcp_axis_pointing_up", "+x")
        ).strip().lower()
        if up_axis_name not in ("+x", "-x", "+y", "-y"):
            print(
                f"[WARN] Invalid gripper-drop tcp_axis_pointing_up "
                f"{up_axis_name!r}; using '+x'."
            )
            up_axis_name = "+x"
        up = np.array([0.0, 0.0, 1.0], dtype=float)

        base_point = np.asarray(
            orientation_cfg.get(
                "tcp_negative_z_toward_base_point", [0.0, 0.0, 0.0]
            ),
            dtype=float,
        ).reshape(3,)
        negative_z = base_point - np.asarray(approach, dtype=float).reshape(3,)
        negative_z -= up * float(np.dot(negative_z, up))
        if float(np.linalg.norm(negative_z)) < 1e-8:
            negative_z = -np.asarray(R_fallback, dtype=float)[:3, 2].copy()
            negative_z -= up * float(np.dot(negative_z, up))
        if float(np.linalg.norm(negative_z)) < 1e-8:
            negative_z = np.array([-1.0, 0.0, 0.0], dtype=float)
            negative_z -= up * float(np.dot(negative_z, up))
        negative_z /= float(np.linalg.norm(negative_z))
        z_axis = -negative_z

        if up_axis_name in ("+x", "-x"):
            x_axis = up if up_axis_name == "+x" else -up
            y_axis = np.cross(z_axis, x_axis)
            y_axis /= float(np.linalg.norm(y_axis))
        else:
            y_axis = up if up_axis_name == "+y" else -up
            x_axis = np.cross(y_axis, z_axis)
            x_axis /= float(np.linalg.norm(x_axis))
        return np.column_stack((x_axis, y_axis, z_axis))

    def filter_reachable_candidates(
        self,
        candidates: List[Dict[str, Any]],
        ignore_close: bool = False,
    ) -> List[Dict[str, Any]]:
        """Check every pose and return candidates accepted by IK and motion constraints."""
        if not candidates:
            return []

        T_curr = self.handles.arm.get_T_base_tcp()
        curr_p = np.array(T_curr[:3, 3]).reshape(3,)

        max_z_offset = self.config.get('motion', 'constraints', 'max_z_offset', default=0.25)
        min_dist = self.config.get('motion', 'constraints', 'min_distance_threshold', default=0.05)

        accepted: List[Dict[str, Any]] = []
        print(f"[PLAN] Checking all {len(candidates)} candidate pose(s) for reachability.")
        for index, cand in enumerate(candidates):
            pose = cand["pose"]
            approach = np.array(
                cand.get("command_approach", cand["approach"])
            ).reshape(3,)

            reachable, analysis = self.handles.arm.check_pose_reachable(pose)
            cost = float(np.linalg.norm(approach - curr_p))
            z_offset = float(approach[2] - curr_p[2])
            status = {
                "candidate_index": index,
                "checked": True,
                "ik_reachable": bool(reachable),
                "accepted": False,
                "reason": None,
                "cost_m": cost,
                "z_offset_m": z_offset,
                "analysis": analysis,
            }
            cand["reachability"] = status

            if not reachable:
                reason = "IK/unreachable"
                if isinstance(analysis, dict):
                    reason = str(
                        analysis.get("reason")
                        or analysis.get("error")
                        or reason
                    )
                status["reason"] = reason
                print(
                    f"[PLAN] Candidate {index + 1}/{len(candidates)}: "
                    f"REJECTED ({reason})."
                )
                continue

            if ignore_close and cost < min_dist:
                status["reason"] = (
                    f"distance {cost:.4f} m is below minimum {min_dist:.4f} m"
                )
                print(
                    f"[PLAN] Candidate {index + 1}/{len(candidates)}: "
                    f"REJECTED ({status['reason']})."
                )
                continue
            if z_offset > max_z_offset:
                status["reason"] = (
                    f"upward offset {z_offset:.4f} m exceeds maximum "
                    f"{max_z_offset:.4f} m"
                )
                print(
                    f"[PLAN] Candidate {index + 1}/{len(candidates)}: "
                    f"REJECTED ({status['reason']})."
                )
                continue

            status["accepted"] = True
            status["reason"] = "reachable"
            accepted.append(cand)
            print(
                f"[PLAN] Candidate {index + 1}/{len(candidates)}: "
                f"ACCEPTED (distance={cost:.4f} m, z_offset={z_offset:.4f} m)."
            )

        print(
            f"[PLAN] Reachability filter complete: "
            f"{len(accepted)}/{len(candidates)} candidate pose(s) accepted."
        )
        return accepted

    def select_best_candidate(
        self,
        candidates: List[Dict[str, Any]],
        ignore_close: bool = False,
    ) -> Tuple[Optional[Dict[str, Any]], float]:
        """Filter every candidate, then return the nearest accepted pose."""
        accepted = self.filter_reachable_candidates(
            candidates,
            ignore_close=ignore_close,
        )
        if not accepted:
            return None, float("inf")

        best_candidate = min(
            accepted,
            key=lambda candidate: float(candidate["reachability"]["cost_m"]),
        )
        best_cost = float(best_candidate["reachability"]["cost_m"])
        print(
            f"[PLAN] Selected candidate "
            f"{int(best_candidate['reachability']['candidate_index']) + 1} "
            f"with distance={best_cost:.4f} m."
        )
        return best_candidate, best_cost

    def select_debug_candidate(
        self,
        candidates: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Return the nearest candidate for visualization, without reachability checks."""
        if not candidates:
            return None

        current_position = np.asarray(
            self.handles.arm.get_T_base_tcp()[:3, 3],
            dtype=float,
        ).reshape(3,)
        valid_candidates = []
        for candidate in candidates:
            approach = np.asarray(
                candidate.get("command_approach", candidate.get("approach", [])),
                dtype=float,
            ).reshape(-1)
            if approach.size >= 3:
                valid_candidates.append((float(np.linalg.norm(approach[:3] - current_position)), candidate))
        if not valid_candidates:
            return candidates[0]
        return min(valid_candidates, key=lambda item: item[0])[1]

    def compute_approach_poses(
        self,
        T_base_tcp,
        target_cams_info: List[Dict[str, Any]],
        T_cam: np.ndarray,
        camera_type: str,
        z_offset: float = None,
        x_offset: float = None,
        y_offset: float = None,
        ignore_rotation: bool = False,
        orientation_profile: str = "target",
        planning_tcp_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Compute approach poses for detected targets."""
        if z_offset is None:
            z_offset = self.config.offsets['initial_approach']
        if x_offset is None:
            x_offset = self.config.offsets.get('x', 0.0)
        if y_offset is None:
            y_offset = self.config.offsets.get('y', 0.0)

        if not target_cams_info:
            return []
        
        tool_name = str(getattr(self.handles, "tool_name", "") or "active_tcp")
        active_tcp_label = f"Active robot TCP ({tool_name})"
        planning_tool_name = str(planning_tcp_name or tool_name)
        planning_tcp_label = f"Planning TCP ({planning_tool_name})"
        T_active_planning = np.eye(4)
        if planning_tool_name != tool_name:
            tool_transforms = getattr(self.handles, "tool_tcp_transforms", {}) or {}
            try:
                T_flange_active = np.asarray(tool_transforms[tool_name], dtype=float)
                T_flange_planning = np.asarray(tool_transforms[planning_tool_name], dtype=float)
            except KeyError as e:
                raise ValueError(f"Missing TCP calibration for tool {e.args[0]!r}") from e
            T_active_planning = np.linalg.inv(T_flange_active) @ T_flange_planning

        T_base_planning = np.asarray(T_base_tcp, dtype=float) @ T_active_planning
        R_base_planning = T_base_planning[0:3, 0:3]
        is_tool_camera = camera_type != "fixed"
        if is_tool_camera:
            T_base_cam = T_base_tcp @ T_cam
            camera_tool_names = getattr(self.handles, "tool_camera_tool_names", {}) or {}
            camera_tool_name = str(
                camera_tool_names.get(camera_type)
                or getattr(self.handles, "camera_tool_name", None)
                or tool_name
            )
            if camera_tool_name == "tcp_gripper":
                camera_label = "Gripper camera (T_tcp_gripper_camera)"
            elif camera_tool_name == "tcp_drill":
                camera_label = "Drill camera (T_tcp_drill_camera)"
            else:
                camera_label = "Tool camera (T_tcp_camera)"
        else:  # fixed
            T_base_cam = T_cam
            camera_label = "Fixed camera (T_base_fixed_camera)"
        
        candidates = []
        for info in target_cams_info:
            target_cam = info["target_cam"]
            u, v = info["pixel"]
            image = info.get("image")
            prompt = info.get("prompt", "target")
            angle_deg_cam = info.get("angle_cam", None)

            p_cam_h = np.concatenate([np.asarray(target_cam).reshape(3,), [1.0]])
            p_base = (T_base_cam @ p_cam_h)[:3]
            
            approach = p_base + np.array([x_offset, y_offset, z_offset])
            
            T_planning_new = T_base_planning.copy()
            T_planning_new[0:3, 3] = approach


            # yaw_rad = math.radians(float(angle_deg_cam))
            # R_z = utils.rotz(yaw_rad)

            # R_target = R_base_tcp @ R_z
            # T_new[0:3, 0:3] = R_target
            if angle_deg_cam is not None and not ignore_rotation:
                yaw_rad = math.radians(float(angle_deg_cam))
                R_z = utils.rotz(yaw_rad)
                R_target = R_base_planning @ R_z
            else:
                # fallback—keep current rotation or use default
                R_target = R_base_planning

            if orientation_profile == "gripper_load":
                T_planning_new[0:3, 0:3] = self._orient_tcp_axis_to_stem_vertical_down(
                    info.get("stem_angle_cam"),
                    info.get("stem_direction_cam"),
                    T_base_cam,
                    R_target,
                )
            elif orientation_profile == "gripper_object_pickup":
                T_planning_new[0:3, 0:3] = self._orient_tcp_axis_to_stem_vertical_down(
                    info.get("angle_cam"),
                    None,
                    T_base_cam,
                    R_target,
                    orientation_config_name="gripper_object_pickup_orientation",
                    axis_config_name="tcp_axis_along_object",
                )
            elif orientation_profile == "gripper_drop":
                T_planning_new[0:3, 0:3] = self._orient_gripper_drop_tcp(
                    approach,
                    R_target,
                )
            else:
                T_planning_new[0:3, 0:3] = self._orient_tcp_y_to_base(
                    approach,
                    R_target,
                    orientation_profile=orientation_profile,
                )

            T_active_new = T_planning_new @ np.linalg.inv(T_active_planning)
            pose = utils.matrix_to_ur_pose(T_active_new)
            if is_tool_camera:
                T_base_cam_planned = T_active_new @ T_cam
            else:
                T_base_cam_planned = T_cam
            
            cand = {
                "prompt": prompt,
                "pose": pose,
                "approach": approach,
                "target_cam": target_cam,
                "target_base": p_base,
                "camera_type": camera_type,
                "active_tcp_label": active_tcp_label,
                "planning_tcp_label": planning_tcp_label,
                "has_separate_planning_tcp": planning_tool_name != tool_name,
                "camera_label": camera_label,
                "T_base_tcp": T_base_tcp.copy(),
                "T_base_tcp_planned": T_active_new.copy(),
                "T_base_planning_tcp": T_base_planning.copy(),
                "T_base_planning_tcp_planned": T_planning_new.copy(),
                "T_active_planning_tcp": T_active_planning.copy(),
                "T_base_camera": T_base_cam.copy(),
                "T_base_camera_planned": T_base_cam_planned.copy(),
                "camera_position_base": T_base_cam[:3, 3].copy(),
                "planned_camera_position_base": T_base_cam_planned[:3, 3].copy(),
                "angle_deg_cam": angle_deg_cam,
                "stem_angle_deg_cam": info.get("stem_angle_cam"),
                "orientation_label": (
                    f"{planning_tcp_label}: +Z down; "
                    f"{self._gripper_load_stem_axis().upper()} along stem"
                    if orientation_profile == "gripper_load"
                    else f"{planning_tcp_label}: +Z down; "
                    f"{str(self.config.get('motion', 'gripper_object_pickup_orientation', 'tcp_axis_along_object', default='+x')).upper()} along object"
                    if orientation_profile == "gripper_object_pickup"
                    else f"{planning_tcp_label}: "
                    f"{str(self.config.get('motion', 'gripper_drop_orientation', 'tcp_axis_pointing_up', default='+x')).upper()} up; "
                    "-Z toward robot base"
                    if orientation_profile == "gripper_drop"
                    else None
                ),
                "command_approach": T_active_new[:3, 3].copy(),
                "u": u,
                "v": v,
                "image": image,
                
            }
            candidates.append(cand)

        return candidates

    def servo_towards_dynamic(
        self,
        target_pose_fn: Callable[[], Optional[List[float]]],
        speed: float,
    ) -> str:
        """
        Visual servo with correct rotation handling (SO(3) relative rotation),
        and with a parameterized rotation step.

        Config parameters used:
          - motion.servo.step_size_m
          - motion.servo.pos_tolerance_m
          - motion.servo.rotation_step_rad
          - motion.servo.rotation_step_scale
          - motion.servo.rot_tolerance_rad
          - motion.servo.max_missing_detections
          - motion.servo.steady_tolerance_m
          - motion.servo.steady_tolerance_rot_rad
        """

        servo_cfg = self.config.servo
        servo_time   = servo_cfg["time"]
        min_step_m   =servo_cfg["min_step_m"]
        servo_speed   = servo_cfg["servo_speed"]
        pos_tol      = servo_cfg["pos_tolerance_m"]
        max_time     = servo_cfg["max_time_s"]

        # ---- NEW: rotation step parameters ----
        step_rad    = np.deg2rad(servo_cfg.get("rotation_step_deg", math.degrees(5)))  # 5°
        rot_tol     = np.deg2rad(servo_cfg.get("rot_tolerance_deg", math.degrees(1.0)))

        # ---- Steady state parameters ----
        steady_tolerance_pos = servo_cfg.get("steady_tolerance_m", 1e-3)
        steady_tolerance_rot = servo_cfg.get("steady_tolerance_rot_rad", math.radians(0.5))
        steady_iters         = servo_cfg.get("steady_iters", 5)
        max_missing          = servo_cfg.get("max_missing_detections", 3)
        min_speed_scale      = servo_cfg.get("min_speed_scale", 0.5)   # 10% of servo_speed near goal
        slow_dist_m          = servo_cfg.get("slow_dist_m", 0.1)      # start slowing within 5 cm

        last_pos_dist = None
        last_rot_dist = None
        steady_count  = 0
        missing_count = 0

        t0 = time.time()
        status = "unknown"
        while True:
            # ---------------------------------------------------
            # Acquire target pose
            # ---------------------------------------------------
            t_loop_start = time.time()
            T_curr = self.handles.arm.get_T_base_tcp()
            target_pose = target_pose_fn()
            if target_pose is None:
                missing_count += 1
                print(f"[SERVO] Missing detection {missing_count}/{max_missing}")
                if missing_count >= max_missing:
                    return "target_lost"
                time.sleep(servo_time)
                continue

            missing_count = 0  # reset misses

            # Ensure 6 DOF
            target_pose = np.array(target_pose, float).flatten()
            if target_pose.size < 6:
                # T_curr = self.handles.arm.get_T_base_tcp()
                curr_ur = np.array(utils.matrix_to_ur_pose(T_curr), float)
                target_pose = np.hstack([target_pose[:3], curr_ur[3:6]])

            t_p = target_pose[:3]
            t_r = target_pose[3:6]

            T_curr = self.handles.arm.get_T_base_tcp()
            curr_p = np.array(T_curr[:3, 3], float)
            pos_diff = t_p - curr_p
            pos_dist = float(np.linalg.norm(pos_diff))
            servo_time = pos_dist/servo_speed
            if pos_dist <= pos_tol:
                print("[SERVO] Target reached")
                status = "target_reached"
                break

            new_pose = np.hstack([t_p, t_r])
            self.handles.arm.servoL(
                pose=new_pose.tolist(),
                time_s=servo_time,
                speed=speed,
            )

            #when close wait for complete each cycle
            # if pos_dist < 0.02:
                # time.sleep(servo_time)
            # ---------------------------------------------------
            # Current pose
            # ---------------------------------------------------
            # T_curr = self.handles.arm.get_T_base_tcp()
            # curr_p = np.array(T_curr[:3, 3], float)
            # curr_ur = np.array(utils.matrix_to_ur_pose(T_curr), float)
            # curr_r = curr_ur[3:6]

            # # ---------------------------------------------------
            # # Position error
            # # ---------------------------------------------------
            # pos_diff = t_p - curr_p
            # pos_dist = float(np.linalg.norm(pos_diff))

            # # ---------------------------------------------------
            # # Rotation error using SO(3) relative rotation
            # # ---------------------------------------------------
            # R_curr, _   = cv2.Rodrigues(curr_r.reshape(3, 1))
            # R_target, _ = cv2.Rodrigues(t_r.reshape(3, 1))

            # R_err = R_target @ R_curr.T   # relative rotation
            # rot_vec, _ = cv2.Rodrigues(R_err)
            # rot_vec = rot_vec.flatten()
            # rot_dist = float(np.linalg.norm(rot_vec))

            # # ---------------------------------------------------
            # # Check for goal
            # # ---------------------------------------------------
            # if pos_dist <= pos_tol and rot_dist <= rot_tol:
            #     print("[SERVO] Target reached")
            #     status = "target_reached"
            #     break
            # # ---------------------------------------------------
            # # Steady-state test
            # # ---------------------------------------------------
            # if last_pos_dist is not None and last_rot_dist is not None:
            #     if (
            #         abs(last_pos_dist - pos_dist) < steady_tolerance_pos
            #         and abs(last_rot_dist - rot_dist) < steady_tolerance_rot
            #     ):
            #         steady_count += 1
            #     else:
            #         steady_count = 0

            # last_pos_dist = pos_dist
            # last_rot_dist = rot_dist

            # if steady_count >= steady_iters:
            #     print("[SERVO] Steady state")
            #     status = "steady_state"
            #     break

            # if time.time() - t0 > max_time:
            #     print("[SERVO] Timeout")
            #     status = "timeout"
            #     break
            # # ---------------------------------------------------
            # # Compute bounded position step
            # # ---------------------------------------------------
            # # set_speed =  step_m/loop_elapsed
            # # ---------------------------------------------------
            # # Compute bounded position step
            # # ---------------------------------------------------
            # # As we get closer to the target, reduce servo_speed
            # # from 1.0 * servo_speed (far) down to min_speed_scale * servo_speed (at the goal).
            # if slow_dist_m > 1e-6:
            #     if pos_dist >= slow_dist_m:
            #         speed_scale = 1.0
            #     else:
            #         # Linear ramp: pos_dist == 0  -> min_speed_scale
            #         #              pos_dist == slow_dist_m -> 1.0
            #         alpha = pos_dist / slow_dist_m
            #         speed_scale = min_speed_scale + (1.0 - min_speed_scale) * alpha
            # else:
            #     speed_scale = 1.0

            # current_servo_speed = servo_speed * speed_scale

            # step_m = current_servo_speed * servo_time
            # step_m = max(step_m, min_step_m)

            # if pos_dist > 1e-9:
            #     pos_step = (pos_diff / pos_dist) * min(step_m, pos_dist)
            #     new_p = curr_p + pos_step
            # else:
            #     new_p = curr_p.copy()


            # # step_m = servo_speed * servo_time
            # # step_m = max(step_m, min_step_m)
            # # if pos_dist > 1e-9:
            # #     pos_step = (pos_diff / pos_dist) * min(step_m, pos_dist)
            # #     new_p = curr_p + pos_step
            # # else:
            # #     new_p = curr_p.copy()

            # # ---------------------------------------------------
            # # Compute bounded rotation step (parameterized)
            # # ---------------------------------------------------
            # if rot_dist > 1e-9:
            #     # rot_step_vec = (rot_vec / rot_dist) * min(step_rad, rot_dist)
            #     # R_step, _ = cv2.Rodrigues(rot_step_vec.reshape(3,1))
            #     # R_new = R_step @ R_curr
            #     # new_rvec, _ = cv2.Rodrigues(R_new)
            #     # new_r = new_rvec.flatten()
            #     # --- Try both directions for rotation axis ---
            #     rot_step_mag = min(step_rad, rot_dist)

            #     # Shortest-path axis (Rodrigues result)
            #     axis_short = rot_vec / rot_dist
            #     step_short = axis_short * rot_step_mag

            #     # Opposite direction axis
            #     step_long = -axis_short * rot_step_mag

            #     # Build candidate rotations
            #     R_step_short, _ = cv2.Rodrigues(step_short.reshape(3,1))
            #     R_step_long,  _ = cv2.Rodrigues(step_long.reshape(3,1))

            #     R_new_short = R_step_short @ R_curr
            #     R_new_long  = R_step_long @ R_curr

            #     # Convert both to rvec
            #     r_short, _ = cv2.Rodrigues(R_new_short)
            #     r_long, _  = cv2.Rodrigues(R_new_long)

            #     # Choose the rotation that reduces angular error the most
            #     err_short = np.linalg.norm(r_short.flatten() - t_r)
            #     err_long  = np.linalg.norm(r_long.flatten() - t_r)

            #     if err_short <= err_long:
            #         new_r = r_short.flatten()
            #     else:
            #         new_r = r_long.flatten()

            # else:
            #     new_r = curr_r.copy()

            # # ---------------------------------------------------
            # # Apply motion
            # # ---------------------------------------------------
            # new_pose = np.hstack([new_p, new_r])

            # # self.handles.arm.servoL(
            # #     pose=new_pose.tolist(),
            # #     time_s=servo_time,
            # #     speed=speed,
            # # )
            # self.handles.arm.servoL(
            #     pose=new_pose.tolist(),
            #     time_s=servo_time,
            #     speed=speed,
            # )

            # servo_time = time.time() - t_loop_start
            # time.sleep(1.0)
    
        #print wait till done
        time.sleep(2.0)
        return status

    def move_to_best_reachable(
        self,
        candidates: List[Dict[str, Any]],
        speed: float = None,
        ignore_close: bool = False,
        move: bool = True,
        debug: bool = False,
        selected_candidate: Optional[Dict[str, Any]] = None,
    ) -> bool:
        ...
        if speed is None:
            speed = self.config.default_speed
        
        if not candidates:
            return False
        
        if selected_candidate is None:
            best_candidate, best_cost = self.select_best_candidate(
                candidates,
                ignore_close=ignore_close,
            )
        else:
            best_candidate = selected_candidate
            reachability = best_candidate.get("reachability", {}) or {}
            best_cost = float(reachability.get("cost_m", float("nan")))
        if best_candidate is None:
            print("[WARN] No reachable poses found")
            return False
        
        if debug and best_candidate.get("image") is not None:
            utils.draw_point(
                best_candidate["image"],
                best_candidate["u"],
                best_candidate["v"],
                text=best_candidate["prompt"],
            )

        if not move:
            print("[INFO] move=False, not moving the arm")
            return True
        
        candidate_index = (best_candidate.get("reachability", {}) or {}).get(
            "candidate_index"
        )
        candidate_label = (
            f"candidate {int(candidate_index) + 1}, "
            if candidate_index is not None
            else ""
        )
        print(
            f"[INFO] Moving to {candidate_label}pose, distance={best_cost:.4f}m: "
            f"{best_candidate['pose']}"
        )

        self.handles.arm.moveL(best_candidate["pose"], speed=speed)

        return True
    
class GenericDetector:
    def __init__(self, detector: Sam3Detector, config: Config, debug:bool=False):
        self.detector = detector
        self.config = config
        self.debug = debug
    
    def detect_generic(
            self,
            prompt: str,
            color: np.ndarray,
            conf: float = 0.2,
            orientation_align: str = 'long'

        ) -> List[Dict[str, Any]]:
            if self.debug:     
                cv2.imshow("SAM3 Raw", color)
                cv2.waitKey(0)
                
            masks = self.detector.segment(
                color_bgr=color,
                text_prompt=prompt,
                confidence_threshold=conf,
                category=prompt,
                orientation_align=orientation_align,
            )
            
            if self.debug and masks:
                output_dir = self.config.get('debug', 'output_dir', default='data/image_samples')
                output_file = self.config.get('debug', 'images', 'generic_detect', default='generic_detect.png')
                colors = self.config.get('debug', 'colors', default={})
    
                vis = draw_mask_debug(
                    color, masks,
                    output_path=f"{output_dir}/{output_file}",
                    category_colors={prompt: tuple(colors.get(prompt, [0, 0, 255]))}
                )
                cv2.imshow("SAM3 Debug Visualization", vis)
                cv2.waitKey(0)

            return masks
    
class ScrewDetector:
    """Handles screw detection logic."""
    
    def __init__(self, detector: Sam3Detector, config: Config):
        self.detector = detector
        self.config = config
        self._last_debug_overview: Optional[Dict[str, np.ndarray]] = None

    @staticmethod
    def _format_pose(pose: Any) -> str:
        values = np.asarray(pose, dtype=float).reshape(-1)
        if values.size < 3:
            return "unavailable"
        position = ", ".join(f"{value:.3f}" for value in values[:3])
        if values.size < 6:
            return f"[{position}] m"
        rotation = ", ".join(f"{value:.3f}" for value in values[3:6])
        return f"[{position}] m  [{rotation}] rad"

    @staticmethod
    def _nice_scale_length(span_m: float) -> float:
        """Choose a readable metric scale-bar length for the reach-plan panel."""
        target = max(float(span_m) / 4.0, 1e-4)
        exponent = math.floor(math.log10(target))
        normalized = target / (10 ** exponent)
        if normalized <= 1.0:
            nice = 1.0
        elif normalized <= 2.0:
            nice = 2.0
        elif normalized <= 5.0:
            nice = 5.0
        else:
            nice = 10.0
        return nice * (10 ** exponent)

    def _make_reach_plan_panel(
        self,
        *,
        width: int,
        height: int,
        current_tcp_pose: Any,
        candidate: Dict[str, Any],
    ) -> np.ndarray:
        """Draw the current and requested TCP locations in the robot base XY plane."""
        panel = np.full((height, width, 3), 24, dtype=np.uint8)
        cv2.rectangle(panel, (0, 0), (width - 1, height - 1), (70, 70, 70), 2, cv2.LINE_AA)
        cv2.rectangle(panel, (0, 0), (width - 1, 42), (38, 38, 38), -1)
        cv2.putText(panel, "Robot Base Reach Plan (XY)", (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (240, 240, 240), 2, cv2.LINE_AA)

        current = np.asarray(current_tcp_pose, dtype=float).reshape(-1)
        planned = np.asarray(candidate.get("pose", []), dtype=float).reshape(-1)
        target = np.asarray(candidate.get("target_base", []), dtype=float).reshape(-1)
        approach = np.asarray(candidate.get("approach", []), dtype=float).reshape(-1)
        camera_now = np.asarray(candidate.get("camera_position_base", []), dtype=float).reshape(-1)
        camera_planned = np.asarray(candidate.get("planned_camera_position_base", []), dtype=float).reshape(-1)
        active_tcp_label = str(candidate.get("active_tcp_label", "Active robot TCP"))
        planning_tcp_label = str(candidate.get("planning_tcp_label", active_tcp_label))
        camera_label = str(candidate.get("camera_label", "Camera"))
        orientation_label = candidate.get("orientation_label")
        T_planning_now = np.asarray(candidate.get("T_base_planning_tcp", []), dtype=float)
        T_planning_planned = np.asarray(
            candidate.get("T_base_planning_tcp_planned", []), dtype=float
        )
        has_separate_planning_tcp = (
            bool(candidate.get("has_separate_planning_tcp", False))
            and
            T_planning_now.shape == (4, 4)
            and T_planning_planned.shape == (4, 4)
        )
        planning_now = T_planning_now[:3, 3] if has_separate_planning_tcp else np.array([])
        planning_planned = (
            T_planning_planned[:3, 3] if has_separate_planning_tcp else np.array([])
        )
        if current.size < 3 or planned.size < 3 or target.size < 3 or approach.size < 3:
            cv2.putText(panel, "Reach-plan pose data unavailable", (18, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 2, cv2.LINE_AA)
            return panel

        # All markers are positions in the robot base frame, so the scale is directly comparable.
        point_rows = [[0.0, 0.0], current[:2], planned[:2], target[:2], approach[:2]]
        if has_separate_planning_tcp:
            point_rows.extend((planning_now[:2], planning_planned[:2]))
        if camera_now.size >= 3:
            point_rows.append(camera_now[:2])
        if camera_planned.size >= 3:
            point_rows.append(camera_planned[:2])
        points = np.asarray(point_rows, dtype=float)
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        span = max(float(np.max(maxs - mins)), 0.10)
        padding = max(0.05, span * 0.15)
        mins -= padding
        maxs += padding
        span_x = max(float(maxs[0] - mins[0]), 1e-6)
        span_y = max(float(maxs[1] - mins[1]), 1e-6)

        plot_left, plot_top = 54, 58
        plot_right, plot_bottom = width - 18, 235
        plot_w, plot_h = plot_right - plot_left, plot_bottom - plot_top
        scale_px_per_m = min(plot_w / span_x, plot_h / span_y)
        offset_x = plot_left + (plot_w - span_x * scale_px_per_m) / 2.0
        offset_y = plot_top + (plot_h - span_y * scale_px_per_m) / 2.0

        def project(point: np.ndarray) -> Tuple[int, int]:
            x = int(round(offset_x + (float(point[0]) - mins[0]) * scale_px_per_m))
            y = int(round(offset_y + (maxs[1] - float(point[1])) * scale_px_per_m))
            return x, y

        cv2.rectangle(panel, (plot_left, plot_top), (plot_right, plot_bottom), (55, 55, 55), 1, cv2.LINE_AA)
        base_xy = project(np.array([0.0, 0.0]))
        tcp_xy = project(current[:2])
        tcp_planned_xy = project(planned[:2])
        target_xy = project(target[:2])
        approach_xy = project(approach[:2])
        cv2.arrowedLine(panel, tcp_xy, tcp_planned_xy, (0, 220, 255), 2, cv2.LINE_AA, tipLength=0.04)
        cv2.drawMarker(panel, base_xy, (255, 120, 0), cv2.MARKER_SQUARE, 18, 2, cv2.LINE_AA)
        cv2.circle(panel, tcp_xy, 6, (70, 70, 255), -1, cv2.LINE_AA)
        cv2.drawMarker(panel, target_xy, (0, 255, 255), cv2.MARKER_CROSS, 17, 2, cv2.LINE_AA)
        cv2.drawMarker(panel, tcp_planned_xy, (0, 220, 0), cv2.MARKER_TILTED_CROSS, 18, 2, cv2.LINE_AA)
        for label, point, color in (
            ("Base", base_xy, (255, 120, 0)),
            (active_tcp_label, tcp_xy, (70, 70, 255)),
            ("Target", target_xy, (0, 255, 255)),
            (f"Planned {active_tcp_label}", tcp_planned_xy, (0, 220, 0)),
        ):
            cv2.putText(panel, label, (point[0] + 8, point[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
        if has_separate_planning_tcp:
            planning_now_xy = project(planning_now[:2])
            planning_planned_xy = project(planning_planned[:2])
            cv2.circle(panel, planning_now_xy, 5, (255, 140, 40), 2, cv2.LINE_AA)
            cv2.drawMarker(panel, planning_planned_xy, (200, 80, 255), cv2.MARKER_STAR, 17, 2, cv2.LINE_AA)
            cv2.putText(panel, f"{planning_tcp_label} now", (planning_now_xy[0] + 8, planning_now_xy[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 140, 40), 1, cv2.LINE_AA)
            cv2.putText(panel, f"{planning_tcp_label} planned", (planning_planned_xy[0] + 8, planning_planned_xy[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 80, 255), 1, cv2.LINE_AA)
        if camera_now.size >= 3:
            camera_now_xy = project(camera_now[:2])
            cv2.drawMarker(panel, camera_now_xy, (255, 0, 255), cv2.MARKER_DIAMOND, 16, 2, cv2.LINE_AA)
            cv2.putText(panel, f"{camera_label} now", (camera_now_xy[0] + 8, camera_now_xy[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 0, 255), 1, cv2.LINE_AA)
        if camera_planned.size >= 3:
            camera_planned_xy = project(camera_planned[:2])
            same_camera_position = camera_now.size >= 3 and np.allclose(camera_now[:3], camera_planned[:3])
            if not same_camera_position:
                cv2.drawMarker(panel, camera_planned_xy, (255, 255, 0), cv2.MARKER_STAR, 16, 2, cv2.LINE_AA)
                cv2.putText(panel, f"{camera_label} planned", (camera_planned_xy[0] + 8, camera_planned_xy[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1, cv2.LINE_AA)

        scale_m = self._nice_scale_length(span)
        scale_px = int(round(scale_m * scale_px_per_m))
        scale_x, scale_y = plot_left + 12, plot_bottom - 14
        cv2.line(panel, (scale_x, scale_y), (scale_x + scale_px, scale_y), (235, 235, 235), 2, cv2.LINE_AA)
        cv2.line(panel, (scale_x, scale_y - 4), (scale_x, scale_y + 4), (235, 235, 235), 2, cv2.LINE_AA)
        cv2.line(panel, (scale_x + scale_px, scale_y - 4), (scale_x + scale_px, scale_y + 4), (235, 235, 235), 2, cv2.LINE_AA)
        cv2.putText(panel, f"{scale_m:.3g} m", (scale_x, scale_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (235, 235, 235), 1, cv2.LINE_AA)

        delta_m = float(np.linalg.norm(planned[:3] - current[:3]))
        text_lines = (
            f"Base:        [0.000, 0.000, 0.000] m",
            f"{active_tcp_label} now: {self._format_pose(current)}",
            f"Screw target:{self._format_pose(target)}",
            f"Planned {active_tcp_label}: {self._format_pose(planned)}",
            f"{active_tcp_label} -> planned: {delta_m:.3f} m",
        )
        camera_lines = []
        if has_separate_planning_tcp:
            camera_lines.append(
                f"Planned {planning_tcp_label}: "
                f"{self._format_pose(utils.matrix_to_ur_pose(T_planning_planned))}"
            )
        if orientation_label:
            camera_lines.append(f"Orientation: {orientation_label}")
        if camera_now.size >= 3:
            camera_lines.append(f"{camera_label} now: {self._format_pose(camera_now)}")
        if camera_planned.size >= 3 and not np.allclose(camera_now[:3], camera_planned[:3]):
            camera_lines.append(f"{camera_label} planned: {self._format_pose(camera_planned)}")
        text_lines += tuple(camera_lines)
        y = 255
        for line in text_lines:
            cv2.putText(panel, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (230, 230, 230), 1, cv2.LINE_AA)
            y += 23
        return panel

    @staticmethod
    def _make_reach_plan_pending_panel(width: int, height: int) -> np.ndarray:
        """Reserve the fourth dashboard panel while target-to-pose planning runs."""
        panel = np.full((height, width, 3), 24, dtype=np.uint8)
        cv2.rectangle(panel, (0, 0), (width - 1, height - 1), (70, 70, 70), 2, cv2.LINE_AA)
        cv2.rectangle(panel, (0, 0), (width - 1, 42), (38, 38, 38), -1)
        cv2.putText(panel, "Robot Base Reach Plan (XY)", (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (240, 240, 240), 2, cv2.LINE_AA)
        cv2.putText(panel, "Planning target pose...", (18, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (190, 190, 190), 2, cv2.LINE_AA)
        return panel

    def show_reach_plan_3d_debug(self, candidate: Optional[Dict[str, Any]]) -> None:
        """Show base, TCP, camera, and planned-pose coordinate frames in a separate 3D GUI."""
        if candidate is None:
            return
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("[WARN] matplotlib is unavailable; cannot display 3D reach-plan debug view.")
            return

        try:
            T_tcp_now = np.asarray(candidate["T_base_tcp"], dtype=float).reshape(4, 4)
            T_tcp_planned = np.asarray(candidate["T_base_tcp_planned"], dtype=float).reshape(4, 4)
            T_camera_now = np.asarray(candidate["T_base_camera"], dtype=float).reshape(4, 4)
            T_camera_planned = np.asarray(candidate["T_base_camera_planned"], dtype=float).reshape(4, 4)
            target = np.asarray(candidate["target_base"], dtype=float).reshape(3,)
        except (KeyError, TypeError, ValueError) as e:
            print(f"[WARN] Incomplete reach-plan data for 3D debug view: {e}")
            return

        active_tcp_label = str(candidate.get("active_tcp_label", "Active robot TCP"))
        planning_tcp_label = str(candidate.get("planning_tcp_label", active_tcp_label))
        camera_label = str(candidate.get("camera_label", "Camera"))
        orientation_label = candidate.get("orientation_label")
        T_planning_now = np.asarray(candidate.get("T_base_planning_tcp", []), dtype=float)
        T_planning_planned = np.asarray(
            candidate.get("T_base_planning_tcp_planned", []), dtype=float
        )
        has_separate_planning_tcp = (
            bool(candidate.get("has_separate_planning_tcp", False))
            and
            T_planning_now.shape == (4, 4)
            and T_planning_planned.shape == (4, 4)
        )

        origin_rows = [
            np.zeros(3),
            T_tcp_now[:3, 3],
            T_tcp_planned[:3, 3],
            T_camera_now[:3, 3],
            T_camera_planned[:3, 3],
            target,
        ]
        if has_separate_planning_tcp:
            origin_rows.extend((T_planning_now[:3, 3], T_planning_planned[:3, 3]))
        origins = np.vstack(origin_rows)
        mins = origins.min(axis=0)
        maxs = origins.max(axis=0)
        span = max(float(np.max(maxs - mins)), 0.10)
        margin = max(0.05, span * 0.15)
        axis_length = max(0.025, span * 0.10)

        fig = plt.figure("Robot Frames Debug", figsize=(11, 8))
        ax = fig.add_subplot(111, projection="3d")
        title = "Robot Base Reach Plan - Coordinate Frames"
        if orientation_label:
            title = f"{title}\n{orientation_label}"
        ax.set_title(title)

        def draw_frame(transform: np.ndarray, label: str, color: str) -> None:
            origin = transform[:3, 3]
            axes = transform[:3, :3]
            for vector, axis_color in zip(axes.T, ("r", "g", "b")):
                ax.quiver(*origin, *vector, length=axis_length, color=axis_color, arrow_length_ratio=0.16)
            ax.scatter(*origin, color=color, s=38)
            ax.text(*origin, f" {label}", color=color)

        draw_frame(np.eye(4), "Base", "black")
        draw_frame(T_tcp_now, f"{active_tcp_label} now", "tab:red")
        draw_frame(T_tcp_planned, f"{active_tcp_label} planned", "tab:green")
        if has_separate_planning_tcp:
            draw_frame(T_planning_now, f"{planning_tcp_label} now", "tab:blue")
            draw_frame(T_planning_planned, f"{planning_tcp_label} planned", "tab:pink")
        draw_frame(T_camera_now, f"{camera_label} now", "tab:purple")
        if not np.allclose(T_camera_now, T_camera_planned):
            draw_frame(T_camera_planned, f"{camera_label} planned", "tab:orange")
        target_label = str(candidate.get("prompt", "Target"))
        ax.scatter(*target, color="gold", marker="x", s=85, linewidths=2, label=target_label)
        ax.plot(
            [T_tcp_now[0, 3], T_tcp_planned[0, 3]],
            [T_tcp_now[1, 3], T_tcp_planned[1, 3]],
            [T_tcp_now[2, 3], T_tcp_planned[2, 3]],
            color="tab:cyan",
            linestyle="--",
            label=f"{active_tcp_label} motion",
        )
        if has_separate_planning_tcp:
            ax.plot(
                [T_planning_now[0, 3], T_planning_planned[0, 3]],
                [T_planning_now[1, 3], T_planning_planned[1, 3]],
                [T_planning_now[2, 3], T_planning_planned[2, 3]],
                color="tab:blue",
                linestyle=":",
                label=f"{planning_tcp_label} motion",
            )
        # Use one shared data range so one metre is the same visual distance on every axis.
        plot_center = (mins + maxs) / 2.0
        plot_half_range = max(float(np.max(maxs - mins)) / 2.0 + margin, 0.05)
        ax.set_xlim(plot_center[0] - plot_half_range, plot_center[0] + plot_half_range)
        ax.set_ylim(plot_center[1] - plot_half_range, plot_center[1] + plot_half_range)
        ax.set_zlim(plot_center[2] - plot_half_range, plot_center[2] + plot_half_range)
        try:
            ax.set_box_aspect((1, 1, 1))
        except AttributeError:
            pass
        ax.set_xlabel("Base X (m)")
        ax.set_ylabel("Base Y (m)")
        ax.set_zlabel("Base Z (m)")
        ax.legend(loc="upper left")
        plt.tight_layout()
        print("[DEBUG] Showing 3D robot-frame view. Close it to continue.")
        plt.show(block=True)


    def detect_screws(
        self,
        color: np.ndarray,
        camera_type: Optional[str] = None,
        heads_conf: float = None,
        stems_conf: float = None,
        box_conf: float = None,
        detection_mode: Optional[str] = None,
        debug: bool = False
    ) -> List[Dict[str, Any]]:
        """Detect screw heads according to the configured support mode."""
        conf_map = self.config.confidence_for_camera(camera_type)
        if heads_conf is None:
            heads_conf = conf_map.get('screw_head', 0.2)
        if stems_conf is None:
            stems_conf = conf_map.get('screw_stem', 0.2)
        if box_conf is None:
            box_conf = conf_map.get('screw_box', 0.2)

        prompts = self.config.prompts
        resolved_mode = self._normalize_screw_detection_mode(detection_mode)

        heads = self.detector.segment(
            color_bgr=color,
            text_prompt=prompts['screw_head'],
            confidence_threshold=heads_conf,
            category=prompts['screw_head'],
        )
        
        stems = self.detector.segment(
            color_bgr=color,
            text_prompt=prompts['screw_stem'],
            confidence_threshold=stems_conf,
            category=prompts['screw_stem'],
        )
        
        boxes = self.detector.segment(
            color_bgr=color,
            text_prompt=prompts['screw_box'],
            confidence_threshold=box_conf,
            category=prompts['screw_box'],
        )

        min_area = self.config.get('filtering', 'stem', 'min_area_px', default=500)
        stems = self.filter_by_area(stems, min_area)

        overlap_result = self.get_overlapping_heads(
            heads,
            stems,
            boxes,
            mode=resolved_mode,
            return_combined_contour_maps=True,
        )
        if isinstance(overlap_result, dict):
            self._attach_head_target_hints(
                overlap_result.get("overlap_heads", []) or [],
                overlap_result.get("overlap_stems", []) or [],
                overlap_result.get("overlap_boxes", []) or [],
            )
            overlap_result["mode"] = resolved_mode

        if debug and isinstance(overlap_result, dict):
            self.debug_visualize(
                color,
                heads,
                stems,
                boxes,
                overlap_result["overlap_heads"],
                overlap_contour_maps=overlap_result.get("combined_contour_maps", []),
            )

        return overlap_result
    

    def build_contour_overlay(
        self,
        color: np.ndarray,
        screws,
        contour_type: str = "overlap",
        debug: bool = False,
        *,
        head_label_prefix: Optional[str] = None,
        label_support_contours: bool = True,
        include_scores_in_labels: bool = True,
        draw_head_outlines: bool = True,
        draw_support_outlines: bool = True,
        label_color_bgr: Tuple[int, int, int] = (255, 255, 255),
    ) -> bytes:
        """
        Build a contour-only visualization for the LLM selector.

        Accepts either:
          - enriched detect_screws() dict (with overlap_heads + combined_contour_maps), or
          - a plain list of masks (treated as the labeled regions).

        Returns:
          JPEG bytes of the rendered contour image (blank background + contours/labels).
        """
        vis = color.copy()
        h, w = vis.shape[:2]

        if isinstance(screws, dict):
            overlap_heads = screws.get("overlap_heads", []) or []
            support_contours = screws.get("combined_contour_maps", []) or []
        else:
            overlap_heads = screws or []
            support_contours = []

        def _bbox_from_mask(m: Dict[str, Any], contour) -> Tuple[int, int, int, int]:
            if contour is not None and len(contour) > 0:
                x, y, bw, bh = cv2.boundingRect(contour)
                return int(x), int(y), int(bw), int(bh)
            x, y, bw, bh = [int(v) for v in m.get("bbox", [0, 0, 0, 0])]
            return int(x), int(y), int(bw), int(bh)

        def _format_label_with_score(prefix_label: str, mask: Dict[str, Any]) -> str:
            if not include_scores_in_labels:
                return prefix_label
            score = mask.get("score", None)
            if score is None:
                return prefix_label
            try:
                return f"{prefix_label} {float(score):.2f}"
            except Exception:
                return prefix_label

        def _draw_label(
            text: str,
            bbox_xywh: Tuple[int, int, int, int],
            color_bgr: Tuple[int, int, int],
            *,
            inside: bool = False,
            contour=None,
        ) -> None:
            text_alpha = 0.5
            x, y, bw, bh = bbox_xywh
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            pad = 1
            badge_w = tw + pad * 2
            badge_h = th + pad * 2
            margin = 1

            def _clamp_rect(x1: int, y1: int) -> Tuple[int, int, int, int]:
                x1 = int(np.clip(x1, 0, max(w - badge_w, 0)))
                y1 = int(np.clip(y1, 0, max(h - badge_h, 0)))
                return x1, y1, x1 + badge_w, y1 + badge_h

            def _intersects_bbox(rect: Tuple[int, int, int, int]) -> bool:
                rx1, ry1, rx2, ry2 = rect
                bx1, by1 = x, y
                bx2, by2 = x + max(bw, 1), y + max(bh, 1)
                return not (rx2 <= bx1 or rx1 >= bx2 or ry2 <= by1 or ry1 >= by2)

            if inside:
                rect = None
                if contour is not None and len(contour) > 0:
                    try:
                        cx0, cy0, cw0, ch0 = cv2.boundingRect(contour)
                        if cw0 > 0 and ch0 > 0:
                            local = np.zeros((ch0, cw0), dtype=np.uint8)
                            shifted = contour.copy()
                            shifted[:, 0, 0] = shifted[:, 0, 0] - cx0
                            shifted[:, 0, 1] = shifted[:, 0, 1] - cy0
                            cv2.drawContours(local, [shifted], -1, 255, thickness=-1)
                            dist = cv2.distanceTransform(local, cv2.DIST_L2, 5)
                            _, _, _, max_loc = cv2.minMaxLoc(dist)
                            center_x = int(cx0 + max_loc[0])
                            center_y = int(cy0 + max_loc[1])
                            x1 = center_x - badge_w // 2
                            y1 = center_y - badge_h // 2
                            rect = _clamp_rect(x1, y1)
                    except Exception:
                        rect = None

                if rect is None:
                    # Fallback: keep label inside the bbox near the center.
                    x1 = x + max((bw - badge_w) // 2, 0)
                    y1 = y + max((bh - badge_h) // 2, 0)
                    rect = _clamp_rect(x1, y1)

                x1, y1, x2, y2 = rect
                text_org = (x1 + pad, y2 - pad)
                text_layer = vis.copy()
                cv2.putText(
                    text_layer,
                    text,
                    text_org,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 0),
                    3,
                    lineType=cv2.LINE_AA,
                )
                cv2.putText(
                    text_layer,
                    text,
                    text_org,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    label_color_bgr,
                    2,
                    lineType=cv2.LINE_AA,
                )
                cv2.addWeighted(text_layer, text_alpha, vis, 1.0 - text_alpha, 0.0, dst=vis)
                return

            # Prefer side placement so the label stays visually next to the contour.
            candidates = [
                _clamp_rect(x + bw + margin, y),                      # right
                _clamp_rect(x - badge_w - margin, y),                 # left
                _clamp_rect(x + bw + margin, y + bh - badge_h),       # right-bottom
                _clamp_rect(x - badge_w - margin, y + bh - badge_h),  # left-bottom
                _clamp_rect(x, y - badge_h - margin),                 # above-left
                _clamp_rect(x + bw - badge_w, y - badge_h - margin),  # above-right
                _clamp_rect(x, y + bh + margin),                      # below-left
                _clamp_rect(x + bw - badge_w, y + bh + margin),       # below-right
            ]

            rect = candidates[0]
            for cand in candidates:
                if not _intersects_bbox(cand):
                    rect = cand
                    break

            x1, y1, x2, y2 = rect
            text_org = (x1 + pad, y2 - pad)
            # No filled badge background; draw outlined text for readability.
            text_layer = vis.copy()
            cv2.putText(
                text_layer,
                text,
                text_org,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                3,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                text_layer,
                text,
                text_org,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                label_color_bgr,
                2,
                lineType=cv2.LINE_AA,
            )
            cv2.addWeighted(text_layer, text_alpha, vis, 1.0 - text_alpha, 0.0, dst=vis)

        def _draw_mask_outline(
            mask: Dict[str, Any],
            color_bgr: Tuple[int, int, int],
            *,
            thickness: int = 2,
        ) -> None:
            contour = mask.get("contour", None)
            if contour is not None and len(contour) > 0:
                cv2.drawContours(vis, [contour], -1, (0, 0, 0), thickness + 2)
                cv2.drawContours(vis, [contour], -1, color_bgr, thickness)
                return

            x, y, bw, bh = [int(v) for v in mask.get("bbox", [0, 0, 0, 0])]
            cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 0, 0), thickness + 1)
            cv2.rectangle(vis, (x, y), (x + bw, y + bh), color_bgr, thickness)

        # Draw supporting contour maps first (stems/boxes), no mask fill.
        if contour_type == "overlap" or contour_type == "both":
            for idx, m in enumerate(support_contours):
                cat = str(m.get("category", "")).lower()
                if "box" in cat:
                    color_bgr = (255, 0, 0)
                elif "stem" in cat:
                    color_bgr = (0, 165, 255)
                else:
                    color_bgr = (200, 200, 200)
                label = _format_label_with_score(f"C{idx}", m)

                if draw_support_outlines:
                    _draw_mask_outline(m, color_bgr, thickness=2)

                contour = m.get("contour", None)
                bbox_xywh = _bbox_from_mask(m, contour)
                if label_support_contours:
                    _draw_label(label, bbox_xywh, color_bgr, inside=True, contour=contour)

        # Always draw screw-head contours as context so the operator and LLM can see the actual heads.
        if contour_type == "heads_only" or contour_type == "overlap" or contour_type == "both":
            for idx, m in enumerate(overlap_heads):
                contour = m.get("contour", None)
                color_bgr = (0, 255, 0)
                if draw_head_outlines:
                    _draw_mask_outline(m, color_bgr, thickness=2)

                if head_label_prefix is not None:
                    label = _format_label_with_score(f"{head_label_prefix}{idx}", m)
                    bbox_xywh = _bbox_from_mask(m, contour)
                    _draw_label(label, bbox_xywh, color_bgr)
                elif contour_type == "heads_only":
                    label = _format_label_with_score(f"R{idx}", m)
                    bbox_xywh = _bbox_from_mask(m, contour)
                    _draw_label(label, bbox_xywh, color_bgr)

        if contour_type not in ("overlap", "heads_only", "both"):
            raise ValueError(f"Unknown contour_type: {contour_type}")


        ok, buf = cv2.imencode(".jpg", vis)
        if not ok:
            raise RuntimeError("Failed to encode contour overlay image.")
        return buf.tobytes()
    
    def detect_target_marks(
        self,
        color: np.ndarray,
        camera_type: Optional[str] = None,
        heads_conf: float = None,
        target_marks_conf: float = None,
        debug: bool = False
    ) -> List[Dict[str, Any]]:
        """Detect cross symbol markers."""

        conf_map = self.config.confidence_for_camera(camera_type)
        if target_marks_conf is None:
             target_marks_conf = conf_map.get('target_marker', 0.2)
        
        target_marks = self.detector.segment(
            color_bgr=color,
            text_prompt=self.config.prompts['target_marker'],
            confidence_threshold=target_marks_conf,
            category=self.config.prompts['target_marker'],
        )
        raw_target_count = len(target_marks)
        target_marks = self._dedupe_masks_by_iou(
            target_marks,
            iou_threshold=float(
                self.config.get(
                    'filtering', 'target_marker', 'dedupe_iou', default=0.5
                )
            ),
        )
        if raw_target_count != len(target_marks):
            print(
                f"[INFO] Collapsed {raw_target_count} overlapping target-mark masks "
                f"to {len(target_marks)} unique candidate(s)."
            )
    
        # if heads_conf is None:
        #     heads_conf = conf_map.get('screw_head', 0.2)
        
        # heads = self.detector.segment(
        #     color_bgr=color,
        #     text_prompt=self.config.prompts['screw_head'],
        #     confidence_threshold=heads_conf,
        #     category=self.config.prompts['screw_head'],
        # )
        
        # filtered_target_marks = filter_masks_by_non_overlap(target_marks,heads,filter_side='sources')
       
        prompt = self.config.prompts['target_marker']
        
        if debug and target_marks:
            output_dir = self.config.get('debug', 'output_dir', default='data/image_samples')
            output_file = self.config.get('debug', 'images', 'target_marks', default='sam3_target_marks_and_screw_head.png')
            colors = self.config.get('debug', 'colors', default={})
            
            color_map = {
                self.config.prompts['target_marker']: tuple(colors.get('target_marker', [0, 255, 0])),
                # self.config.prompts['screw_head']: tuple(colors.get('screw_head', [0, 165, 255])),
            }

            draw_mask_debug(
                color, target_marks,
                output_path=f"{output_dir}/sam3_target_marks_and_screw_head.png",
                category_colors=color_map
            )

            # draw_mask_debug(
            #     color, target_marks,
            #     output_path=f"{output_dir}/{output_file}",
            #     category_colors={prompt: tuple(colors.get('target_marker', [0, 0, 255]))}
            # )
        
        return target_marks
    
    
    def  filter_by_area(self, masks: List[Dict], min_area: int) -> List[Dict]:
        """Filter masks by minimum area."""
        return [m for m in masks 
                if np.count_nonzero(m.get("segmentation", [])) >= min_area]
    
    def _dedupe_masks_by_bbox(self, masks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        unique: List[Dict[str, Any]] = []
        for m in masks:
            key = tuple(m.get("bbox", []))
            if key not in seen:
                seen.add(key)
                unique.append(m)
        return unique

    @staticmethod
    def _dedupe_masks_by_iou(
        masks: List[Dict[str, Any]],
        *,
        iou_threshold: float,
    ) -> List[Dict[str, Any]]:
        """Keep the highest-confidence mask from each overlapping detection cluster."""
        threshold = min(1.0, max(0.0, float(iou_threshold)))
        ordered = sorted(
            masks,
            key=lambda mask: float(mask.get("score", 0.0)),
            reverse=True,
        )
        kept: List[Dict[str, Any]] = []
        kept_segmentations: List[np.ndarray] = []
        for candidate in ordered:
            segmentation = candidate.get("segmentation")
            if segmentation is None:
                kept.append(candidate)
                continue
            candidate_mask = np.asarray(segmentation, dtype=bool)
            duplicate = False
            for kept_mask in kept_segmentations:
                if candidate_mask.shape != kept_mask.shape:
                    continue
                union = int(np.count_nonzero(candidate_mask | kept_mask))
                if union == 0:
                    continue
                intersection = int(np.count_nonzero(candidate_mask & kept_mask))
                if intersection / union >= threshold:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(candidate)
                kept_segmentations.append(candidate_mask)
        return kept

    def _get_refs_overlapping_heads(
        self,
        refs: List[Dict[str, Any]],
        heads: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Return reference masks (e.g. stems/boxes) that overlap at least one selected head."""
        if not refs or not heads:
            return []

        kept: List[Dict[str, Any]] = []
        for r in refs:
            r_mask = r.get("segmentation")
            if r_mask is None:
                continue
            if any(masks_overlap(h["segmentation"], r_mask) for h in heads):
                kept.append(r)
        return self._dedupe_masks_by_bbox(kept)

    def _normalize_screw_detection_mode(self, mode: Optional[str]) -> str:
        raw_mode = mode or self.config.get(
            'screwpickup_detection',
            'mode',
            default=self.config.get('screw_detection', 'mode', default='head_and_stem'),
        )
        norm = str(raw_mode or 'head_and_stem').strip().lower()
        aliases = {
            "stem_only": "stem_only",
            "stems_only": "stem_only",
            "stem": "stem_only",
            "head_only": "head_only",
            "heads_only": "head_only",
            "head": "head_only",
            "head_and_stem": "head_and_stem",
            "stem_and_head": "head_and_stem",
            "head_stem": "head_and_stem",
            "head_and_box": "head_and_box",
            "box_and_head": "head_and_box",
            "hand_and_box": "head_and_box",
            "head_box": "head_and_box",
            "head_and_support": "head_and_support",
            "overlap": "head_and_support",
            "both": "head_and_support",
        }
        resolved = aliases.get(norm)
        if resolved is None:
            print(
                f"[WARN] Unknown screw detection mode '{raw_mode}', "
                "falling back to 'head_and_stem'."
            )
            return "head_and_stem"
        return resolved

    def _compute_head_target_pixel_from_refs(
        self,
        head: Dict[str, Any],
        refs: List[Dict[str, Any]],
    ) -> Optional[Tuple[int, int]]:
        """Pick a point on the head side by choosing the head pixel farthest from overlapping refs."""
        if not refs:
            return None

        h_seg = head.get("segmentation")
        if h_seg is None:
            return None
        h_mask = np.squeeze(h_seg).astype(bool)
        if h_mask.ndim != 2 or not np.any(h_mask):
            return None

        ref_union = np.zeros_like(h_mask, dtype=bool)
        found_overlap = False
        for r in refs:
            r_seg = r.get("segmentation")
            if r_seg is None:
                continue
            r_mask = np.squeeze(r_seg).astype(bool)
            if r_mask.ndim != 2 or r_mask.shape != h_mask.shape:
                continue
            try:
                if masks_overlap(h_mask, r_mask):
                    ref_union |= r_mask
                    found_overlap = True
            except Exception:
                continue

        if not found_overlap:
            return None

        # Use the overlapping portion as the "stem entry" anchor when possible.
        anchor_mask = ref_union & h_mask
        if not np.any(anchor_mask):
            anchor_mask = ref_union
        if not np.any(anchor_mask):
            return None

        ays, axs = np.where(anchor_mask)
        ax = float(axs.mean())
        ay = float(ays.mean())

        # Bias away from edge noise by preferring an eroded core if available.
        core = cv2.erode(h_mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
        candidate_mask = core if np.any(core) else h_mask
        ys, xs = np.where(candidate_mask)
        if xs.size == 0:
            return None

        d2 = (xs.astype(np.float32) - ax) ** 2 + (ys.astype(np.float32) - ay) ** 2
        i = int(np.argmax(d2))
        return int(xs[i]), int(ys[i])

    def _attach_head_target_hints(
        self,
        overlap_heads: List[Dict[str, Any]],
        overlap_stems: List[Dict[str, Any]],
        overlap_boxes: List[Dict[str, Any]],
    ) -> None:
        """Mutate overlap heads with a preferred target pixel near the screw head portion."""
        for h in overlap_heads or []:
            h_seg = np.squeeze(h.get("segmentation", [])).astype(bool)
            best_stem = None
            best_overlap_area = 0
            if h_seg.ndim == 2:
                for stem in overlap_stems or []:
                    stem_seg = np.squeeze(stem.get("segmentation", [])).astype(bool)
                    if stem_seg.shape != h_seg.shape:
                        continue
                    overlap_area = int(np.count_nonzero(h_seg & stem_seg))
                    if overlap_area > best_overlap_area:
                        best_overlap_area = overlap_area
                        best_stem = stem
            if best_stem is not None and best_stem.get("angle_deg") is not None:
                h["stem_angle_deg"] = float(best_stem["angle_deg"])
            if best_stem is not None:
                head_center = h.get("center")
                stem_center = best_stem.get("center")
                if head_center is not None and stem_center is not None:
                    direction = np.asarray(stem_center, dtype=float) - np.asarray(head_center, dtype=float)
                    direction_norm = float(np.linalg.norm(direction))
                    if direction_norm >= 1e-8:
                        h["stem_direction_cam"] = [
                            float(direction[0] / direction_norm),
                            float(direction[1] / direction_norm),
                        ]

            target_px = self._compute_head_target_pixel_from_refs(h, overlap_stems or [])
            source = "stem_farthest_point"
            if target_px is None:
                target_px = self._compute_head_target_pixel_from_refs(h, overlap_boxes or [])
                source = "box_farthest_point"
            if target_px is None:
                continue
            h["target_pixel"] = (int(target_px[0]), int(target_px[1]))
            h["target_pixel_source"] = source

    def  get_overlapping_heads(
        self,
        heads,
        stems,
        boxes,
        *,
        mode: Optional[str] = None,
        return_combined_contour_maps: bool = False,
    ):
        """Find heads according to the requested support mode.

        Default return (backward-compatible):
          - List[head_mask_dict]

        Optional enriched return:
          - {
              "overlap_heads": [...],
              "combined_contour_maps": [...],  # stems + boxes that overlap selected heads
              "overlap_stems": [...],
              "overlap_boxes": [...],
            }
        """
        resolved_mode = self._normalize_screw_detection_mode(mode)

        if resolved_mode == "stem_only":
            # Downstream selection expects the legacy overlap_heads key; in this
            # mode it intentionally holds the selectable stem regions.
            overlap_heads = self._dedupe_masks_by_bbox(stems)
            overlap_stems = []
            overlap_boxes = []
        elif resolved_mode == "head_only":
            overlap_heads = self._dedupe_masks_by_bbox(heads)
            overlap_stems = []
            overlap_boxes = []
        elif resolved_mode == "head_and_stem":
            overlap_heads = self._dedupe_masks_by_bbox(filter_masks_by_overlap(heads, stems))
            overlap_stems = self._get_refs_overlapping_heads(stems, overlap_heads)
            overlap_boxes = []
        elif resolved_mode == "head_and_box":
            overlap_heads = self._dedupe_masks_by_bbox(filter_masks_by_overlap(heads, boxes))
            overlap_stems = []
            overlap_boxes = self._get_refs_overlapping_heads(boxes, overlap_heads)
        else:
            overlap_stem = filter_masks_by_overlap(heads, stems)
            overlap_box = filter_masks_by_overlap(heads, boxes)
            overlap_heads = self._dedupe_masks_by_bbox(overlap_stem + overlap_box)
            overlap_stems = self._get_refs_overlapping_heads(stems, overlap_heads)
            overlap_boxes = self._get_refs_overlapping_heads(boxes, overlap_heads)

        if not return_combined_contour_maps:
            return overlap_heads

        combined_contour_maps = self._dedupe_masks_by_bbox(overlap_stems + overlap_boxes)

        return {
            "overlap_heads": overlap_heads,
            "combined_contour_maps": combined_contour_maps,
            "overlap_stems": overlap_stems,
            "overlap_boxes": overlap_boxes,
            "mode": resolved_mode,
        }
    
    def  debug_visualize(self, color, heads, stems, boxes, overlap_heads, overlap_contour_maps=None):
        """Create debug visualizations."""
        output_dir = self.config.get('debug', 'output_dir', default='data/image_samples')
        colors = self.config.get('debug', 'colors', default={})
        prompts = self.config.prompts
        overlap_contour_maps = overlap_contour_maps or []

        def _make_panel(title: str, image_bgr: np.ndarray, width: int, height: int) -> np.ndarray:
            panel = np.full((height, width, 3), 24, dtype=np.uint8)
            cv2.rectangle(panel, (0, 0), (width - 1, height - 1), (70, 70, 70), 2, cv2.LINE_AA)
            cv2.rectangle(panel, (0, 0), (width - 1, 42), (38, 38, 38), -1)
            cv2.putText(panel, title, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (240, 240, 240), 2, cv2.LINE_AA)

            if image_bgr is None or image_bgr.size == 0:
                cv2.putText(panel, "No image", (20, max(70, height // 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (180, 180, 180), 2, cv2.LINE_AA)
                return panel

            image_area_w = width - 16
            image_area_h = height - 58
            src_h, src_w = image_bgr.shape[:2]
            scale = min(image_area_w / max(1, src_w), image_area_h / max(1, src_h))
            dst_w = max(1, int(round(src_w * scale)))
            dst_h = max(1, int(round(src_h * scale)))
            resized = cv2.resize(image_bgr, (dst_w, dst_h), interpolation=cv2.INTER_AREA)
            x0 = (width - dst_w) // 2
            y0 = 48 + (image_area_h - dst_h) // 2
            panel[y0:y0 + dst_h, x0:x0 + dst_w] = resized
            return panel
        
        # Convert BGR arrays to tuples
        color_map = {
            prompts['screw_head']: tuple(colors.get('screw_head', [0, 255, 0])),
            prompts['screw_stem']: tuple(colors.get('screw_stem', [0, 165, 255])),
            prompts['screw_box']: tuple(colors.get('screw_box', [255, 0, 0])),
        }
        
        all_masks = stems + heads + boxes
        all_vis = draw_mask_debug(
            color, all_masks,
            output_path=f"{output_dir}/sam3_all_detections.png",
            category_colors=color_map
        )

        # Show overlapping heads together with the overlapping supporting contours (stems/boxes).
        overlap_vis_masks = overlap_heads
        overlap_color_map = {
            prompts['screw_head']: color_map[prompts['screw_head']],
            prompts['screw_stem']: color_map[prompts['screw_stem']],
            prompts['screw_box']: color_map[prompts['screw_box']],
        }
        overlap_heads_vis = draw_mask_debug(
            color, overlap_vis_masks,
            output_path=f"{output_dir}/sam3_overlap_heads.png",
            category_colors=overlap_color_map
        )

        overlap_contours_vis = color.copy()
        try:
            overlap_contour_bytes = self.build_contour_overlay(
                color,
                {
                    "overlap_heads": overlap_heads,
                    "combined_contour_maps": overlap_contour_maps,
                },
                contour_type="overlap",
                debug=False,
            )
            overlap_contours_vis = cv2.imdecode(
                np.frombuffer(overlap_contour_bytes, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            if overlap_contours_vis is None:
                raise RuntimeError("Failed to decode overlap contour overlay image")
        except Exception as e:
            print(f"[WARN] Failed to build overlap contour overlay image: {e}")

        cv2.imwrite(f"{output_dir}/sam3_overlap_contours.png", overlap_contours_vis)
        print(f"[DEBUG] Saved contour debug image to: {output_dir}/sam3_overlap_contours.png")

        try:
            self._last_debug_overview = {
                "panels": [
                    ("All Detections", all_vis),
                    ("Overlap Heads", overlap_heads_vis),
                    ("Overlap Contours", overlap_contours_vis),
                ],
                "make_panel": _make_panel,
            }
            tile_w = 520
            tile_h = 420
            gap = 14
            dashboard = np.full((tile_h, tile_w * 3 + gap * 4, 3), 14, dtype=np.uint8)
            panels = self._last_debug_overview["panels"]
            x = gap
            for title, img in panels:
                panel = _make_panel(title, img, tile_w, tile_h)
                dashboard[:, x:x + tile_w] = panel
                x += tile_w + gap
            cv2.namedWindow("SAM3 Debug Overview", cv2.WINDOW_NORMAL)
            cv2.imshow("SAM3 Debug Overview", dashboard)
            print("[DEBUG] Press q/Esc or close the SAM3 Debug Overview window to continue.")
            while True:
                try:
                    if cv2.getWindowProperty("SAM3 Debug Overview", cv2.WND_PROP_VISIBLE) < 1:
                        break
                except Exception:
                    break
                key = cv2.waitKey(50) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    cv2.destroyWindow("SAM3 Debug Overview")
                    break
        except Exception as e:
            print(f"[WARN] debug_visualize GUI display failed: {e}")




class ScrewOrientationChecker:
    """Checks if screw is properly attached and oriented."""
    
    def __init__(self, handles, detector: Sam3Detector, config: Config):
        self.handles = handles
        self.detector = detector
        self.config = config
        self.camera_helper = CameraHelper(handles, config)
    
    def check_attachment(
        self,
        screwstem_conf: float = None,
        debug: bool = False
    ) -> Tuple[bool, bool]:
        """Check if screw is attached and vertical.
        
        Returns:
            (is_vertical, screw_on_head)
        """

        # ---------------------------------------------
        # DEBUG: Config / input info
        # ---------------------------------------------
        if debug:
            print("\n[DEBUG] ---- check_attachment() START ----")

        if screwstem_conf is None:
            screwstem_conf = self.config.confidence.get('screw_stem', 0.2)
        if debug:
            print(f"[DEBUG] screwstem_conf = {screwstem_conf}")

        # ---------------------------------------------
        # DEBUG: Camera read
        # ---------------------------------------------
        depth, color = self.handles.cam_fixed.get_rgbd()
        if debug:
            print("[DEBUG] Retrieved RGBD frame")
            print(f"[DEBUG] depth shape: {None if depth is None else depth.shape}")
            print(f"[DEBUG] color shape: {None if color is None else color.shape}")

        prompt = self.config.prompts['screw_stem']
        if debug:
            print(f"[DEBUG] Using segmentation prompt: {prompt}")

        # ---------------------------------------------
        # Segmentation
        # ---------------------------------------------
        stems = self.detector.segment(
            color_bgr=color,
            text_prompt=prompt,
            confidence_threshold=screwstem_conf,
            category=prompt,
        )
        if debug:
            print(f"[DEBUG] SAM returned {len(stems)} stem candidates")

        # ---------------------------------------------
        # Filter by min mask area
        # ---------------------------------------------
        min_area = self.config.get('filtering', 'stem', 'min_area_px', default=500)
        if debug:
            print(f"[DEBUG] Filtering stems with area >= {min_area}")

        filtered_stems = []
        for s in stems:
            seg = s.get("segmentation", [])
            area = np.count_nonzero(seg)
            if debug:
                print(f"[DEBUG] Stem area: {area}")
            if area >= min_area:
                filtered_stems.append(s)

        stems = filtered_stems
        if debug:
            print(f"[DEBUG] Remaining stems after area filtering: {len(stems)}")

        if not stems:
            if debug:
                print("[DEBUG] No stems remain — returning (False, False)")
            return False, False

        # ---------------------------------------------
        # Get TCP projection to camera img
        # ---------------------------------------------
        tcp_u, tcp_v = self.camera_helper.project_tcp_to_fixed_cam()
        if debug:
            print(f"[DEBUG] TCP projection: (u={tcp_u}, v={tcp_v})")

        if tcp_u is None or tcp_v is None:
            if debug:
                print("[DEBUG] TCP projection invalid — returning (False, False)")
            return False, False

        # ---------------------------------------------
        # Filtering near TCP
        # ---------------------------------------------
        max_dist = self.config.get('filtering', 'stem', 'max_distance_to_tcp_px', default=150.0)
        h_tol = self.config.get('filtering', 'attachment_check', 'horizontal_tolerance_px', default=100)
        require_below = self.config.get('filtering', 'attachment_check', 'require_below_tcp', default=True)

        if debug:
            print(f"[DEBUG] max_dist={max_dist}, h_tol={h_tol}, require_below={require_below}")

        nearby_stems = []
        for m in stems:
            cx, cy = mask_centroid(m["segmentation"])
            dist = float(np.hypot(cx - tcp_u, cy - tcp_v))
            
            below_check = (cy >= tcp_v) if require_below else True
            horiz_check = (tcp_u - h_tol <= cx <= tcp_u + h_tol)

            if debug:
                print(f"[DEBUG] Stem centroid={(cx,cy)}, dist={dist}, below={below_check}, horiz={horiz_check}")

            if dist <= max_dist and below_check and horiz_check:
                nearby_stems.append(m)

        if debug:
            print(f"[DEBUG] Nearby stems after TCP-distance filter: {len(nearby_stems)}")

        if debug:
            draw_mask_debug(
                color,
                nearby_stems,
                output_path="data/image_samples/sam3_stem_filtered_vertical_candidates.png",
                category_colors={prompt: (0, 165, 255)},
            )
            print("[DEBUG] Saved vertical candidate debug mask image → sam3_stem_filtered_vertical_candidates.png")

        if not nearby_stems:
            if debug:
                print("[DEBUG] No nearby stems — returning (False, False)")
            return False, False

        screw_on_head = True
        if debug:
            print("[DEBUG] screw_on_head = True (default assumption)")

        # ---------------------------------------------
        # Check verticality
        # ---------------------------------------------
        for m in nearby_stems:
            bbox = m.get("bbox")
            vertical = self.is_vertical(bbox)
            if debug:
                print(f"[DEBUG] Checking verticality on bbox={bbox} → {vertical}")

            if vertical:
                if debug:
                    print("[DEBUG] Found vertical stem! Returning (True, True)")
                return True, screw_on_head

        if debug:
            print("[DEBUG] No vertical stems found — returning (False, True)")

        return False, screw_on_head

    def  is_vertical(self, bbox: Optional[List[int]]) -> bool:
        """Check if bbox indicates vertical orientation."""
        if bbox is None or len(bbox) != 4:
            return False
        
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            return False
        
        min_aspect = self.config.get('filtering', 'stem', 'min_aspect_ratio', default=3.5)
        aspect = h / w
        return aspect >= min_aspect

class ManualScrewCycleManager:
    """
    Manual mode controller for hand-guided screwdriver operation.

    Responsibilities:
      - Start admittance_control_4d.urp for compliant hand-guiding.
      - Wait for trigger press (or ENTER on keyboard) to confirm "ready to screw".
      - Stop admittance program cleanly.
      - Run screwdriver_client and monitor its status.
      - Optionally provide voice feedback with voice_client.
    """

    def __init__(
        self,
        handles,
        screwdriver_client: "ScrewdriverClient | None",
        voice_client: "OpenAIRealtimeClient | None" = None,
        poll_interval: float = 0.1,
        cancel_check=None,
    ) -> None:
        """
        :param handles: Object providing arm_dash and arm handles.
        :param screwdriver_client: Screwdriver client instance or None if disabled.
        :param voice_client: Optional voice client for spoken feedback.
        :param poll_interval: Polling interval (seconds) for trigger/key checks.
        """
        cfg_file = utils.check_path_exists(CFG_PATH,__file__)       
        self.config = Config(cfg_file) 
        self.handles = handles
        self.screwdriver_client = screwdriver_client
        self.voice_client = voice_client
        self.poll_interval = poll_interval
        self.cancel_check = cancel_check

    def _wait_for_tool_trigger_hold(self, hold_sec: float = 3.0) -> bool:
        """Wait for a fresh trigger hold, ignoring shorter presses."""
        hold_sec = max(0.1, float(hold_sec))

        # The hold that entered manual mode must be released before a second
        # hold can confirm screwdriver execution.
        while bool(self.handles.arm.get_tool_io()):
            if self.cancel_check is not None and self.cancel_check():
                return False
            time.sleep(self.poll_interval)

        pressed_at = None
        print(f"[MANUAL] Hold the tool trigger for {hold_sec:.1f}s to run the screwdriver.")
        while True:
            if self.cancel_check is not None and self.cancel_check():
                return False

            pressed = bool(self.handles.arm.get_tool_io())
            now = time.monotonic()
            if pressed:
                if pressed_at is None:
                    pressed_at = now
                elif now - pressed_at >= hold_sec:
                    print(f"[MANUAL] Confirmed {hold_sec:.1f}s trigger hold.")
                    return True
            elif pressed_at is not None:
                print("[MANUAL] Trigger released too soon; continue holding for confirmation.")
                pressed_at = None

            time.sleep(self.poll_interval)

    def _stop_admittance_program(self) -> None:
        """Stop admittance and wait until the dashboard reports it stopped."""
        print("[MANUAL] Stopping admittance program...")
        self.handles.arm_dash.stop()
        timeout_s = float(
            self.config.get('hardware', 'timing', 'urp_start_timeout', default=10.0)
        )
        poll_s = float(
            self.config.get('hardware', 'timing', 'urp_poll_interval', default=0.25)
        )
        deadline = time.monotonic() + max(0.1, timeout_s)
        while time.monotonic() < deadline:
            state_response = self.handles.arm_dash.program_state()
            state = str(
                state_response.get("programState", state_response)
                if isinstance(state_response, dict)
                else state_response
            ).upper()
            if state.startswith("STOPPED") or not bool(self.handles.arm_dash.is_playing()):
                print(f"[MANUAL] Admittance program stopped: {state}")
                return
            time.sleep(max(0.05, poll_s))
        raise TimeoutError("Timed out waiting for the admittance program to stop.")

    def manual_position(self, debug: bool = False, trigger_hold_sec: float = 3.0) -> bool:

        # 1) Start admittance URP (for compliant hand guiding)
        print("[MANUAL] Starting admittance_control_4d.urp...")
        self.handles.arm_dash.connect()
        time.sleep(0.1)
        self.handles.arm_dash.load_urp(self.config.get('programs', 'admittance', default='admittance_control_4d.urp'))
        time.sleep(0.2)
        self.handles.arm_dash.play()
        time.sleep(0.5)

        # 2) Require a fresh sustained hold to confirm “ready to screw”.
        print("[MANUAL] Admittance active. Guide the tool to the screw location.")
        if not self._wait_for_tool_trigger_hold(trigger_hold_sec):
            try:
                self._stop_admittance_program()
                confirm_active_tcp(
                    self.handles,
                    "tcp_drill",
                    context="cancelled admittance program",
                )
            except Exception as e:
                print(f"[WARN] Failed to stop admittance after cancellation: {e}")
            return False

        # 3) Stop admittance program before running screwdriver
        try:
            self._stop_admittance_program()
            confirm_active_tcp(
                self.handles,
                "tcp_drill",
                context="completed admittance program",
            )
        except Exception as e:
            print(f"[WARN] Failed to stop admittance program cleanly: {e}")
            return False

        return True
 
    def run_cycle(self,debug = False) -> None:
        """
        Run the full manual screw sequence:

          1) Start admittance_control_4d.urp (hand-guiding).
          2) Wait for trigger or ENTER.
          3) Stop admittance.
          4) Run screwdriver_client and monitor status.
        """
        print("\n==================== MANUAL MODE ====================")

        if self.voice_client is not None:
            msg = self.config.get('voice', 'manual_mode_help',
                                         default="Manual mode. Guide the drill to the screw. Press the trigger to run the screwdriver.")
            self.voice_client.speak_openai(msg)

        # 1) Start admittance URP (for compliant hand guiding)
        print("[MANUAL] Starting admittance_control_4d.urp...")
        self.handles.arm_dash.connect()
        time.sleep(0.1)
        self.handles.arm_dash.load_urp(self.config.get('programs', 'admittance', default='admittance_control_4d.urp'))
        time.sleep(0.2)
        self.handles.arm_dash.play()
        time.sleep(0.5)

        # 2) Wait for trigger press to confirm “ready to screw”
        print("[MANUAL] Admittance active. Press trigger to run screwdriver...")
        if not self.wait_tool_button_press():
            try:
                print("[MANUAL] Manual mode cancelled while waiting for confirmation.")
                self.handles.arm_dash.stop()
                self.handles.arm_dash.wait_for_program()
            except Exception as e:
                print(f"[WARN] Failed to stop admittance program after cancellation: {e}")
            return

        # 3) Stop admittance program before running screwdriver
        try:
            print("[MANUAL] Stopping admittance program...")
            self.handles.arm_dash.stop()
            # Keeping original API usage; adjust if your SDK differs.
            self.handles.arm_dash.wait_for_program()
            confirm_active_tcp(
                self.handles,
                "tcp_drill",
                context="completed manual admittance cycle",
            )
        except Exception as e:
            print(f"[WARN] Failed to stop admittance program cleanly: {e}")

        # 4) Run screwdriver operation
        if self.screwdriver_client is None:
            print("[WARN] Screwdriver client not enabled, skipping screw operation.")
            if self.voice_client is not None:
                self.voice_client.speak_openai(
                    "Screwdriver client is not enabled. Skipping screw operation."
                )
            return

        if self.voice_client is not None:
            msg = self.config.get('voice', 'install_start',
                                         default="Screw installation starting now. Please move away from the robot.")
            self.voice_client.speak_openai(msg)
       

        print("[MANUAL] Running screwdriver client operation...")
        self.screwdriver_client.run_screw_async(debug=debug)

        while True:
            status = self.screwdriver_client.get_status()
            print("[MANUAL] Screwdriver status:", status)
            if status.state in ("completed", "error"):
                break
            time.sleep(0.3)

        if status.state == "error":
            print("[MANUAL] Screwdriver operation failed:", status.error)
            if self.voice_client is not None:
                msg = self.config.get('voice', 'screwdriver_error',
                                         default="Screwdriver operation failed. Please check the screw.")
                self.voice_client.speak_openai(msg)
        else:
            print("[MANUAL] Screwdriver operation completed successfully.")
            if self.voice_client is not None:
                msg = self.config.get('voice', 'screwdriver_success',
                                         default="Screw installation complated.")
                self.voice_client.speak_openai(msg)

    def  wait_tool_button_press(self) -> bool:
        """
        Block until either:
          - Tool trigger is pressed (handle.arm.get_tool_io() returns truthy), or
          - ENTER key is pressed on stdin.
        """
        print("[INFO] Waiting for trigger or ENTER KEY to continue...")
        while True:
            # Read tool IO
            try:
                pressed = self.handles.arm.get_tool_io()
            except Exception as e:
                msg = f"Failed to read stop button state before start: {e}"
                raise RuntimeError(msg)

            if self.cancel_check is not None and self.cancel_check():
                print("[MANUAL] Cancellation requested while waiting for trigger.")
                return False

            if pressed:
                print("[STEP] Tool button pressed.")
                return True

            # Non-blocking check for keyboard ENTER
            try:
                rlist, _, _ = select.select([sys.stdin], [], [], 0)
            except Exception:
                rlist = []

            if rlist:
                # Read the line (user hit Enter or typed something)
                _ = sys.stdin.readline()
                print("[STEP] Enter key detected on keyboard.")
                return True

            time.sleep(self.poll_interval)

class ScrewCycleManager:
    """Manages complete screw pickup and installation cycles."""

    def __init__(
        self,
        handles,
        detector: Sam3Detector,
        screwdriver_client: Optional[ScrewdriverClient] = None,
        voice_client: Optional[OpenAIRealtimeClient] = None,
        move_to_start: bool = True,
        cancel_check=None,
    ):
    
        cfg_file = utils.check_path_exists(CFG_PATH,__file__)       
        config = Config(cfg_file) 
        self.handles = handles
        self.detector = detector
        self.config = config
        self.screwdriver_client = screwdriver_client
        self.voice_client = voice_client
        self.cancel_check = cancel_check
        openai_sdk_client = getattr(voice_client, "openai_sdk_client", None)
        self.image_query_client = OpenAiQueryClient.from_config_dict(
            self.config.get('llm', 'openai_query', default={}) or {},
            openai_sdk_client=openai_sdk_client,
        )
        self.camera_helper = CameraHelper(handles, config)
        self.motion_planner = MotionPlanner(handles, config)
        self.screw_detector = ScrewDetector(detector, config)
        # self.generic_detector = GenericDetector(detector, config)
        self.generic_cycle = GenericCycleManager(
                            handles=handles,
                            detector=detector,
                            voice_client=None,
                            move_to_start=False,
                        )
        self.orientation_checker = ScrewOrientationChecker(handles, detector, config)
        self._pickup_arm_reference_cache: Dict[Any, Dict[str, Any]] = {}

        if move_to_start:
            self.move_to_start_position()

    def confirm_drill_tcp(self, *, context: str) -> List[float]:
        return confirm_active_tcp(self.handles, "tcp_drill", context=context)

    def move_to_start_position(self):
        j = self.config.get('start_joint')
        print('[STEP] Moving to start position...',j)
        self.handles.arm.moveJ(self.config.get('start_joint'))


    def move_to_screw_target_position(self):
        j = self.config.get('screw_target_joint')
        print('[STEP] Moving to screw_target_joint position...',j)
        self.handles.arm.moveJ(self.config.get('screw_target_joint'))


    def move_to_screw_pickup_position(self):
        j = self.config.get('screw_pickup_joint')
        print('[STEP] Moving to screw_pickup position...',j)
        self.handles.arm.moveJ(self.config.get('screw_pickup_joint'))

    # def move_to_ready_position(self):
    #     j = self.config.get('ready_joint')
    #     print('[STEP] Moving to ready position...',j)
    #     self.handles.arm.moveJ(self.config.get('ready_joint'))


    def run_cycle(self, use_servo: bool = False, enable_screwdriver:bool=False,pickup_screw:bool=True, move_to_cross:bool=True, debug: bool = False) -> bool:
        """Execute one complete screw cycle."""
        print("\n" + "="*50)
        print("NEW SCREW CYCLE")
        print("="*50)
        success = True

        if self.voice_client is not None:
            msg = self.config.get('voice', 'auto_mode_help',default="Auto installation starting.")
            self.voice_client.speak_openai(msg)

        try:
            # Check initial state

            #VARIFY SCREW ATTACHMENT AND ORIENTATION
            # is_vertical, screw_on_head = self.orientation_checker.check_attachment(
            #     debug=debug
            # )
            
            # if not is_vertical and not screw_on_head:
            #     success = self.pickup_screw(debug)
            #     if not self.verify_attachment_loop(debug):
            #         return False
            # elif screw_on_head and not is_vertical:
            #     if not self.verify_attachment_loop(debug):
            #         return False

            #INGORE VARIFICATION FOR NOW
            if pickup_screw:
                success = self.move_to_screw(use_servo=use_servo,debug=debug)
                if success:
                    self.run_pickup_program()
                    success = True


            # Move to target mark and install
            if success and move_to_cross:
                success = self.move_to_target_mark(use_servo=use_servo,debug=debug)
                if success and enable_screwdriver:
                    success = self.run_screwdriver()
            
            return success
            
        except Exception as e:
            print(f"[ERROR] Cycle failed: {e}")
            return False
    
    def move_to_screw(self, use_servo:bool = False, speed: Optional[float] = None,  debug: bool=False):

        print("[STEP] Detecting screws...")
        offset = self.config.offsets['initial_approach']
        clearance = self.config.offsets['pickup_clearance']
        observation_offset = self.config.offsets['observation_offset']
        camera_type  = 'fixed'
    
        # if not use_servo: #dont do approach if in normal mode
            # success = self.detect_and_move_to_screw(debug, camera=camera_type, z_offset=offset + clearance)
        success, status = self.generic_cycle.move_to_generic_prompt(self.config.prompts['screw_stem'],conf=0.2, z_offset=observation_offset, camera_type=camera_type,move=True)
        if success:
            camera_type  = 'arm'
            success = self.detect_and_move_to_screw(debug, camera=camera_type, z_offset=offset)

        camera_type  = 'arm'
        if use_servo:
            print("[INFO] Using visual servo (detection + target selection in servo loop).")
            
            def target_pose_fn() -> Optional[List[float]]:
                prompt = self.config.prompts['screw_head']
                conf = self.config.confidence.get('screw_head', 0.2)

                T_base_tcp = self.handles.arm.get_T_base_tcp()
                depth, color, intr, T_cam = self.camera_helper.get_rgbd_and_intrinsics(camera_type)
                z_offset = self.config.get('motion', 'offsets', 'refine_approach', default=0.1)                
                # masks = self.screw_detector.detect_screws(color, debug=debug)
                masks = self.generic_cycle.detect_generic(prompt=prompt,color=color,conf=conf) #detect just screw head for faster inference
                if not masks:
                    print(f"No screws detected from {camera_type} camera")
                    return False
                
                targets = self.compute_targets(color,prompt,masks, depth, intr)
                candidates = self.motion_planner.compute_approach_poses(T_base_tcp,
                    targets, T_cam, camera_type, z_offset, ignore_rotation=True,
                    orientation_profile="pickup",
                )
    
                if not candidates:
                    print("[SERVO] No candidate poses from targets.")
                    return None

                # 5) Select best candidate given current robot state
                best_cand, best_cost = self.motion_planner.select_best_candidate(
                    candidates,
                    ignore_close=False,
                )
                if best_cand is None:
                    print("[SERVO] No reachable candidates.")
                    return None

                return best_cand["pose"]

            status = self.motion_planner.servo_towards_dynamic(
                target_pose_fn=target_pose_fn,
                speed=speed if speed is not None else self.config.default_speed,
            )
            success = False
            print('servo finish status',status)
            if 'target_reached' in status:
                success = True
            
            return success
    
        #dont use servo   
        if self.config.get('features', 'refine_positioning_screw', default=False):
            refine_offset = self.config.get('motion', 'offsets', 'refine_approach', default=0.1)
            success = self.detect_and_move_to_screw(debug, camera="arm", z_offset=refine_offset)
        
        # print("[STEP] Running pickup program...")
        # if success:
        #     self.run_pickup_program()
        #     return True
        
        return success
    
    def detect_and_move_to_screw(
        self,
        debug: bool,
        camera: str,
        z_offset: float,
        y_offset: float = 0.0,
        x_offset: float = 0.0,
        target_mode: str = "auto",
    ) -> bool:
        """Detect screw and move to it."""
        self.confirm_drill_tcp(context=f"{camera} screw detection and planning")
        depth, color, intr, T_cam = self.camera_helper.get_rgbd_and_intrinsics(camera)
        T_base_tcp = self.handles.arm.get_T_base_tcp()

        target_mode_norm = str(target_mode or "auto").strip().lower()
        if target_mode_norm not in ("auto", "head", "center"):
            print(f"[WARN] Unknown target_mode '{target_mode}', falling back to 'auto'")
            target_mode_norm = "auto"

        screw_result = self.screw_detector.detect_screws(color, camera_type=camera, debug=debug)
        if isinstance(screw_result, dict):
            masks = screw_result.get("overlap_heads", []) or []
        else:
            masks = screw_result or []

        if not masks:
            print(f"No screws detected from {camera} camera")
            return False

        if target_mode_norm == "center":
            center_masks = []
            for m in masks:
                m2 = dict(m)
                m2.pop("target_pixel", None)
                m2.pop("target_pixel_source", None)
                center_masks.append(m2)
            masks = center_masks
        # target_mode=head or auto keeps any head-target hints attached by detect_screws().
        
        prompt = self.config.prompts['screw_head']
        targets = self.compute_targets(color,prompt,masks, depth, intr)
        candidates = self.motion_planner.compute_approach_poses(
            T_base_tcp,
            targets, T_cam, camera, z_offset=z_offset, ignore_rotation=True,
            y_offset=y_offset, x_offset=x_offset, orientation_profile="pickup",
        )
        
        return self.motion_planner.move_to_best_reachable(candidates,speed=self.config.default_speed,debug=debug)

    def _resolve_T_base_cam_for_selector(
        self,
        *,
        camera: str,
        T_cam: np.ndarray,
        T_base_tcp: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        cam = str(camera or "").strip().lower()
        if T_cam is None:
            return None
        if cam == "arm":
            if T_base_tcp is None:
                return None
            return T_base_tcp @ T_cam
        return T_cam

    def _warp_overlay_to_birdseye(
        self,
        image_bgr: np.ndarray,
        *,
        intr: Any,
        T_base_cam: np.ndarray,
        plane_z_base: float = 0.0,
        pixels_per_meter: float = 1200.0,
        padding_px: int = 16,
        margin_m: float = 0.0,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Warp an image to a top-down view of a base-frame Z plane using camera calibration.

        The transform is computed from image corner rays intersected with the plane.
        """
        if image_bgr is None or getattr(image_bgr, "ndim", 0) < 2:
            raise ValueError("image_bgr must be a valid image array")
        if T_base_cam is None or np.shape(T_base_cam) != (4, 4):
            raise ValueError("T_base_cam must be a 4x4 transform")

        h, w = image_bgr.shape[:2]
        fx, fy, cx, cy = utils._get_intrinsics_fx_fy_cx_cy(intr)
        fx = float(fx)
        fy = float(fy)
        cx = float(cx)
        cy = float(cy)
        if fx == 0.0 or fy == 0.0:
            raise ValueError("Invalid intrinsics: fx/fy must be non-zero")

        R_base_cam = np.asarray(T_base_cam[:3, :3], dtype=float)
        p_base_cam = np.asarray(T_base_cam[:3, 3], dtype=float).reshape(3,)
        plane_z_base = float(plane_z_base)
        pixels_per_meter = float(pixels_per_meter)
        padding_px = int(padding_px)
        margin_m = float(margin_m)
        if pixels_per_meter <= 0:
            raise ValueError("pixels_per_meter must be > 0")

        src_pts = np.array(
            [
                [0.0, 0.0],
                [float(w - 1), 0.0],
                [float(w - 1), float(h - 1)],
                [0.0, float(h - 1)],
            ],
            dtype=np.float32,
        )

        plane_xy = []
        for u, v in src_pts:
            d_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=float)
            d_base = R_base_cam @ d_cam
            dz = float(d_base[2])
            if abs(dz) < 1e-9:
                raise ValueError("Camera ray is parallel to birdseye plane")
            t = (plane_z_base - float(p_base_cam[2])) / dz
            if t <= 0:
                raise ValueError("Birdseye plane is behind camera for at least one image corner")
            p = p_base_cam + t * d_base
            plane_xy.append([float(p[0]), float(p[1])])

        plane_xy = np.asarray(plane_xy, dtype=float)
        x_min = float(np.min(plane_xy[:, 0])) - margin_m
        x_max = float(np.max(plane_xy[:, 0])) + margin_m
        y_min = float(np.min(plane_xy[:, 1])) - margin_m
        y_max = float(np.max(plane_xy[:, 1])) + margin_m
        span_x = max(1e-6, x_max - x_min)
        span_y = max(1e-6, y_max - y_min)

        out_w = max(64, int(math.ceil(span_x * pixels_per_meter)) + 2 * padding_px)
        out_h = max(64, int(math.ceil(span_y * pixels_per_meter)) + 2 * padding_px)

        dst_pts = np.empty((4, 2), dtype=np.float32)
        for i, (x_b, y_b) in enumerate(plane_xy):
            dst_x = (x_b - x_min) * pixels_per_meter + padding_px
            # Flip Y so base +Y appears upward in the bird's-eye image.
            dst_y = (y_max - y_b) * pixels_per_meter + padding_px
            dst_pts[i] = [float(dst_x), float(dst_y)]

        H = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(
            image_bgr,
            H,
            (out_w, out_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        meta = {
            "applied": True,
            "plane_z_base": plane_z_base,
            "pixels_per_meter": pixels_per_meter,
            "padding_px": padding_px,
            "margin_m": margin_m,
            "output_size": [int(out_w), int(out_h)],
            "plane_bounds_xy": [x_min, y_min, x_max, y_max],
        }
        return warped, meta

    def select_region_with_vision_llm(
        self,
        color,
        screws,
        request: str,
        *,
        vision_model: str,
        depth: Optional[np.ndarray] = None,
        intr: Any = None,
        birdseye_context: Optional[Dict[str, Any]] = None,
        debug: bool = False,
    ) -> Tuple[Optional[int], Dict[str, Any]]:
        if not screws:
            return None, {"reason": "No screw regions to rank."}

        if isinstance(screws, dict):
            head_regions: List[Dict[str, Any]] = screws.get("overlap_heads", []) or []
            support_regions: List[Dict[str, Any]] = screws.get("combined_contour_maps", []) or []
            overlay_payload = screws
        else:
            head_regions = screws or []
            support_regions = []
            overlay_payload = head_regions

        if not head_regions:
            return None, {"reason": "No selectable screw head regions found."}

        # If only one head exists, no vision ranking is needed even if multiple overlap contours exist.
        if len(head_regions) == 1:
            return 0, {"reason": "Only one screw region detected."}

        if self.image_query_client is None:
            return None, {"reason": "OpenAI image query client is not configured for vision ranking."}

        select_overlap_contours = bool(isinstance(screws, dict) and support_regions)
        selectable_prefix = "R"
        selectable_regions = head_regions
        vision_cfg = self.config.get('llm', 'vision_selector', default={}) or {}
        requested_contour_type = str(
            vision_cfg.get("contour_type", "overlap" if select_overlap_contours else "heads_only")
        ).strip().lower()
        contour_type = requested_contour_type
        if contour_type == "auto":
            contour_type = "both" if select_overlap_contours else "heads_only"
        if contour_type == "overlap" and not support_regions:
            contour_type = "heads_only"
        if contour_type not in ("overlap", "heads_only", "both"):
            contour_type = "both" if select_overlap_contours else "heads_only"

        try:
            contour_overlayed_image = self.screw_detector.build_contour_overlay(
                color,
                overlay_payload,
                contour_type=contour_type,
                debug=debug,
                head_label_prefix="R",
                label_support_contours=False,
                include_scores_in_labels=False,
                draw_head_outlines=False,
                draw_support_outlines=False,
                label_color_bgr=(255, 255, 255),
            )
        except TypeError:
            # Backward compatibility if build_contour_overlay() doesn't accept contour_type yet.
            contour_overlayed_image = self.screw_detector.build_contour_overlay(
                color, overlay_payload, debug=debug
            )

        birdseye_meta: Optional[Dict[str, Any]] = None
        if isinstance(birdseye_context, dict) and birdseye_context.get("enabled"):
            birdseye_meta = {
                "applied": False,
                "reason": "Birdseye rectification disabled for vision selector image.",
            }

        def _bbox_overlap_area(a_xywh: List[int], b_xywh: List[int]) -> int:
            ax, ay, aw, ah = [int(v) for v in (a_xywh or [0, 0, 0, 0])]
            bx, by, bw, bh = [int(v) for v in (b_xywh or [0, 0, 0, 0])]
            ax2, ay2 = ax + max(aw, 0), ay + max(ah, 0)
            bx2, by2 = bx + max(bw, 0), by + max(bh, 0)
            ix1, iy1 = max(ax, bx), max(ay, by)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            return max(0, ix2 - ix1) * max(0, iy2 - iy1)

        def _map_support_contour_to_head_mask(support_mask: Dict[str, Any]) -> Optional[Tuple[int, Dict[str, Any]]]:
            best_idx: Optional[int] = None
            best_score = -1
            s_seg = support_mask.get("segmentation")
            s_bbox = support_mask.get("bbox", [0, 0, 0, 0])
            for i, h in enumerate(head_regions):
                score = 0
                h_seg = h.get("segmentation")
                if s_seg is not None and h_seg is not None:
                    try:
                        if getattr(s_seg, "shape", None) == getattr(h_seg, "shape", None):
                            score = int(np.count_nonzero(np.logical_and(s_seg.astype(bool), h_seg.astype(bool))))
                    except Exception:
                        score = 0
                    if score <= 0:
                        try:
                            if masks_overlap(h_seg, s_seg):
                                score = 1
                        except Exception:
                            pass
                if score <= 0:
                    score = _bbox_overlap_area(s_bbox, h.get("bbox", [0, 0, 0, 0]))
                if score > best_score:
                    best_score = score
                    best_idx = i

            if best_idx is None or best_score <= 0:
                return None
            return best_idx, head_regions[best_idx]

        def _region_mask_bool(m: Dict[str, Any]) -> Optional[np.ndarray]:
            seg = m.get("segmentation")
            if seg is None:
                return None
            try:
                seg_bool = np.squeeze(seg).astype(bool)
            except Exception:
                return None
            if getattr(seg_bool, "ndim", 0) != 2:
                return None
            if seg_bool.shape[:2] != color.shape[:2]:
                return None
            return seg_bool

        def _estimate_region_length_width_px(m: Dict[str, Any]) -> Tuple[float, float]:
            contour = m.get("contour")
            if contour is not None and len(contour) >= 3:
                try:
                    (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
                    long_side = float(max(rw, rh))
                    short_side = float(min(rw, rh))
                    if long_side > 0 and short_side >= 0:
                        return long_side, short_side
                except Exception:
                    pass

            seg_bool = _region_mask_bool(m)
            if seg_bool is not None and np.any(seg_bool):
                ys, xs = np.where(seg_bool)
                w_px = float(np.max(xs) - np.min(xs) + 1)
                h_px = float(np.max(ys) - np.min(ys) + 1)
                return max(w_px, h_px), min(w_px, h_px)

            x, y, bw, bh = [int(v) for v in m.get("bbox", [0, 0, 0, 0])]
            return float(max(bw, bh)), float(min(bw, bh))

        def _estimate_region_length_width_m(
            m: Dict[str, Any],
            *,
            fallback_length_px: float,
            fallback_width_px: float,
        ) -> Tuple[Optional[float], Optional[float]]:
            if depth is None or intr is None:
                return None, None

            seg_bool = _region_mask_bool(m)
            if seg_bool is not None and np.any(seg_bool):
                ys, xs = np.where(seg_bool)
            else:
                x, y, bw, bh = [int(v) for v in m.get("bbox", [0, 0, 0, 0])]
                if bw <= 0 or bh <= 0:
                    return None, None
                yy, xx = np.mgrid[y:y + bh, x:x + bw]
                ys = yy.reshape(-1)
                xs = xx.reshape(-1)

            if xs.size < 4:
                return None, None

            # Downsample for speed if the region is large.
            max_pts = 2500
            if xs.size > max_pts:
                step = int(np.ceil(xs.size / max_pts))
                xs = xs[::step]
                ys = ys[::step]

            H, W = depth.shape[:2]
            xs_i = np.clip(xs.astype(int), 0, max(W - 1, 0))
            ys_i = np.clip(ys.astype(int), 0, max(H - 1, 0))
            d_raw = depth[ys_i, xs_i].astype(np.float32)
            valid = d_raw > 0
            if not np.any(valid):
                return None, None
            xs_i = xs_i[valid].astype(np.float32)
            ys_i = ys_i[valid].astype(np.float32)
            z_m = d_raw[valid].astype(np.float32) * 0.001
            if z_m.size < 4:
                return None, None

            fx, fy, cx, cy = utils._get_intrinsics_fx_fy_cx_cy(intr)
            fx = float(fx)
            fy = float(fy)
            cx = float(cx)
            cy = float(cy)
            if abs(fx) < 1e-9 or abs(fy) < 1e-9:
                return None, None

            x_m = (xs_i - cx) * z_m / fx
            y_m = (ys_i - cy) * z_m / fy
            pts = np.stack([x_m, y_m, z_m], axis=1).astype(np.float32)

            if pts.shape[0] >= 8:
                try:
                    mean = np.mean(pts, axis=0, keepdims=True)
                    centered = pts - mean
                    _, _, vt = np.linalg.svd(centered, full_matrices=False)
                    axes = vt[:2].T  # first two principal axes
                    proj = centered @ axes
                    extents = np.max(proj, axis=0) - np.min(proj, axis=0)
                    long_m = float(np.max(extents))
                    short_m = float(np.min(extents))
                    if long_m > 0 and short_m >= 0:
                        return long_m, short_m
                except Exception:
                    pass

            # Fallback: pixel size scaled by median depth.
            z_med = float(np.median(z_m))
            m_per_px = 0.5 * ((z_med / fx) + (z_med / fy))
            return float(fallback_length_px) * m_per_px, float(fallback_width_px) * m_per_px

        def _region_color_metadata(m: Dict[str, Any]) -> Dict[str, Any]:
            seg_bool = _region_mask_bool(m)
            pixels = None
            if seg_bool is not None and np.any(seg_bool):
                pixels = color[seg_bool]
            else:
                x, y, bw, bh = [int(v) for v in m.get("bbox", [0, 0, 0, 0])]
                x1 = int(np.clip(x, 0, max(color.shape[1] - 1, 0)))
                y1 = int(np.clip(y, 0, max(color.shape[0] - 1, 0)))
                x2 = int(np.clip(x + max(bw, 1), x1 + 1, color.shape[1]))
                y2 = int(np.clip(y + max(bh, 1), y1 + 1, color.shape[0]))
                roi = color[y1:y2, x1:x2]
                if roi.size > 0:
                    pixels = roi.reshape(-1, roi.shape[-1])

            if pixels is None or len(pixels) == 0:
                return {
                    "dominant_color": "unknown",
                    "mean_bgr": [0, 0, 0],
                    "mean_rgb": [0, 0, 0],
                    "mean_hsv": [0, 0, 0],
                }

            mean_bgr = np.mean(pixels[:, :3], axis=0).astype(float)
            mean_bgr_u8 = np.clip(np.round(mean_bgr), 0, 255).astype(np.uint8).reshape(1, 1, 3)
            mean_hsv = cv2.cvtColor(mean_bgr_u8, cv2.COLOR_BGR2HSV)[0, 0].astype(int)
            b, g, r = [int(v) for v in mean_bgr_u8[0, 0]]
            h_val, s_val, v_val = [int(v) for v in mean_hsv]

            if v_val < 45:
                name = "black"
            elif s_val < 35:
                if v_val > 200:
                    name = "white"
                elif v_val > 120:
                    name = "silver_gray"
                else:
                    name = "dark_gray"
            elif h_val < 10 or h_val >= 170:
                name = "red"
            elif h_val < 22:
                name = "orange"
            elif h_val < 35:
                name = "yellow"
            elif h_val < 85:
                name = "green"
            elif h_val < 130:
                name = "blue"
            elif h_val < 160:
                name = "purple"
            else:
                name = "brown" if (r > g > b) else "red"

            # Special-case brown-ish low-value orange hues.
            if name in ("orange", "yellow") and v_val < 160 and r >= g and g > b:
                name = "brown"

            return {
                "dominant_color": name,
                "mean_bgr": [int(b), int(g), int(r)],
                "mean_rgb": [int(r), int(g), int(b)],
                "mean_hsv": [h_val, s_val, v_val],
            }

        regions = []
        label_to_region_id: Dict[str, int] = {}
        label_to_mask: Dict[str, Dict[str, Any]] = {}
        for idx, m in enumerate(selectable_regions):
            label = f"{selectable_prefix}{idx}"
            label_to_region_id[label] = idx
            mapped_head_idx = None
            mapped_mask = None
            if selectable_prefix == "C":
                mapped = _map_support_contour_to_head_mask(m)
                if mapped is not None:
                    mapped_head_idx, mapped_mask = mapped
            else:
                mapped_head_idx, mapped_mask = idx, m
            if mapped_mask is not None:
                label_to_mask[label] = mapped_mask
            length_px, width_px = _estimate_region_length_width_px(m)
            length_m, width_m = _estimate_region_length_width_m(
                m,
                fallback_length_px=length_px,
                fallback_width_px=width_px,
            )
            color_meta = _region_color_metadata(m)
            regions.append(
                {
                    "region_id": idx,
                    "label": label,
                    "bbox_xywh": [int(v) for v in m.get("bbox", [0, 0, 0, 0])],
                    "sam_score": float(m.get("score", 0.0)),
                    "mapped_head_region_id": mapped_head_idx,
                    "source_category": str(m.get("category", "")),
                    "estimated_length_px": round(float(length_px), 1),
                    "estimated_width_px": round(float(width_px), 1),
                    "estimated_length_m": (round(float(length_m), 4) if length_m is not None else None),
                    "estimated_width_m": (round(float(width_m), 4) if width_m is not None else None),
                    **color_meta,
                }
            )

        selectable_labels = [r["label"] for r in regions]
        default_prompt_template = (
            "You are selecting one labeled contour for a robot pickup request.\n"
            "User request: {request}\n"
            "The image contains labeled contours. Select exactly ONE contour label.\n"
            "Selectable labels: {selectable_labels_csv}\n"
            "Only choose labels that start with '{selectable_prefix}'. Do not choose other labels.\n"
            "Make sure there is a screw inside the selected contour box.\n"
            "Choose the ONE contour that best corresponds to the requested screw location/type based on nearby contour shape/context.\n"
            "If uncertain, choose the best available candidate instead of refusing.\n"
            "Return JSON only in this format: {json_example}"
        )
        prompt_template = str(vision_cfg.get("prompt_template", default_prompt_template))
        prompt_vars = {
            "request": request,
            "selectable_prefix": selectable_prefix,
            "selectable_labels_csv": ", ".join(selectable_labels),
            "selectable_labels": selectable_labels,
            "contour_type": contour_type,
            "json_example": '{"best_label": "R0", "reason": "short reason"}',
        }
        try:
            prompt_text = prompt_template.format(**prompt_vars)
        except Exception as e:
            print(f"[WARN] Failed to format llm.vision_selector.prompt_template: {e}")
            prompt_text = default_prompt_template.format(**prompt_vars)
        if bool(vision_cfg.get("include_unlabeled_context_image", False)):
            prompt_text += (
                "\n\nTwo images are provided in order: "
                "(1) the labeled selector image, "
                "(2) the original unlabeled camera image for visual context."
            )

        if debug:
            print("[DEBUG] Vision selector prompt sent to GPT API:")
            print(prompt_text)
            try:
                out_dir = ROOT / "sample_images"
                out_dir.mkdir(parents=True, exist_ok=True)
                prompt_path = out_dir / "vision_selector_prompt.txt"
                prompt_path.write_text(prompt_text, encoding="utf-8")
                print(f"[DEBUG] Saved vision selector prompt to: {prompt_path}")
            except Exception as e:
                print(f"[WARN] Failed to save vision selector prompt debug file: {e}")

        vision_query_cfg = self.config.get('llm', 'vision_selector', 'query', default={}) or {}
        query_mime_type = str(vision_query_cfg.get("mime_type", "image/jpeg"))
        query_fallback_models = vision_query_cfg.get("fallback_models")
        if not isinstance(query_fallback_models, list):
            query_fallback_models = None
        response_create_kwargs = vision_query_cfg.get("responses_create", {}) if isinstance(vision_query_cfg, dict) else {}
        if not isinstance(response_create_kwargs, dict):
            response_create_kwargs = {}
        include_unlabeled_context_image = bool(vision_cfg.get("include_unlabeled_context_image", False))

        sent_image_path: Optional[str] = None
        sent_unlabeled_image_path: Optional[str] = None
        query_image_items: List[Dict[str, Any]] = []
        try:
            out_dir = ROOT / "data" / "image_samples"
            out_dir.mkdir(parents=True, exist_ok=True)
            sent_ext = ".jpg" if "jpeg" in query_mime_type.lower() or "jpg" in query_mime_type.lower() else ".png"
            sent_path = out_dir / f"vision_selector_sent_image{sent_ext}"
            if isinstance(contour_overlayed_image, (bytes, bytearray)):
                sent_path.write_bytes(bytes(contour_overlayed_image))
            else:
                overlay_bgr = np.asarray(contour_overlayed_image)
                ok_enc, buf = cv2.imencode(sent_ext, overlay_bgr)
                if not ok_enc:
                    raise RuntimeError("Failed to encode vision selector image for saving")
                sent_path.write_bytes(buf.tobytes())
            sent_image_path = str(sent_path)
            print(f"[DEBUG] Saved vision selector image sent to LLM: {sent_path}")
            query_image_items.append(
                {
                    "image_bytes": contour_overlayed_image,
                    "mime_type": query_mime_type,
                }
            )
        except Exception as e:
            print(f"[WARN] Failed to save vision selector image sent to LLM: {e}")

        if include_unlabeled_context_image:
            try:
                out_dir = ROOT / "data" / "image_samples"
                out_dir.mkdir(parents=True, exist_ok=True)
                raw_ext = ".jpg" if "jpeg" in query_mime_type.lower() or "jpg" in query_mime_type.lower() else ".png"
                raw_path = out_dir / f"vision_selector_unlabeled_image{raw_ext}"
                ok_raw_enc, raw_buf = cv2.imencode(raw_ext, color)
                if not ok_raw_enc:
                    raise RuntimeError("Failed to encode unlabeled context image for saving")
                raw_bytes = raw_buf.tobytes()
                raw_path.write_bytes(raw_bytes)
                sent_unlabeled_image_path = str(raw_path)
                print(f"[DEBUG] Saved unlabeled vision selector image sent to LLM: {raw_path}")
                query_image_items.append(
                    {
                        "image_bytes": raw_bytes,
                        "mime_type": query_mime_type,
                    }
                )
            except Exception as e:
                print(f"[WARN] Failed to save unlabeled vision selector image sent to LLM: {e}")

        if not query_image_items:
            query_image_items.append(
                {
                    "image_bytes": contour_overlayed_image,
                    "mime_type": query_mime_type,
                }
            )

        try:
            query_result = self.image_query_client.query(
                prompt_text=prompt_text,
                image_items=query_image_items,
                model=vision_model,
                mime_type=query_mime_type,
                fallback_models=query_fallback_models,
                response_create_kwargs=response_create_kwargs,
            )
            if not query_result.get("ok"):
                return None, {
                    "reason": query_result.get("reason", "Vision model call failed."),
                    "vision_model": vision_model,
                    "tried_models": query_result.get("tried_models", []),
                    "errors": query_result.get("errors", []),
                }

            out_text = query_result.get("output_text", "") or ""
            model_used = query_result.get("model_used", vision_model)
            model_errors = query_result.get("errors", [])
            selected_token, raw = self._parse_region_selector_output(out_text)
            info = {
                "raw_response": raw,
                "vision_model": model_used or vision_model,
                "vision_model_requested": vision_model,
                "regions": regions,
                "selectable_labels": selectable_labels,
                "selectable_prefix": selectable_prefix,
            }
            if sent_image_path is not None:
                info["sent_image_path"] = sent_image_path
            if sent_unlabeled_image_path is not None:
                info["sent_unlabeled_image_path"] = sent_unlabeled_image_path
            if birdseye_meta is not None:
                info["birdseye"] = birdseye_meta
            print(f"[INFO] Vision model selected token: {selected_token}, response: {raw}")
            if selected_token is None:
                return None, {**info, "reason": "Could not parse contour label from vision model response."}

            m = re.fullmatch(r"([A-Z])(\d+)", selected_token)
            if not m:
                return None, {**info, "reason": f"Parsed invalid label token: {selected_token}"}
            selected_prefix = m.group(1).upper()
            region_id = int(m.group(2))
            if selected_prefix != selectable_prefix:
                # Be tolerant of legacy numeric outputs like {"best_region_id": 0} in C-label mode.
                if (
                    selectable_prefix == "C"
                    and selected_prefix == "R"
                    and (
                        re.search(r'"best_region_id"\s*:\s*\d+', raw) is not None
                        or re.fullmatch(r"\s*\d+\s*", raw) is not None
                    )
                ):
                    selected_prefix = "C"
                else:
                    return None, {
                        **info,
                        "reason": f"Vision model returned label {selected_token}, expected prefix {selectable_prefix}.",
                    }
            if region_id < 0 or region_id >= len(selectable_regions):
                return None, {
                    **info,
                    "reason": f"Vision model returned out-of-range label {selected_token}.",
                }
            selected_label = f"{selected_prefix}{region_id}"
            selected_mask = label_to_mask.get(selected_label)
            if selected_mask is None:
                return None, {**info, "reason": f"Selected label {selected_label} could not be mapped to a screw head mask."}
            return region_id, {
                **info,
                "selected_label": selected_label,
                "selected_region_id": label_to_region_id[selected_label],
                "selected_region_mask": selectable_regions[region_id],
                "selected_mask": selected_mask,
            }
        except Exception as e:
            return None, {"reason": f"Vision model call failed: {e}", "vision_model": vision_model}

    def _parse_region_selector_output(self, text: str) -> Tuple[Optional[str], str]:
        raw = (text or "").strip()
        if not raw:
            return None, "empty response"

        def _norm_label_token(value: Any, default_prefix: str = "R") -> Optional[str]:
            if isinstance(value, int):
                return f"{default_prefix}{value}"
            if not isinstance(value, str):
                return None
            m = re.fullmatch(r"\s*([A-Za-z]?)(\d+)\s*", value)
            if not m:
                return None
            prefix = (m.group(1) or default_prefix).upper()
            return f"{prefix}{int(m.group(2))}"

        try:
            obj = json.loads(raw)
            label = (
                obj.get("best_label")
                or obj.get("best_region_label")
                or obj.get("best_contour_label")
                or obj.get("label")
            )
            norm = _norm_label_token(label, default_prefix="C")
            if norm is not None:
                return norm, raw

            rid = obj.get("best_region_id")
            norm = _norm_label_token(rid, default_prefix="R")
            if norm is not None:
                return norm, raw
        except Exception:
            pass

        m = re.search(r'"best_label"\s*:\s*"([A-Za-z]?)(\d+)"', raw, flags=re.IGNORECASE)
        if m:
            prefix = (m.group(1) or "C").upper()
            return f"{prefix}{int(m.group(2))}", raw

        m = re.search(r'"best_region_id"\s*:\s*"?(?:([A-Za-z]?)(\d+))"?', raw, flags=re.IGNORECASE)
        if m:
            prefix = (m.group(1) or "R").upper()
            return f"{prefix}{int(m.group(2))}", raw

        m = re.search(r"\b([A-Za-z])(\d+)\b", raw, flags=re.IGNORECASE)
        if m:
            return f"{m.group(1).upper()}{int(m.group(2))}", raw

        # Last resort: only accept a plain integer response, defaulting to R* for backward compatibility.
        m = re.fullmatch(r"\s*(\d+)\s*", raw)
        if m:
            return f"R{int(m.group(1))}", raw

        return None, raw

    def select_region_by_locked_label(
        self,
        screws,
        selected_label: str,
    ) -> Tuple[Optional[int], Dict[str, Any]]:
        if not screws:
            return None, {"reason": "No screw regions to select."}

        if isinstance(screws, dict):
            head_regions: List[Dict[str, Any]] = screws.get("overlap_heads", []) or []
            support_regions: List[Dict[str, Any]] = screws.get("combined_contour_maps", []) or []
        else:
            head_regions = screws or []
            support_regions = []

        if not head_regions:
            return None, {"reason": "No selectable screw head regions found."}

        token = str(selected_label or "").strip().upper()

        def _bbox_overlap_area(a_xywh: List[int], b_xywh: List[int]) -> int:
            ax, ay, aw, ah = [int(v) for v in (a_xywh or [0, 0, 0, 0])]
            bx, by, bw, bh = [int(v) for v in (b_xywh or [0, 0, 0, 0])]
            ax2, ay2 = ax + max(aw, 0), ay + max(ah, 0)
            bx2, by2 = bx + max(bw, 0), by + max(bh, 0)
            ix1, iy1 = max(ax, bx), max(ay, by)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            return max(0, ix2 - ix1) * max(0, iy2 - iy1)

        def _map_support_contour_to_head_mask(support_mask: Dict[str, Any]) -> Optional[Tuple[int, Dict[str, Any]]]:
            best_idx: Optional[int] = None
            best_score = -1
            s_seg = support_mask.get("segmentation")
            s_bbox = support_mask.get("bbox", [0, 0, 0, 0])
            for i, h in enumerate(head_regions):
                score = 0
                h_seg = h.get("segmentation")
                if s_seg is not None and h_seg is not None:
                    try:
                        if getattr(s_seg, "shape", None) == getattr(h_seg, "shape", None):
                            score = int(np.count_nonzero(np.logical_and(s_seg.astype(bool), h_seg.astype(bool))))
                    except Exception:
                        score = 0
                    if score <= 0:
                        try:
                            if masks_overlap(h_seg, s_seg):
                                score = 1
                        except Exception:
                            pass
                if score <= 0:
                    score = _bbox_overlap_area(s_bbox, h.get("bbox", [0, 0, 0, 0]))
                if score > best_score:
                    best_score = score
                    best_idx = i

            if best_idx is None or best_score <= 0:
                return None
            return best_idx, head_regions[best_idx]

        if len(head_regions) == 1 and not re.fullmatch(r"[A-Z]\d+", token or ""):
            return 0, {
                "reason": "Locked selection fell back to the only visible screw region.",
                "selected_label": "R0",
                "selected_region_id": 0,
                "selected_region_mask": head_regions[0],
                "selected_mask": head_regions[0],
                "selectable_labels": ["R0"],
                "selectable_prefix": "R",
            }

        m = re.fullmatch(r"([A-Z])(\d+)", token)
        if not m:
            return None, {"reason": f"Invalid locked selection label: {selected_label}"}

        prefix = m.group(1).upper()
        region_id = int(m.group(2))

        if prefix == "R":
            selectable_labels = [f"R{i}" for i in range(len(head_regions))]
            if region_id < 0 or region_id >= len(head_regions):
                return None, {"reason": f"Locked selection {token} is out of range for current head regions."}
            selected_mask = head_regions[region_id]
            return region_id, {
                "reason": "Resolved locked head-region selection.",
                "selected_label": token,
                "selected_region_id": region_id,
                "selected_region_mask": selected_mask,
                "selected_mask": selected_mask,
                "selectable_labels": selectable_labels,
                "selectable_prefix": "R",
            }

        if prefix == "C":
            selectable_labels = [f"C{i}" for i in range(len(support_regions))]
            if region_id < 0 or region_id >= len(support_regions):
                return None, {"reason": f"Locked selection {token} is out of range for current contour regions."}
            selected_region_mask = support_regions[region_id]
            mapped = _map_support_contour_to_head_mask(selected_region_mask)
            if mapped is None:
                return None, {"reason": f"Locked contour {token} could not be mapped to a screw head."}
            _, selected_mask = mapped
            return region_id, {
                "reason": "Resolved locked contour selection.",
                "selected_label": token,
                "selected_region_id": region_id,
                "selected_region_mask": selected_region_mask,
                "selected_mask": selected_mask,
                "selectable_labels": selectable_labels,
                "selectable_prefix": "C",
            }

        return None, {"reason": f"Unsupported locked selection prefix in label: {selected_label}"}

    def llm_detect_and_move_to_screw(
        self,
        request: str,
        *,
        camera: str,
        z_offset: float,
        detection_mode: Optional[str] = None,
        orientation_profile: str = "pickup",
        planning_tcp_name: Optional[str] = None,
        vision_model: Optional[str] = None,
        region_selector: Optional[Callable[..., Tuple[Optional[int], Dict[str, Any]]]] = None,
        locked_selection_label: Optional[str] = None,
        llm_target_mode: str = "auto",
        llm_birdseye: bool = False,
        llm_birdseye_options: Optional[Dict[str, Any]] = None,
        debug: bool = False,
    ) -> Tuple[bool, Dict[str, Any]]:
        """LLM-assisted screw selection + motion."""
        self.confirm_drill_tcp(context=f"{camera} LLM screw detection and planning")
        depth, color, intr, T_cam = self.camera_helper.get_rgbd_and_intrinsics(camera)
        T_base_tcp = self.handles.arm.get_T_base_tcp()

        selector_cb = region_selector
        if selector_cb is None:
            model_name = str(vision_model or self.config.get('llm', 'openai_query', 'default_model', default="gpt-4.1"))
            selector_cb = lambda **kwargs: self.select_region_with_vision_llm(  # noqa: E731
                vision_model=model_name,
                **kwargs,
            )
        if llm_birdseye:
            birdseye_cfg = dict(self.config.get('llm', 'vision_selector', 'birdseye', default={}) or {})
            birdseye_cfg.update(dict(llm_birdseye_options or {}))
            birdseye_context: Dict[str, Any] = {
                "enabled": True,
                "camera": camera,
                "intr": intr,
                "T_cam": T_cam,
                "T_base_tcp": T_base_tcp,
                "plane_z_base": float(birdseye_cfg.get("plane_z_base", 0.0)),
                "pixels_per_meter": float(birdseye_cfg.get("pixels_per_meter", 1200.0)),
                "padding_px": int(birdseye_cfg.get("padding_px", 16)),
                "margin_m": float(birdseye_cfg.get("margin_m", 0.0)),
            }
            base_selector_cb = selector_cb

            def _selector_with_birdseye(**kwargs):
                kwargs.setdefault("birdseye_context", birdseye_context)
                return base_selector_cb(**kwargs)

            selector_cb = _selector_with_birdseye

        heads, selected_prompt, available = self.llm_detect_requested_heads(
            color,
            request,
            region_selector=selector_cb,
            locked_selection_label=locked_selection_label,
            target_mode=llm_target_mode,
            depth=depth,
            intr=intr,
            camera=camera,
            detection_mode=detection_mode,
            debug=debug,
        )
        if not heads:
            return False, {
                "reason": "No screw matching request was found.",
                "selected_prompt": selected_prompt,
                "available_prompts": available,
            }

        targets = self.compute_targets(
            color,
            label=f"requested:{selected_prompt}",
            masks=heads,
            depth=depth,
            intr=intr,
        )
        if not targets:
            return False, {
                "reason": "Detected requested screw, but no valid depth targets were available.",
                "selected_prompt": selected_prompt,
                "available_prompts": available,
            }

        candidates = self.motion_planner.compute_approach_poses(
            T_base_tcp,
            targets,
            T_cam,
            camera,
            z_offset,
            ignore_rotation=True,
            orientation_profile=orientation_profile,
            planning_tcp_name=planning_tcp_name,
        )
        best_candidate, best_cost = self.motion_planner.select_best_candidate(
            candidates,
            ignore_close=False,
        )
        if debug:
            debug_candidate = best_candidate or self.motion_planner.select_debug_candidate(candidates)
            if best_candidate is None and debug_candidate is not None:
                print("[DEBUG] Showing closest unreachable candidate in 3D frame view.")
            self.screw_detector.show_reach_plan_3d_debug(debug_candidate)
        if best_candidate is None:
            return False, {
                "reason": "No reachable pose was found for the selected screw.",
                "selected_prompt": selected_prompt,
                "available_prompts": available,
            }
        success = self.motion_planner.move_to_best_reachable(
            candidates,
            speed=self.config.default_speed,
            debug=debug,
            selected_candidate=best_candidate,
        )
        selected_target_base = None
        selected_target_cam = None
        selected_pixel = None
        if best_candidate is not None:
            target_base = np.asarray(best_candidate.get("target_base", []), dtype=float).reshape(-1)
            target_cam = np.asarray(best_candidate.get("target_cam", []), dtype=float).reshape(-1)
            if target_base.size >= 3:
                selected_target_base = [float(v) for v in target_base[:3]]
            if target_cam.size >= 3:
                selected_target_cam = [float(v) for v in target_cam[:3]]
            try:
                selected_pixel = [
                    float(best_candidate.get("u")),
                    float(best_candidate.get("v")),
                ]
            except Exception:
                selected_pixel = None
        return bool(success), {
            "selected_prompt": selected_prompt,
            "available_prompts": available,
            "target_count": len(targets),
            "llm_target_mode": str(llm_target_mode),
            "selected_target_base": selected_target_base,
            "selected_target_cam": selected_target_cam,
            "selected_pixel": selected_pixel,
            "selected_candidate_distance_m": (float(best_cost) if np.isfinite(best_cost) else None),
        }

    def refine_pickup_target_locally(
        self,
        *,
        target_base: Any,
        z_offset: float,
        camera: str = "arm",
        debug: bool = False,
        debug_gui: bool = False,
        debug_gui_wait_ms: int = 1,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Refine pickup using only nearby screw-head detections around the projected pickup point."""
        cam = str(camera or "").strip().lower()
        if cam != "arm":
            return False, {"reason": "Local pickup refinement currently supports only the arm camera."}

        self.confirm_drill_tcp(context="local screw pickup refinement planning")
        verify_cfg = self.config.get('pickup_verification', default={}) or {}
        conf_map = self.config.confidence_for_camera(cam)
        roi_half_size_px = max(24, int(verify_cfg.get("refine_roi_half_size_px", 150)))
        max_pick_distance_px = float(verify_cfg.get("refine_max_pick_distance_px", roi_half_size_px))
        head_conf = float(verify_cfg.get("refine_head_confidence", conf_map.get("screw_head", 0.4)))
        debug_gui_wait_ms = max(1, int(debug_gui_wait_ms))

        depth, color, intr, T_cam = self.camera_helper.get_rgbd_and_intrinsics(cam)
        T_base_tcp = self.handles.arm.get_T_base_tcp()
        proj_info = self.camera_helper.project_base_point_to_image(
            target_base,
            camera_type=cam,
            T_base_tcp=T_base_tcp,
        )
        if not proj_info.get("ok"):
            return False, {
                "reason": f"Failed to project coarse pickup target into arm image: {proj_info.get('reason', 'unknown')}",
                "projection": proj_info,
            }

        H, W = color.shape[:2]
        u_proj, v_proj = proj_info["pixel"]
        x_c = int(round(u_proj))
        y_c = int(round(v_proj))
        x1 = max(0, x_c - roi_half_size_px)
        y1 = max(0, y_c - roi_half_size_px)
        x2 = min(W, x_c + roi_half_size_px)
        y2 = min(H, y_c + roi_half_size_px)
        if x2 <= x1 or y2 <= y1:
            return False, {
                "reason": "Projected pickup ROI is outside the current arm image.",
                "projection": proj_info,
            }

        masked_color = np.zeros_like(color)
        masked_color[y1:y2, x1:x2] = color[y1:y2, x1:x2]
        screw_result = self.screw_detector.detect_screws(
            masked_color,
            camera_type=cam,
            heads_conf=head_conf,
            debug=False,
        )
        if isinstance(screw_result, dict):
            head_masks = screw_result.get("overlap_heads", []) or []
        else:
            head_masks = screw_result or []

        nearby_heads: List[Dict[str, Any]] = []
        scored_heads: List[Tuple[float, Dict[str, Any], Tuple[float, float]]] = []
        for m in head_masks:
            try:
                cx, cy = mask_centroid(m["segmentation"])
            except Exception:
                bbox = [int(v) for v in m.get("bbox", [0, 0, 0, 0])]
                cx = float(bbox[0] + max(bbox[2], 0) / 2.0)
                cy = float(bbox[1] + max(bbox[3], 0) / 2.0)

            if not (x1 <= cx < x2 and y1 <= cy < y2):
                continue

            dist_px = float(math.hypot(cx - float(u_proj), cy - float(v_proj)))
            if dist_px > max_pick_distance_px:
                continue

            nearby_heads.append(m)
            scored_heads.append((dist_px, m, (cx, cy)))

        if not scored_heads:
            return False, {
                "reason": "No nearby screw-head detections found inside the local pickup ROI.",
                "projection": proj_info,
                "roi_xyxy": [int(x1), int(y1), int(x2), int(y2)],
                "detected_head_count": len(head_masks),
            }

        scored_heads.sort(key=lambda item: item[0])
        best_dist_px, best_mask, best_centroid = scored_heads[0]

        targets = self.compute_targets(
            color,
            label="pickup_refine_local",
            masks=[best_mask],
            depth=depth,
            intr=intr,
        )
        if not targets:
            return False, {
                "reason": "Local screw-head refinement found a mask but no valid depth target.",
                "projection": proj_info,
                "roi_xyxy": [int(x1), int(y1), int(x2), int(y2)],
            }

        candidates = self.motion_planner.compute_approach_poses(
            T_base_tcp,
            targets,
            T_cam,
            cam,
            z_offset=z_offset,
            ignore_rotation=True,
            orientation_profile="pickup",
        )
        best_candidate, best_cost = self.motion_planner.select_best_candidate(
            candidates,
            ignore_close=False,
        )
        if best_candidate is None:
            if debug:
                debug_candidate = self.motion_planner.select_debug_candidate(candidates)
                if debug_candidate is not None:
                    print("[DEBUG] Showing closest unreachable candidate in 3D frame view.")
                    self.screw_detector.show_reach_plan_3d_debug(debug_candidate)
            return False, {
                "reason": "No reachable refined pickup pose was found from the local ROI detection.",
                "projection": proj_info,
                "roi_xyxy": [int(x1), int(y1), int(x2), int(y2)],
            }

        if debug:
            self.screw_detector.show_reach_plan_3d_debug(best_candidate)

        curr_p = np.asarray(T_base_tcp[:3, 3], dtype=float).reshape(3,)
        best_approach = np.asarray(best_candidate.get("approach", curr_p), dtype=float).reshape(-1)
        move_delta = None
        move_delta_norm = None
        if best_approach.size >= 3:
            delta_vec = best_approach[:3] - curr_p
            move_delta = [float(delta_vec[0]), float(delta_vec[1]), float(delta_vec[2])]
            move_delta_norm = float(np.linalg.norm(delta_vec))

        vis = None
        if debug:
            try:
                vis = color.copy()
                shade = np.zeros_like(vis)
                shade[:] = (0, 0, 0)
                vis = cv2.addWeighted(vis, 0.35, shade, 0.65, 0.0)
                vis[y1:y2, x1:x2] = color[y1:y2, x1:x2]
                cv2.rectangle(vis, (x1, y1), (x2 - 1, y2 - 1), (255, 255, 0), 2, cv2.LINE_AA)
                cv2.drawMarker(
                    vis,
                    (int(round(u_proj)), int(round(v_proj))),
                    (255, 255, 255),
                    cv2.MARKER_CROSS,
                    18,
                    2,
                    cv2.LINE_AA,
                )
                for _, mask, (cx, cy) in scored_heads:
                    contour = mask.get("contour")
                    if contour is not None and len(contour) > 0:
                        cv2.drawContours(vis, [contour], -1, (0, 165, 255), 2, cv2.LINE_AA)
                    cv2.circle(vis, (int(round(cx)), int(round(cy))), 5, (0, 165, 255), -1, cv2.LINE_AA)
                contour = best_mask.get("contour")
                if contour is not None and len(contour) > 0:
                    cv2.drawContours(vis, [contour], -1, (0, 255, 0), 3, cv2.LINE_AA)
                cv2.circle(
                    vis,
                    (int(round(best_centroid[0])), int(round(best_centroid[1]))),
                    7,
                    (0, 255, 0),
                    -1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    vis,
                    f"refine dist={best_dist_px:.1f}px",
                    (x1 + 8, max(24, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                debug_path = "data/image_samples/pickup_refine_local.png"
                cv2.imwrite(debug_path, vis)
                print(f"[DEBUG] Saved local pickup refine image to: {debug_path}")
            except Exception as e:
                print(f"[WARN] Local pickup refine debug visualization failed: {e}")
                vis = None

        if debug_gui:
            try:
                if vis is None:
                    vis = color.copy()
                    shade = np.zeros_like(vis)
                    shade[:] = (0, 0, 0)
                    vis = cv2.addWeighted(vis, 0.35, shade, 0.65, 0.0)
                    vis[y1:y2, x1:x2] = color[y1:y2, x1:x2]
                    cv2.rectangle(vis, (x1, y1), (x2 - 1, y2 - 1), (255, 255, 0), 2, cv2.LINE_AA)
                    cv2.drawMarker(
                        vis,
                        (int(round(u_proj)), int(round(v_proj))),
                        (255, 255, 255),
                        cv2.MARKER_CROSS,
                        18,
                        2,
                        cv2.LINE_AA,
                    )
                    for _, mask, (cx, cy) in scored_heads:
                        contour = mask.get("contour")
                        if contour is not None and len(contour) > 0:
                            cv2.drawContours(vis, [contour], -1, (0, 165, 255), 2, cv2.LINE_AA)
                        cv2.circle(vis, (int(round(cx)), int(round(cy))), 5, (0, 165, 255), -1, cv2.LINE_AA)
                    contour = best_mask.get("contour")
                    if contour is not None and len(contour) > 0:
                        cv2.drawContours(vis, [contour], -1, (0, 255, 0), 3, cv2.LINE_AA)
                    cv2.circle(
                        vis,
                        (int(round(best_centroid[0])), int(round(best_centroid[1]))),
                        7,
                        (0, 255, 0),
                        -1,
                        cv2.LINE_AA,
                    )

                roi_crop = color[y1:y2, x1:x2].copy()
                if roi_crop.size == 0:
                    roi_crop = np.zeros((240, 240, 3), dtype=np.uint8)
                roi_h, roi_w = roi_crop.shape[:2]
                rel_u = int(np.clip(round(u_proj - x1), 0, max(roi_w - 1, 0)))
                rel_v = int(np.clip(round(v_proj - y1), 0, max(roi_h - 1, 0)))
                cv2.drawMarker(
                    roi_crop,
                    (rel_u, rel_v),
                    (255, 255, 255),
                    cv2.MARKER_CROSS,
                    18,
                    2,
                    cv2.LINE_AA,
                )
                for _, mask, (cx, cy) in scored_heads:
                    contour = mask.get("contour")
                    if contour is not None and len(contour) > 0:
                        contour_local = contour.copy()
                        contour_local[:, 0, 0] = contour_local[:, 0, 0] - x1
                        contour_local[:, 0, 1] = contour_local[:, 0, 1] - y1
                        cv2.drawContours(roi_crop, [contour_local], -1, (0, 165, 255), 2, cv2.LINE_AA)
                    cv2.circle(
                        roi_crop,
                        (int(round(cx - x1)), int(round(cy - y1))),
                        5,
                        (0, 165, 255),
                        -1,
                        cv2.LINE_AA,
                    )
                contour = best_mask.get("contour")
                if contour is not None and len(contour) > 0:
                    contour_local = contour.copy()
                    contour_local[:, 0, 0] = contour_local[:, 0, 0] - x1
                    contour_local[:, 0, 1] = contour_local[:, 0, 1] - y1
                    cv2.drawContours(roi_crop, [contour_local], -1, (0, 255, 0), 3, cv2.LINE_AA)

                info_panel = np.full((180, 520, 3), 24, dtype=np.uint8)
                lines = [
                    f"status: {'refined' if best_candidate is not None else 'failed'}",
                    f"projected: ({u_proj:.1f}, {v_proj:.1f})  roi: [{x1}, {y1}] - [{x2}, {y2}]",
                    f"heads in roi: {len(scored_heads)} / detected: {len(head_masks)}",
                    f"selected centroid: ({best_centroid[0]:.1f}, {best_centroid[1]:.1f})",
                    f"pixel dist: {best_dist_px:.1f}px  move dist: {float(best_cost):.4f}m",
                    (
                        f"move delta: dx={move_delta[0]:+.4f} dy={move_delta[1]:+.4f} dz={move_delta[2]:+.4f} m"
                        if move_delta is not None
                        else "move delta: unavailable"
                    ),
                ]
                y_text = 32
                for line in lines:
                    cv2.putText(
                        info_panel,
                        line,
                        (14, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.58,
                        (235, 235, 235),
                        1,
                        cv2.LINE_AA,
                    )
                    y_text += 28

                panel_h = 420
                panel_w = 520

                def _fit_panel(image_bgr: np.ndarray, title: str) -> np.ndarray:
                    panel = np.full((panel_h, panel_w, 3), 16, dtype=np.uint8)
                    cv2.rectangle(panel, (0, 0), (panel_w - 1, panel_h - 1), (70, 70, 70), 2, cv2.LINE_AA)
                    cv2.rectangle(panel, (0, 0), (panel_w - 1, 40), (38, 38, 38), -1)
                    cv2.putText(panel, title, (14, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (240, 240, 240), 2, cv2.LINE_AA)
                    if image_bgr is None or image_bgr.size == 0:
                        return panel
                    src_h, src_w = image_bgr.shape[:2]
                    scale = min((panel_w - 16) / max(1, src_w), (panel_h - 56) / max(1, src_h))
                    dst_w = max(1, int(round(src_w * scale)))
                    dst_h = max(1, int(round(src_h * scale)))
                    resized = cv2.resize(image_bgr, (dst_w, dst_h), interpolation=cv2.INTER_AREA)
                    x0 = (panel_w - dst_w) // 2
                    y0 = 48 + ((panel_h - 56) - dst_h) // 2
                    panel[y0:y0 + dst_h, x0:x0 + dst_w] = resized
                    return panel

                left_panel = _fit_panel(vis, "Refine Overview")
                right_panel = _fit_panel(roi_crop, "ROI Head Detection")
                gap = 14
                dashboard_h = panel_h + gap + info_panel.shape[0] + gap
                dashboard_w = panel_w * 2 + gap * 3
                dashboard = np.full((dashboard_h, dashboard_w, 3), 10, dtype=np.uint8)
                dashboard[gap:gap + panel_h, gap:gap + panel_w] = left_panel
                dashboard[gap:gap + panel_h, gap * 2 + panel_w:gap * 2 + panel_w * 2] = right_panel
                dashboard[gap + panel_h + gap:gap + panel_h + gap + info_panel.shape[0], gap:gap + info_panel.shape[1]] = info_panel
                window_name = "pickup_refine_local_debug"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.imshow(window_name, dashboard)
                # Pump the GUI a couple of times so the frame is actually drawn
                # before any subsequent robot motion blocks the thread.
                cv2.waitKey(1)
                cv2.imshow(window_name, dashboard)
                cv2.waitKey(1)

                if debug:
                    print("[DEBUG] Close the pickup_refine_local_debug window or press q/Esc to continue.")
                    while True:
                        try:
                            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                                break
                        except Exception:
                            break
                        key = cv2.waitKey(50) & 0xFF
                        if key in (27, ord('q'), ord('Q')):
                            break
                    try:
                        cv2.destroyWindow(window_name)
                        cv2.waitKey(1)
                    except Exception:
                        pass
                else:
                    cv2.waitKey(debug_gui_wait_ms)
            except Exception as e:
                print(f"[WARN] Local pickup refine debug GUI failed: {e}")

        self.handles.arm.moveL(best_candidate["pose"], speed=self.config.default_speed)

        refined_target_base = np.asarray(best_candidate.get("target_base", []), dtype=float).reshape(-1)
        return True, {
            "projection": proj_info,
            "roi_xyxy": [int(x1), int(y1), int(x2), int(y2)],
            "detected_head_count": len(head_masks),
            "nearby_head_count": len(scored_heads),
            "selected_head_centroid": [float(best_centroid[0]), float(best_centroid[1])],
            "selected_head_distance_px": float(best_dist_px),
            "refined_target_base": (
                [float(v) for v in refined_target_base[:3]]
                if refined_target_base.size >= 3
                else None
            ),
            "move_delta_base_m": move_delta,
            "move_delta_norm_m": move_delta_norm,
            "selected_candidate_distance_m": (float(best_cost) if np.isfinite(best_cost) else None),
        }

    def llm_detect_requested_heads(
        self,
        color: np.ndarray,
        request: str,
        *,
        region_selector: Callable[..., Tuple[Optional[int], Dict[str, Any]]],
        locked_selection_label: Optional[str] = None,
        target_mode: str = "auto",
        depth: Optional[np.ndarray] = None,
        intr: Any = None,
        camera: Optional[str] = None,
        detection_mode: Optional[str] = None,
        debug: bool = False,
    ) -> Tuple[List[Dict[str, Any]], str, List[str]]:
        """Detect screw heads and use an injected LLM region selector to choose one target head."""
        target_mode_norm = str(target_mode or "auto").strip().lower()
        if target_mode_norm not in ("auto", "head", "center"):
            print(f"[WARN] Unknown llm target mode '{target_mode}', falling back to 'auto'")
            target_mode_norm = "auto"

        def _mask_without_target_hint(mask: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
            if mask is None:
                return None
            # Copy so center-target mode does not mutate shared detection results.
            m = dict(mask)
            m.pop("target_pixel", None)
            m.pop("target_pixel_source", None)
            return m

        screws = self.screw_detector.detect_screws(
            color,
            camera_type=camera,
            detection_mode=detection_mode,
            debug=debug,
        )
        if not screws:
            return [], "no_screws_detected", []

        if isinstance(screws, dict):
            head_regions = screws.get("overlap_heads", []) or []
        else:
            head_regions = screws

        if locked_selection_label:
            region_id, region_info = self.select_region_by_locked_label(screws, locked_selection_label)
        else:
            selector_kwargs = dict(
                color=color,
                screws=screws,
                request=request,
                debug=debug,
            )
            if depth is not None:
                selector_kwargs["depth"] = depth
            if intr is not None:
                selector_kwargs["intr"] = intr
            try:
                region_id, region_info = region_selector(**selector_kwargs)
            except TypeError:
                # Backward compatibility for custom selectors that don't accept depth/intr.
                selector_kwargs.pop("depth", None)
                selector_kwargs.pop("intr", None)
                region_id, region_info = region_selector(**selector_kwargs)

        if region_id is None:
            # Conservative fallback only when a single candidate exists.
            if len(head_regions) == 1:
                print("[WARN] Vision selector unavailable; using the only visible screw region.")
                if target_mode_norm == "center":
                    selected_heads = [_mask_without_target_hint(head_regions[0])]
                else:
                    selected_heads = [head_regions[0]]
                selected_prompt = "vision_fallback_single_region"
                available = ["R0"]
            else:
                return [], "vision_selector_failed", [region_info.get("reason", "unknown")]
        else:
            selected_label = region_info.get("selected_label", f"R{region_id}")
            selected_mask = region_info.get("selected_mask")
            selected_region_mask = region_info.get("selected_region_mask")

            is_c_label = isinstance(selected_label, str) and selected_label.upper().startswith("C")
            if target_mode_norm == "head":
                if selected_mask is not None:
                    selected_heads = [selected_mask]
                else:
                    selected_heads = [head_regions[region_id]]
            elif target_mode_norm == "center":
                center_mask = selected_region_mask if (is_c_label and selected_region_mask is not None) else selected_mask
                if center_mask is None:
                    center_mask = head_regions[region_id]
                selected_heads = [_mask_without_target_hint(center_mask)]
            else:
                # auto: C* -> contour center (screw center context), R* -> head mask
                if is_c_label and selected_region_mask is not None:
                    selected_heads = [_mask_without_target_hint(selected_region_mask)]
                elif selected_mask is not None:
                    selected_heads = [selected_mask]
                else:
                    selected_heads = [head_regions[region_id]]
            selected_prompt = selected_label
            available = region_info.get("selectable_labels", [f"R{i}" for i in range(len(head_regions))])

        if debug:
            try:
                dbg_masks = selected_heads if region_id is not None else head_regions
                vis = draw_mask_debug(
                    color,
                    dbg_masks,
                    output_path="data/image_samples/sam3_llm_requested_screw.png",
                    category_colors={m.get("category", "obj"): (0, 255, 0) for m in dbg_masks},
                )
                highlight_masks = selected_heads if selected_heads else dbg_masks
                for idx, m in enumerate(highlight_masks):
                    contour = m.get("contour")
                    if contour is not None and len(contour) > 0:
                        cv2.drawContours(vis, [contour], -1, (0, 255, 255), 3, cv2.LINE_AA)
                    try:
                        cx, cy = mask_centroid(m["segmentation"])
                        cv2.drawMarker(
                            vis,
                            (int(round(cx)), int(round(cy))),
                            (255, 255, 255),
                            cv2.MARKER_CROSS,
                            18,
                            2,
                            cv2.LINE_AA,
                        )
                        cv2.circle(
                            vis,
                            (int(round(cx)), int(round(cy))),
                            8,
                            (0, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        label = str(selected_prompt) if idx == 0 else f"{selected_prompt}:{idx}"
                        cv2.putText(
                            vis,
                            f"center {label}",
                            (int(round(cx)) + 10, int(round(cy)) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                    except Exception:
                        pass
                cv2.imwrite("data/image_samples/sam3_llm_requested_screw.png", vis)
            except Exception as e:
                print(f"[WARN] Debug visualization failed: {e}")

        return selected_heads, selected_prompt, available

    def move_to_target_mark_with_status(
        self,
        use_servo: bool = False,
        speed: Optional[float] = None,
        debug: bool = False,
        use_fixed_camera: bool = True,
    ) -> Tuple[bool, Dict[str, Any]]:

        print("[STEP] Moving to target mark...")
        clearance = self.config.offsets['target_mark_clearance']
        status_info: Dict[str, Any] = {
            "used_fixed_camera": bool(use_fixed_camera),
            "fixed_camera_success": False,
            "used_arm_camera": False,
            "arm_camera_success": False,
            "used_servo": bool(use_servo),
        }

        success = False
        if use_fixed_camera:
            fixed_success = self.detect_and_move_to_target_mark(
                camera_type="fixed",
                z_offset=self.config.screw_length + clearance,
                debug=debug,
            )
            status_info["fixed_camera_success"] = bool(fixed_success)
            success = bool(fixed_success)
            if not fixed_success:
                print(
                    "[INFO] Fixed camera did not produce a reachable target-mark "
                    "pose; skipping arm-camera refinement."
                )
                return False, status_info
        else:
            print("[INFO] Skipping fixed-camera reacquisition; refining target mark from current pose.")

        if self.config.get('features', 'refine_positioning_target', default=False):
            print('Refining Target Mark')
            camera_type  = 'arm'
            refined_target_mark_clearance = float(
                self.config.get(
                    'motion',
                    'offsets',
                    'refined_target_mark_clearance',
                    default=0.0,
                )
            )
            refined_target_mark_z_offset = self.config.screw_length + refined_target_mark_clearance
            status_info["arm_refine_clearance"] = refined_target_mark_clearance
            status_info["arm_refine_z_offset"] = refined_target_mark_z_offset
            if not use_servo: #dont do approach if in normal mode
                status_info["used_arm_camera"] = True
                arm_success = self.detect_and_move_to_target_mark(
                    camera_type=camera_type,
                    heads_conf=0.4,
                    z_offset=refined_target_mark_z_offset,
                    debug=debug,
                )
                status_info["arm_camera_success"] = bool(arm_success)
                success = bool(arm_success)
        
            if use_servo:
                print("[INFO] Using visual servo (detection + target selection in servo loop).")
                status_info["used_arm_camera"] = True
                
                def target_pose_fn() -> Optional[List[float]]:
                    prompt = self.config.prompts['target_marker']
                    conf = self.config.confidence.get('target_marker', 0.2)

                    T_base_tcp = self.handles.arm.get_T_base_tcp()
                    depth, color, intr, T_cam = self.camera_helper.get_rgbd_and_intrinsics(camera_type)         
                    # masks = self.screw_detector.detect_screws(color, debug=debug)
                    masks = self.generic_cycle.detect_generic(prompt=prompt,color=color,conf=conf) #detect just screw head for faster inference
                    if not masks:
                        print(f"No target mark detected from {camera_type} camera")
                        return False
                    
                    targets = self.compute_targets(color,prompt,masks, depth, intr)
                    candidates = self.motion_planner.compute_approach_poses(T_base_tcp,
                        targets, T_cam, camera_type, refined_target_mark_z_offset , ignore_rotation = True
                    )
        
                    if not candidates:
                        print("[SERVO] No candidate poses from targets.")
                        return None

                    # 5) Select best candidate given current robot state
                    best_cand, best_cost = self.motion_planner.select_best_candidate(
                        candidates,
                        ignore_close=False,
                    )
                    if best_cand is None:
                        print("[SERVO] No reachable candidates.")
                        return None

                    return best_cand["pose"]

                status = self.motion_planner.servo_towards_dynamic(
                    target_pose_fn=target_pose_fn,
                    speed=speed if speed is not None else self.config.default_speed,
                )
                success = False
                print('servo finish status',status)
                if 'target_reached' in status:
                    success = True
                status_info["arm_camera_success"] = bool(success)
                
                return success, status_info
        

    
            # print('Refining Target Mark')
            # refine_offset = self.config.get('motion', 'offsets', 'refine_approach', default=0.1)
            # success = self.detect_and_move_to_target_mark( camera_type="arm",heads_conf = 0.4, z_offset=self.config.screw_length ,debug=debug)
        
        return success, status_info

    def move_to_target_mark(
        self,
        use_servo: bool = False,
        speed: Optional[float] = None,
        debug: bool = False,
        use_fixed_camera: bool = True,
    ) -> bool:
        success, _status_info = self.move_to_target_mark_with_status(
            use_servo=use_servo,
            speed=speed,
            debug=debug,
            use_fixed_camera=use_fixed_camera,
        )
        return success
        
    def detect_and_move_to_target_mark(self,camera_type, z_offset,heads_conf=None,target_marks_conf=None, debug: bool= False):
        """Detect and move to target mark."""

        self.confirm_drill_tcp(context=f"{camera_type} target-mark detection and planning")
        depth, color, intr, T_cam = self.camera_helper.get_rgbd_and_intrinsics(camera_type)
        T_base_tcp = self.handles.arm.get_T_base_tcp()
        masks = self.screw_detector.detect_target_marks(
            color,
            camera_type=camera_type,
            heads_conf=heads_conf,
            target_marks_conf=target_marks_conf,
            debug=debug,
        )
        if not masks:
            print("[WARN] No target marks detected")
            return False
            
        prompt = self.config.prompts['target_marker']
        targets = self.compute_targets(color,prompt,masks, depth, intr)
        if len(targets) == 0:
            print("[WARN] No valid 3D targets for target marks")
            return False
 
        
        candidates = self.motion_planner.compute_approach_poses(T_base_tcp,
            targets, T_cam, camera_type, z_offset, ignore_rotation = True
        )
        best_candidate = None
        if debug:
            best_candidate, _ = self.motion_planner.select_best_candidate(
                candidates,
                ignore_close=False,
            )
            debug_candidate = best_candidate or self.motion_planner.select_debug_candidate(candidates)
            if best_candidate is None and debug_candidate is not None:
                print("[DEBUG] Showing closest unreachable target-mark candidate in 3D frame view.")
            self.screw_detector.show_reach_plan_3d_debug(debug_candidate)
            if best_candidate is None:
                print("[WARN] Target marks were detected, but none produced a reachable pose.")
                return False
        
        return self.motion_planner.move_to_best_reachable(
            candidates,
            speed=self.config.default_speed,
            debug=debug,
            selected_candidate=best_candidate,
        )
    
    # def compute_targets(self,color, prompt_label, masks, depth, intr) -> List[Dict]:
    #     """Compute 3D targets from masks."""
    #     targets = []
    #     for mask in masks:
    #         target_info = self.detector.compute_target_from_mask(
    #             mask, depth, intr, depth_scale=0.001
    #         )
    #         if target_info is None:
    #             continue
    #         target_info["image"] = color
    #         target_info["prompt"] = prompt_label
    #         if target_info:
    #             targets.append(target_info)

    #     return targets
    def compute_targets(self, color, label, masks, depth, intr) -> List[Dict]:
        """Compute 3D targets from masks."""
        targets = []
        for mask in masks:
            target_info = self.detector.compute_target_from_mask(
                mask, depth, intr, depth_scale=0.001
            )
            if target_info is None:
                continue

            target_info["image"] = color
            target_info["prompt"] = label
            if mask.get("angle_deg") is not None:
                target_info["angle_cam"] = float(mask["angle_deg"])
            stem_angle = mask.get("stem_angle_deg")
            if stem_angle is None and mask.get("category") == self.config.prompts.get("screw_stem"):
                stem_angle = mask.get("angle_deg")
            if stem_angle is not None:
                target_info["stem_angle_cam"] = float(stem_angle)
            if mask.get("stem_direction_cam") is not None:
                target_info["stem_direction_cam"] = list(mask["stem_direction_cam"])
            targets.append(target_info)

        return targets

    def run_pickup_program(self):
        """Execute URP pickup program."""
        timing = self.config.get('hardware', 'timing', default={})
        program_name = self.config.get('programs', 'pickup', default='screw_pickup.urp')

        self.handles.arm_dash.connect()
        time.sleep(float(timing.get('dashboard_connect_delay', 0.1)))

        load_response = self.handles.arm_dash.load_urp(program_name)
        print(f"[DASHBOARD] load_urp response: {load_response}")
        time.sleep(max(0.0, float(timing.get('urp_load_delay', 1.0))))

        play_response = self.handles.arm_dash.play()
        print(f"[DASHBOARD] play response: {play_response}")
        self.handles.arm_dash.wait_for_program(
            start_timeout=float(timing.get('urp_start_timeout', 10.0)),
            finish_timeout=float(timing.get('urp_finish_timeout', 300.0)),
            poll_interval=float(timing.get('urp_poll_interval', 0.25)),
        )

        self.confirm_drill_tcp(context=f"completed {program_name}")

        time.sleep(timing.get('post_pickup_wait', 3.0))


    def _pickup_mask_bbox(
        self,
        mask: np.ndarray,
        *,
        pad_px: int = 0,
        shape_hw: Optional[Tuple[int, int]] = None,
    ) -> Tuple[int, int, int, int]:
        mask_u8 = np.squeeze(mask).astype(np.uint8)
        ys, xs = np.where(mask_u8 > 0)
        if xs.size == 0 or ys.size == 0:
            raise ValueError("Pickup verification mask is empty")

        if shape_hw is None:
            h, w = mask_u8.shape[:2]
        else:
            h, w = int(shape_hw[0]), int(shape_hw[1])

        x0 = max(0, int(xs.min()) - int(pad_px))
        y0 = max(0, int(ys.min()) - int(pad_px))
        x1 = min(w, int(xs.max()) + 1 + int(pad_px))
        y1 = min(h, int(ys.max()) + 1 + int(pad_px))
        if x1 <= x0 or y1 <= y0:
            raise ValueError("Pickup verification mask produced an invalid ROI")
        return x0, y0, x1, y1

    def _draw_pickup_contour_panel(
        self,
        image_rgb: np.ndarray,
        candidates: List[Dict[str, Any]],
        best: Optional[Dict[str, Any]],
        *,
        title: str,
        passed: Optional[bool] = None,
        roi_mask: Optional[np.ndarray] = None,
        extra_lines: Optional[List[str]] = None,
    ) -> np.ndarray:
        panel = cv2.cvtColor(np.asarray(image_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)

        if isinstance(roi_mask, np.ndarray):
            roi_u8 = np.squeeze(roi_mask).astype(np.uint8)
            try:
                contours, _ = cv2.findContours((roi_u8 > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(panel, contours, -1, (0, 255, 255), 1, cv2.LINE_AA)
            except Exception:
                pass

        for item in candidates:
            contour = item.get("contour")
            if contour is None:
                continue
            color = (110, 110, 110)
            thick = 1
            if item is best:
                if passed is None:
                    color = (0, 200, 255)
                else:
                    color = (0, 255, 0) if passed else (0, 0, 255)
                thick = 2
            cv2.drawContours(panel, [contour], -1, color, thick, cv2.LINE_AA)
            center = item.get("center")
            if center is not None:
                cv2.circle(panel, (int(center[0]), int(center[1])), 3, color, -1, cv2.LINE_AA)

        lines = [title]
        if extra_lines:
            lines.extend([str(line) for line in extra_lines if str(line)])

        overlay_h = 12 + 24 * max(1, len(lines))
        cv2.rectangle(panel, (0, 0), (panel.shape[1], min(panel.shape[0], overlay_h)), (0, 0, 0), -1)
        y = 22
        for idx, line in enumerate(lines):
            scale = 0.62 if idx == 0 else 0.5
            thick = 2 if idx == 0 else 1
            cv2.putText(panel, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)
            y += 22
        return cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)

    def _pickup_debug_fit_image(self, image_bgr: Optional[np.ndarray], width: int, height: int) -> np.ndarray:
        canvas = np.full((height, width, 3), 18, dtype=np.uint8)
        if image_bgr is None or not isinstance(image_bgr, np.ndarray) or image_bgr.size == 0:
            cv2.putText(
                canvas,
                "No image",
                (20, max(36, height // 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (180, 180, 180),
                2,
                cv2.LINE_AA,
            )
            return canvas

        src_h, src_w = image_bgr.shape[:2]
        scale = min(width / max(1, src_w), height / max(1, src_h))
        dst_w = max(1, int(round(src_w * scale)))
        dst_h = max(1, int(round(src_h * scale)))
        resized = cv2.resize(image_bgr, (dst_w, dst_h), interpolation=cv2.INTER_AREA)
        x0 = (width - dst_w) // 2
        y0 = (height - dst_h) // 2
        canvas[y0:y0 + dst_h, x0:x0 + dst_w] = resized
        return canvas

    def _pickup_debug_render_tile(
        self,
        title: str,
        image_bgr: Optional[np.ndarray],
        *,
        width: int,
        height: int,
        footer_lines: Optional[List[str]] = None,
        border_color: Tuple[int, int, int] = (70, 70, 70),
    ) -> np.ndarray:
        footer_lines = footer_lines or []
        tile = np.full((height, width, 3), 28, dtype=np.uint8)
        cv2.rectangle(tile, (0, 0), (width - 1, height - 1), border_color, 2, cv2.LINE_AA)
        cv2.rectangle(tile, (0, 0), (width - 1, 42), (38, 38, 38), -1)
        cv2.putText(tile, title, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (240, 240, 240), 2, cv2.LINE_AA)

        footer_h = 22 * max(1, len(footer_lines)) + 18 if footer_lines else 0
        image_area_h = max(80, height - 54 - footer_h)
        tile[48:48 + image_area_h, 8:width - 8] = self._pickup_debug_fit_image(image_bgr, width - 16, image_area_h)

        if footer_lines:
            y = 48 + image_area_h + 22
            for line in footer_lines:
                cv2.putText(tile, str(line), (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (205, 205, 205), 1, cv2.LINE_AA)
                y += 22
        return tile

    def _pickup_debug_wrapped_text(
        self,
        image: np.ndarray,
        lines: List[str],
        *,
        origin: Tuple[int, int],
        max_chars: int = 34,
        line_gap: int = 22,
        color: Tuple[int, int, int] = (235, 235, 235),
    ) -> None:
        import textwrap

        x, y = origin
        for line in lines:
            wrapped = textwrap.wrap(str(line), width=max_chars) or [""]
            for part in wrapped:
                cv2.putText(image, part, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)
                y += line_gap

    def _build_pickup_debug_dashboard(self, info: Dict[str, Any]) -> np.ndarray:
        passed = bool(info.get("passed", False))
        skipped = bool(info.get("skipped", False))
        error = str(info.get("error", "") or "")

        if error:
            headline = "PICKUP CHECK ERROR"
            banner_color = (0, 140, 210)
            subline = "The pickup verification failed before a final decision."
        elif skipped:
            headline = "PICKUP CHECK DISABLED"
            banner_color = (90, 90, 90)
            subline = "pickup_verification is disabled in config."
        elif passed:
            headline = "PICKED UP"
            banner_color = (28, 120, 44)
            subline = "Arm-camera contour verification passed."
        else:
            headline = "NOT PICKED UP"
            banner_color = (32, 32, 170)
            subline = "Arm-camera contour verification failed."

        width = 1520
        banner_h = 120
        tile_w = 360
        tile_h = 340
        gap = 14
        height = banner_h + gap + tile_h + gap
        canvas = np.full((height, width, 3), 14, dtype=np.uint8)

        cv2.rectangle(canvas, (0, 0), (width, banner_h), banner_color, -1)
        cv2.putText(canvas, headline, (26, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.45, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(canvas, subline, (28, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (235, 235, 235), 2, cv2.LINE_AA)

        raw_rgb = info.get("raw_rgb")
        raw_bgr = None
        if isinstance(raw_rgb, np.ndarray) and raw_rgb.ndim == 3:
            raw_bgr = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2BGR)
            try:
                crop_box = info.get("crop_box_xyxy")
                if isinstance(crop_box, (list, tuple)) and len(crop_box) == 4:
                    x0, y0, x1, y1 = [int(v) for v in crop_box]
                    cv2.rectangle(raw_bgr, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 255), 2)
            except Exception:
                pass

        live_panel_bgr = None
        if isinstance(info.get("input_masked_rgb"), np.ndarray):
            live_panel_bgr = cv2.cvtColor(info["input_masked_rgb"], cv2.COLOR_RGB2BGR)

        template_panel_bgr = None
        if isinstance(info.get("reference_masked_rgb"), np.ndarray):
            template_panel_bgr = cv2.cvtColor(info["reference_masked_rgb"], cv2.COLOR_RGB2BGR)

        status_tile = np.full((tile_h, tile_w, 3), 24, dtype=np.uint8)
        cv2.rectangle(status_tile, (0, 0), (tile_w - 1, tile_h - 1), (70, 70, 70), 2, cv2.LINE_AA)
        cv2.rectangle(status_tile, (0, 0), (tile_w - 1, 42), (38, 38, 38), -1)
        cv2.putText(status_tile, "Pickup Metrics", (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (240, 240, 240), 2, cv2.LINE_AA)

        lines = [
            f"Result: {'picked up' if passed else 'not picked up'}" if not skipped and not error else f"Result: {'skipped' if skipped else 'error'}",
            f"Method: {info.get('comparison_method', '-')}",
            f"Match: {float(info.get('contour_match_score', 0.0)):.4f}" if info.get("contour_match_score") is not None else "Match: -",
            f"Threshold: <= {float(info.get('contour_match_threshold', 0.0)):.4f}" if info.get("contour_match_threshold") is not None else "Threshold: -",
            f"Area ratio: {float(info.get('contour_area_ratio', 0.0)):.2f}" if info.get("contour_area_ratio") is not None else "Area ratio: -",
            f"Length ratio: {float(info.get('contour_length_ratio', 0.0)):.2f}" if info.get("contour_length_ratio") is not None else "Length ratio: -",
            f"Overlap ratio: {float(info.get('contour_overlap_ratio', 0.0)):.2f}" if info.get("contour_overlap_ratio") is not None else "Overlap ratio: -",
            f"Template candidates: {info.get('template_candidate_count', '-')}",
            f"Live candidates: {info.get('current_candidate_count', '-')}",
            f"Asset: {info.get('asset_name', '-')}",
        ]
        if error:
            lines.extend(["", f"Error: {error}"])
        self._pickup_debug_wrapped_text(status_tile, lines, origin=(14, 72))

        row_y = banner_h + gap
        x = gap
        canvas[row_y:row_y + tile_h, x:x + tile_w] = self._pickup_debug_render_tile(
            "Arm Camera",
            raw_bgr,
            width=tile_w,
            height=tile_h,
            footer_lines=["Current arm frame with ROI"] if raw_bgr is not None else ["No arm frame"],
        )
        x += tile_w + gap
        canvas[row_y:row_y + tile_h, x:x + tile_w] = self._pickup_debug_render_tile(
            "Live Contour",
            live_panel_bgr,
            width=tile_w,
            height=tile_h,
            footer_lines=[
                f"match={float(info.get('contour_match_score', 0.0)):.4f}" if info.get("contour_match_score") is not None else "match=-",
                f"thr<={float(info.get('contour_match_threshold', 0.0)):.4f}" if info.get("contour_match_threshold") is not None else "thr=-",
            ],
            border_color=(60, 110, 60) if passed else (90, 90, 90),
        )
        x += tile_w + gap
        canvas[row_y:row_y + tile_h, x:x + tile_w] = self._pickup_debug_render_tile(
            "Template Contour",
            template_panel_bgr,
            width=tile_w,
            height=tile_h,
            footer_lines=[f"asset={info.get('asset_name', '-')}"],
        )
        x += tile_w + gap
        canvas[row_y:row_y + tile_h, x:x + tile_w] = status_tile
        return canvas

    def _show_pickup_debug_dashboard(self, info: Dict[str, Any], *, wait_ms: int = 1) -> None:
        dashboard = self._build_pickup_debug_dashboard(info)
        cv2.namedWindow("pickup_verification_debug", cv2.WINDOW_NORMAL)
        cv2.imshow("pickup_verification_debug", dashboard)
        cv2.waitKey(max(1, int(wait_ms)))

    def _segment_pickup_stem_candidates(
        self,
        image_rgb: np.ndarray,
        *,
        roi_mask: Optional[np.ndarray],
        stem_prompt: str,
        stem_conf: float,
        min_contour_area_px: float,
    ) -> List[Dict[str, Any]]:
        image_rgb = np.asarray(image_rgb, dtype=np.uint8)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        masks = self.detector.segment(
            color_bgr=image_bgr,
            text_prompt=stem_prompt,
            confidence_threshold=stem_conf,
            category=stem_prompt,
        )

        roi_bool = None
        if isinstance(roi_mask, np.ndarray):
            roi_bool = np.squeeze(roi_mask).astype(bool)

        candidates: List[Dict[str, Any]] = []
        for idx, mask_info in enumerate(masks):
            contour = mask_info.get("contour")
            seg = np.squeeze(mask_info.get("segmentation"))
            if contour is None or seg.ndim != 2:
                continue

            area_px = float(cv2.contourArea(contour))
            if area_px < float(min_contour_area_px):
                continue

            perimeter_px = float(cv2.arcLength(contour, True))
            overlap_ratio = 0.0
            if roi_bool is not None:
                seg_bool = seg.astype(bool)
                cand_px = int(np.count_nonzero(seg_bool))
                if cand_px > 0:
                    overlap_ratio = float(np.count_nonzero(seg_bool & roi_bool) / cand_px)

            item = dict(mask_info)
            item.update(
                {
                    "idx": idx,
                    "contour_area_px": area_px,
                    "contour_length_px": perimeter_px,
                    "overlap_ratio": overlap_ratio,
                }
            )
            candidates.append(item)
        return candidates

    def _get_pickup_arm_reference_template(
        self,
        *,
        asset_name: str,
        assets_dir: Optional[str],
        stem_prompt: str,
        stem_conf: float,
        roi_padding_px: int,
        min_contour_area_px: float,
    ) -> Dict[str, Any]:
        cache_key = (
            str(asset_name),
            str(assets_dir),
            str(stem_prompt),
            float(stem_conf),
            int(roi_padding_px),
            float(min_contour_area_px),
        )
        cached = self._pickup_arm_reference_cache.get(cache_key)
        if cached is not None:
            return cached

        from qbot.mask_asset_tools import load_mask_asset, normalize_mask

        asset = load_mask_asset(asset_name, assets_dir=assets_dir)
        template_rgb = np.asarray(asset["image_rgb"], dtype=np.uint8)
        template_mask = normalize_mask(asset["mask"], template_rgb.shape[:2])
        x0, y0, x1, y1 = self._pickup_mask_bbox(template_mask, pad_px=roi_padding_px, shape_hw=template_rgb.shape[:2])

        template_crop_rgb = template_rgb[y0:y1, x0:x1].copy()
        template_crop_mask = normalize_mask(template_mask[y0:y1, x0:x1], template_crop_rgb.shape[:2])
        template_crop_rgb = apply_mask_to_image(template_crop_rgb, template_crop_mask)

        template_candidates = self._segment_pickup_stem_candidates(
            template_crop_rgb,
            roi_mask=template_crop_mask,
            stem_prompt=stem_prompt,
            stem_conf=stem_conf,
            min_contour_area_px=min_contour_area_px,
        )
        if not template_candidates:
            raise ValueError("SAM could not find a metal-rod contour in the pickup template")

        template_best = max(
            template_candidates,
            key=lambda item: (float(item.get("overlap_ratio", 0.0)), float(item.get("contour_area_px", 0.0)), float(item.get("score", 0.0))),
        )
        template_panel_rgb = self._draw_pickup_contour_panel(
            template_crop_rgb,
            template_candidates,
            template_best,
            title="Template contour",
            roi_mask=template_crop_mask,
            extra_lines=[
                f"candidates={len(template_candidates)}",
                f"overlap={float(template_best.get('overlap_ratio', 0.0)):.2f}",
                f"area={float(template_best.get('contour_area_px', 0.0)):.0f}px",
            ],
        )

        ref = {
            "asset": asset,
            "template_rgb": template_rgb,
            "template_mask": template_mask,
            "crop_box_xyxy": (x0, y0, x1, y1),
            "crop_rgb": template_crop_rgb,
            "crop_mask": template_crop_mask,
            "candidates": template_candidates,
            "best": template_best,
            "panel_rgb": template_panel_rgb,
            "contour": template_best["contour"],
            "contour_area_px": float(template_best.get("contour_area_px", 0.0)),
            "contour_length_px": float(template_best.get("contour_length_px", 0.0)),
        }
        self._pickup_arm_reference_cache[cache_key] = ref
        return ref

    def get_pickup_arm_camera_status(self) -> Dict[str, Any]:
        """
        Inspect the arm-camera pickup state using template-guided SAM contour matching.

        The saved mask still defines the rough ROI, but the pass/fail decision is based on the
        contour of the segmented `screw_stem` / metal rod rather than raw masked RGB similarity.
        """
        verify_cfg = self.config.get('pickup_verification', default={}) or {}
        enabled = bool(verify_cfg.get('enabled', True))
        camera = 'arm'
        asset_name = str(verify_cfg.get('mask_asset_name', 'screw_template'))
        assets_dir = verify_cfg.get('assets_dir')
        stem_prompt = str(verify_cfg.get('arm_template_sam_prompt', self.config.prompts.get('screw_stem', 'metal rod')))
        stem_conf = float(
            verify_cfg.get(
                'arm_template_sam_confidence',
                self.config.confidence_for_camera("arm").get('screw_stem', 0.2),
            )
        )
        roi_padding_px = max(0, int(verify_cfg.get('arm_template_roi_padding_px', 20)))
        min_contour_area_px = max(1.0, float(verify_cfg.get('arm_template_min_contour_area_px', 120.0)))
        contour_match_threshold = max(0.0, float(verify_cfg.get('arm_contour_match_threshold', 0.18)))
        contour_area_ratio_min = max(0.0, float(verify_cfg.get('arm_contour_area_ratio_min', 0.45)))
        contour_area_ratio_max = max(contour_area_ratio_min, float(verify_cfg.get('arm_contour_area_ratio_max', 2.20)))
        contour_length_ratio_min = max(0.0, float(verify_cfg.get('arm_contour_length_ratio_min', 0.60)))
        contour_length_ratio_max = max(contour_length_ratio_min, float(verify_cfg.get('arm_contour_length_ratio_max', 1.70)))
        overlap_ratio_min = max(0.0, min(1.0, float(verify_cfg.get('arm_contour_overlap_ratio_min', 0.10))))

        info: Dict[str, Any] = {
            "enabled": enabled,
            "camera": camera,
            "asset_name": asset_name,
            "comparison_method": "sam_contour",
            "stem_prompt": stem_prompt,
            "stem_confidence": stem_conf,
            "comparison_threshold": contour_match_threshold,
            "comparison_direction": "le",
            "contour_match_threshold": contour_match_threshold,
            "contour_area_ratio_min": contour_area_ratio_min,
            "contour_area_ratio_max": contour_area_ratio_max,
            "contour_length_ratio_min": contour_length_ratio_min,
            "contour_length_ratio_max": contour_length_ratio_max,
            "contour_overlap_ratio_min": overlap_ratio_min,
        }
        if not enabled:
            info["passed"] = True
            info["skipped"] = True
            return info

        try:
            from qbot.mask_asset_tools import normalize_mask

            reference = self._get_pickup_arm_reference_template(
                asset_name=asset_name,
                assets_dir=assets_dir,
                stem_prompt=stem_prompt,
                stem_conf=stem_conf,
                roi_padding_px=roi_padding_px,
                min_contour_area_px=min_contour_area_px,
            )

            _depth_verify, color_verify = self.handles.cam_arm.get_rgbd()
            info["raw_rgb"] = color_verify
            info["mask"] = reference["template_mask"]

            ref_mask = normalize_mask(reference["template_mask"], color_verify.shape[:2])
            x0, y0, x1, y1 = self._pickup_mask_bbox(ref_mask, pad_px=roi_padding_px, shape_hw=color_verify.shape[:2])
            current_crop_rgb = np.asarray(color_verify[y0:y1, x0:x1], dtype=np.uint8).copy()
            current_crop_mask = normalize_mask(ref_mask[y0:y1, x0:x1], current_crop_rgb.shape[:2])
            current_crop_rgb = apply_mask_to_image(current_crop_rgb, current_crop_mask)

            current_candidates = self._segment_pickup_stem_candidates(
                current_crop_rgb,
                roi_mask=current_crop_mask,
                stem_prompt=stem_prompt,
                stem_conf=stem_conf,
                min_contour_area_px=min_contour_area_px,
            )
            if not current_candidates:
                raise ValueError("SAM could not find a metal-rod contour in the live arm image")

            ref_contour = reference["contour"]
            ref_area = max(1e-6, float(reference["contour_area_px"]))
            ref_length = max(1e-6, float(reference["contour_length_px"]))

            for item in current_candidates:
                contour = item["contour"]
                item["contour_match_score"] = float(cv2.matchShapes(ref_contour, contour, cv2.CONTOURS_MATCH_I1, 0.0))
                item["contour_area_ratio"] = float(item["contour_area_px"] / ref_area)
                item["contour_length_ratio"] = float(item["contour_length_px"] / ref_length)
                item["selection_cost"] = (
                    float(item["contour_match_score"])
                    + 0.18 * abs(np.log(max(item["contour_area_ratio"], 1e-6)))
                    + 0.12 * (1.0 - float(item.get("overlap_ratio", 0.0)))
                )

            current_best = min(current_candidates, key=lambda item: float(item.get("selection_cost", float("inf"))))
            contour_match_score = float(current_best["contour_match_score"])
            contour_area_ratio = float(current_best["contour_area_ratio"])
            contour_length_ratio = float(current_best["contour_length_ratio"])
            overlap_ratio = float(current_best.get("overlap_ratio", 0.0))

            passed = (
                contour_match_score <= contour_match_threshold
                and contour_area_ratio_min <= contour_area_ratio <= contour_area_ratio_max
                and contour_length_ratio_min <= contour_length_ratio <= contour_length_ratio_max
                and overlap_ratio >= overlap_ratio_min
            )

            current_panel_rgb = self._draw_pickup_contour_panel(
                current_crop_rgb,
                current_candidates,
                current_best,
                title="Live contour",
                passed=passed,
                roi_mask=current_crop_mask,
                extra_lines=[
                    f"match={contour_match_score:.4f} <= {contour_match_threshold:.4f}",
                    f"area_ratio={contour_area_ratio:.2f}",
                    f"len_ratio={contour_length_ratio:.2f}",
                    f"overlap={overlap_ratio:.2f}",
                ],
            )

            info.update(
                {
                    "passed": passed,
                    "asset_paths": reference["asset"].get("paths", {}),
                    "crop_box_xyxy": (x0, y0, x1, y1),
                    "template_crop_box_xyxy": reference["crop_box_xyxy"],
                    "template_candidate_count": len(reference["candidates"]),
                    "current_candidate_count": len(current_candidates),
                    "masked_pixel_count": int(np.count_nonzero(current_crop_mask)),
                    "comparison_score": contour_match_score,
                    "contour_match_score": contour_match_score,
                    "contour_area_ratio": contour_area_ratio,
                    "contour_length_ratio": contour_length_ratio,
                    "contour_overlap_ratio": overlap_ratio,
                    "contour_template_area_px": ref_area,
                    "contour_current_area_px": float(current_best["contour_area_px"]),
                    "contour_template_length_px": ref_length,
                    "contour_current_length_px": float(current_best["contour_length_px"]),
                    "input_masked_rgb": current_panel_rgb,
                    "reference_masked_rgb": reference["panel_rgb"],
                    "current_crop_rgb": current_crop_rgb,
                    "template_crop_rgb": reference["crop_rgb"],
                }
            )
            return info
        except Exception as e:
            info["passed"] = False
            info["error"] = str(e)
            return info

    #use the camera on the arm to verify if the screw is picked up by comparing the masked RGB image to a saved reference asset. This can help catch pickup failures before moving to the installation step.
    def verify_pickup_arm_camera(
        self,
        debug: bool = False,
        *,
        debug_gui: bool = False,
        debug_gui_wait_ms: int = 1,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify screw pickup by comparing a masked RGB image to a saved reference asset.

        Configuration is read from `pickup_verification` in `qbot/config/cycles.yaml`.
        Returns `(passed, info)` where `passed=True` also covers the disabled/skipped case.
        """
        verify_cfg = self.config.get('pickup_verification', default={}) or {}
        max_attempts = max(1, int(verify_cfg.get("arm_contour_match_attempts", 5)))
        attempt_history: List[Dict[str, Any]] = []
        info: Dict[str, Any] = {}
        passed = False

        def _attempt_summary(attempt_info: Dict[str, Any], attempt_no: int) -> Dict[str, Any]:
            return {
                "attempt": attempt_no,
                "passed": bool(attempt_info.get("passed", False)),
                "error": attempt_info.get("error"),
                "contour_match_score": attempt_info.get("contour_match_score"),
                "contour_match_threshold": attempt_info.get("contour_match_threshold"),
                "contour_area_ratio": attempt_info.get("contour_area_ratio"),
                "contour_length_ratio": attempt_info.get("contour_length_ratio"),
                "contour_overlap_ratio": attempt_info.get("contour_overlap_ratio"),
                "current_candidate_count": attempt_info.get("current_candidate_count"),
            }

        for attempt_idx in range(max_attempts):
            attempt_no = attempt_idx + 1
            if attempt_no > 1:
                print(f"[STEP] Retrying pickup verification contour match ({attempt_no}/{max_attempts})...")

            info = self.get_pickup_arm_camera_status()
            passed = bool(info.get("passed", False))
            attempt_history.append(_attempt_summary(info, attempt_no))

            if bool(info.get("skipped")) or passed:
                break

            if attempt_no < max_attempts:
                error = info.get("error")
                if error:
                    print(f"[WARN] Pickup verification attempt {attempt_no}/{max_attempts} failed: {error}")
                else:
                    print(
                        f"[WARN] Pickup verification attempt {attempt_no}/{max_attempts} did not match "
                        f"(match={float(info.get('contour_match_score', 0.0)):.4f}, "
                        f"threshold<={float(info.get('contour_match_threshold', 0.0)):.4f})."
                    )

        info["verification_attempt"] = len(attempt_history)
        info["verification_attempts_configured"] = max_attempts
        info["verification_attempt_history"] = attempt_history
        debug_gui_wait_ms = max(1, int(debug_gui_wait_ms))

        if bool(info.get("skipped")):
            print("[INFO] pickup_verification disabled; skipping mask similarity check.")
            if debug_gui:
                try:
                    self._show_pickup_debug_dashboard(info, wait_ms=debug_gui_wait_ms)
                except Exception as e:
                    print(f"[WARN] pickup_verification debug GUI failed: {e}")
            return True, info

        if debug:
            try:
                current_masked_rgb = info.get("input_masked_rgb")
                ref_masked_rgb = info.get("reference_masked_rgb")
                if isinstance(current_masked_rgb, np.ndarray) and current_masked_rgb.ndim == 3:
                    current_bgr = cv2.cvtColor(current_masked_rgb, cv2.COLOR_RGB2BGR)
                    panel = current_bgr
                    if isinstance(ref_masked_rgb, np.ndarray) and ref_masked_rgb.ndim == 3:
                        ref_bgr = cv2.cvtColor(ref_masked_rgb, cv2.COLOR_RGB2BGR)
                        if ref_bgr.shape[:2] != current_bgr.shape[:2]:
                            ref_bgr = cv2.resize(ref_bgr, (current_bgr.shape[1], current_bgr.shape[0]))
                        panel = np.hstack([ref_bgr, current_bgr])

                    score = float(info.get("contour_match_score", 0.0))
                    threshold = float(info.get("contour_match_threshold", 0.0))
                    overlay_text = (
                        f"match={score:.4f}  thr={threshold:.4f}  "
                        f"pass={passed}"
                    )
                    cv2.rectangle(panel, (5, 5), (min(panel.shape[1] - 5, 700), 35), (0, 0, 0), -1)
                    cv2.putText(
                        panel,
                        overlay_text,
                        (10, 27),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.imshow("verify_pickup_arm_camera_masked", panel)
                    cv2.waitKey(0)
            except Exception as e:
                print(f"[WARN] verify_pickup_arm_camera debug display failed: {e}")

        if debug_gui:
            try:
                self._show_pickup_debug_dashboard(info, wait_ms=debug_gui_wait_ms)
            except Exception as e:
                print(f"[WARN] pickup_verification debug GUI failed: {e}")

        error = info.get("error")
        if error:
            print(f"[WARN] Pickup verification check failed: {error}")
        else:
            print(
                "[STEP] Pickup verification contour match "
                f"(asset={info.get('asset_name')}, camera={info.get('camera')}): "
                f"attempts={int(info.get('verification_attempt', 1))}/{int(info.get('verification_attempts_configured', 1))} "
                f"match={float(info.get('contour_match_score', 0.0)):.4f} "
                f"(threshold<={float(info.get('contour_match_threshold', 0.0)):.4f}) "
                f"area_ratio={float(info.get('contour_area_ratio', 0.0)):.2f} "
                f"len_ratio={float(info.get('contour_length_ratio', 0.0)):.2f} "
                f"overlap={float(info.get('contour_overlap_ratio', 0.0)):.2f}"
            )
        return passed, info

    def verify_screw_attached_fixed_camera(self, image_rgb: Optional[np.ndarray] = None, debug: bool = False) -> bool:
        """
        Use fixed-camera SAM detection on an RGB crop below the end-effector.

        The crop is defined in `pickup_verification.fixed_crop_box` and clamped so it never starts
        above the projected end-effector row. SAM detects `screw_stem` in the crop, candidates are
        filtered by contour length (pixels), and the closest valid stem to the end-effector is used
        to decide attachment based on a pixel distance threshold.

        Note: if `image_rgb` is provided, it is treated as a BGR OpenCV image and converted to RGB.
        """
        try:
            if image_rgb is None:
                _depth, image_rgb, _intr, _T_cam = self.camera_helper.get_rgbd_and_intrinsics("fixed")
            else:
                image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)

            if image_rgb is None or image_rgb.ndim != 3:
                print("[WARN] Fixed-camera pickup SAM check: invalid RGB image")
                return False

            u, v = self.camera_helper.project_tcp_to_fixed_cam()
            if u is None or v is None:
                print("[WARN] Fixed-camera pickup SAM check: could not project end-effector into fixed camera image")
                return False

            h, w = image_rgb.shape[:2]
            px = int(np.clip(round(u), 0, max(0, w - 1)))
            py = int(np.clip(round(v), 0, max(0, h - 1)))

            verify_cfg = self.config.get('pickup_verification', default={}) or {}
            crop_box_cfg = verify_cfg.get('fixed_crop_box', {}) or {}
            if not isinstance(crop_box_cfg, dict):
                crop_box_cfg = {}

            if crop_box_cfg:
                crop_w = max(32, int(crop_box_cfg.get('width_px', 220)))
                crop_h = max(32, int(crop_box_cfg.get('height_px', 220)))
                x_offset_px = int(crop_box_cfg.get('x_offset_px', -crop_w // 2))
                y_offset_px = int(crop_box_cfg.get('y_offset_px', 0))
                x0 = px + x_offset_px
                y0 = py + y_offset_px
                y0 = max(y0, py)  # crop only under/on the end-effector row
                x1 = x0 + crop_w
                y1 = y0 + crop_h
            else:
                crop_size_px = max(32, int(verify_cfg.get('fixed_crop_size_px', 220)))
                half = crop_size_px // 2
                x0 = px - half
                y0 = py
                x1 = x0 + crop_size_px
                y1 = y0 + crop_size_px

            x0 = max(0, x0)
            y0 = max(0, y0)
            x1 = min(w, x1)
            y1 = min(h, y1)
            if x1 <= x0 or y1 <= y0:
                print("[WARN] Fixed-camera pickup SAM check: invalid crop ROI")
                return False

            crop_rgb = image_rgb[y0:y1, x0:x1].copy()
            crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
            tip_x_crop = int(np.clip(px - x0, 0, max(0, crop_bgr.shape[1] - 1)))
            tip_y_crop = int(np.clip(py - y0, 0, max(0, crop_bgr.shape[0] - 1)))

            stem_prompt = self.config.prompts.get('screw_stem', 'metal rod')
            stem_conf = 0.07 #float(self.config.confidence_for_camera("fixed").get('screw_stem', 0.2))
            min_len_px = float(verify_cfg.get('fixed_sam_stem_min_contour_length_px', 40))
            max_len_px = float(verify_cfg.get('fixed_sam_stem_max_contour_length_px', 800))
            attach_thresh_px = float(verify_cfg.get('fixed_sam_attach_distance_threshold_px', 35))
            min_len_px = max(0.0, min_len_px)
            max_len_px = max(min_len_px, max_len_px)
            attach_thresh_px = max(0.0, attach_thresh_px)

            stem_masks = self.detector.segment(
                color_bgr=crop_bgr,
                text_prompt=stem_prompt,
                confidence_threshold=stem_conf,
                category=stem_prompt,
            )

            ee_pt = np.array([float(tip_x_crop), float(tip_y_crop)], dtype=np.float32)
            candidates: List[Dict[str, Any]] = []
            rejected: List[Dict[str, Any]] = []

            for idx, m in enumerate(stem_masks):
                seg = np.squeeze(m.get("segmentation"))
                if seg.ndim != 2:
                    continue
                seg_u8 = (seg.astype(bool) * 255).astype(np.uint8)
                contours, _ = cv2.findContours(seg_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue
                contour = max(contours, key=cv2.contourArea)
                contour_len = float(cv2.arcLength(contour, True))
                bx, by, bw, bh = cv2.boundingRect(contour)
                pts = contour.reshape(-1, 2).astype(np.float32)
                if pts.size == 0:
                    continue
                min_dist_px = float(np.min(np.linalg.norm(pts - ee_pt[None, :], axis=1)))
                try:
                    cx, cy = mask_centroid(seg)
                    centroid = (float(cx), float(cy))
                except Exception:
                    centroid = None

                item = {
                    "idx": idx,
                    "contour": contour,
                    "contour_length_px": contour_len,
                    "bbox_xywh": (int(bx), int(by), int(bw), int(bh)),
                    "width_px": float(bw),
                    "height_px": float(bh),
                    "min_distance_px": min_dist_px,
                    "centroid": centroid,
                }
                if min_len_px <= contour_len <= max_len_px:
                    candidates.append(item)
                else:
                    rejected.append(item)

            best = min(candidates, key=lambda c: c["min_distance_px"]) if candidates else None
            best_dist_px = float(best["min_distance_px"]) if best is not None else float("inf")
            attached = bool(best is not None and best_dist_px <= attach_thresh_px)

            print(
                "[STEP] Fixed-camera pickup SAM check "
                f"crop=({x0},{y0})-({x1},{y1}) total={len(stem_masks)} valid={len(candidates)} "
                f"len_range=[{min_len_px:.1f},{max_len_px:.1f}] "
                f"best_dist_px={best_dist_px if best is not None else None} "
                f"attach_thresh_px={attach_thresh_px:.1f} attached={attached}"
            )

            if debug:
                all_items = sorted(rejected , key=lambda it: it["idx"])
                if all_items:
                    print("[DEBUG] Fixed-camera stem metrics (pre-filter): rejected")
                    for item in all_items:
                        passed_len = (min_len_px <= item["contour_length_px"] <= max_len_px)
                        bx, by, bw, bh = item["bbox_xywh"]
                        print(
                            "  "
                            f"#{item['idx']} "
                            f"L={item['contour_length_px']:.1f}px "
                            f"W={item['width_px']:.0f}px "
                            f"H={item['height_px']:.0f}px "
                            f"D={item['min_distance_px']:.1f}px "
                            f"bbox=({bx},{by},{bw},{bh}) "
                            f"{'OK' if passed_len else 'LEN_REJ'}"
                        )
                all_items = sorted(candidates , key=lambda it: it["idx"])
                if all_items:
                    print("[DEBUG] Fixed-camera stem metrics (pre-filter): candidates")
                    for item in all_items:
                        passed_len = (min_len_px <= item["contour_length_px"] <= max_len_px)
                        bx, by, bw, bh = item["bbox_xywh"]
                        print(
                            "  "
                            f"#{item['idx']} "
                            f"L={item['contour_length_px']:.1f}px "
                            f"W={item['width_px']:.0f}px "
                            f"H={item['height_px']:.0f}px "
                            f"D={item['min_distance_px']:.1f}px "
                            f"bbox=({bx},{by},{bw},{bh}) "
                            f"{'OK' if passed_len else 'LEN_REJ'}"
                        )
            if debug:
                try:
                    
                    full_dbg = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                    cv2.rectangle(full_dbg, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 255), 2)
                    cv2.drawMarker(full_dbg, (px, py), (0, 0, 255), cv2.MARKER_TARGET, 18, 2, cv2.LINE_AA)

                    crop_dbg = crop_bgr.copy()
                    cv2.drawMarker(crop_dbg, (tip_x_crop, tip_y_crop), (0, 0, 255), cv2.MARKER_TARGET, 18, 2, cv2.LINE_AA)
                    cv2.circle(crop_dbg, (tip_x_crop, tip_y_crop), int(max(3, round(attach_thresh_px))), (0, 255, 255), 1, cv2.LINE_AA)
                    # cv2.putText(
                    #     crop_dbg,
                    #     "End-effector",
                    #     (
                    #         int(np.clip(tip_x_crop + 14, 0, max(0, crop_dbg.shape[1] - 1))),
                    #         int(np.clip(tip_y_crop + 20, 12, max(12, crop_dbg.shape[0] - 1))),
                    #     ),
                    #     cv2.FONT_HERSHEY_SIMPLEX,
                    #     0.55,
                    #     (0, 255, 255),
                    #     2,
                    #     cv2.LINE_AA,
                    # )

                    for item in rejected:
                        cv2.drawContours(crop_dbg, [item["contour"]], -1, (80, 80, 80), 1)
                    for item in candidates:
                        color = (0, 180, 255)
                        thick = 2
                        if best is item:
                            color = (0, 255, 0) if attached else (0, 0, 255)
                            thick = 3
                        cv2.drawContours(crop_dbg, [item["contour"]], -1, color, thick)
                        cxy = item.get("centroid")
                        if cxy is not None:
                            cv2.circle(crop_dbg, (int(cxy[0]), int(cxy[1])), 3, color, -1, cv2.LINE_AA)

                    status = (
                        f"stems={len(stem_masks)} valid={len(candidates)} "
                        f"best_d={best_dist_px if best is not None else -1:.1f}px "
                        f"thr={attach_thresh_px:.1f}px attached={attached}"
                    )
                    # cv2.rectangle(crop_dbg, (4, 4), (min(crop_dbg.shape[1] - 4, 560), 34), (0, 0, 0), -1)
                    # cv2.putText(crop_dbg, status, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

                    cv2.imshow("verify_screw_attached_fixed_camera_full", full_dbg)
                    cv2.imshow("verify_screw_attached_fixed_camera_crop_sam", crop_dbg)
                    cv2.waitKey(0)
                except Exception as e:
                    print(f"[WARN] Fixed-camera pickup SAM check debug image show failed: {e}")

            return attached
        except Exception as e:
            print(f"[WARN] Fixed-camera pickup SAM check failed: {e}")
            return False
    
    def  verify_attachment_loop(self, debug: bool) -> bool:
        """Loop until screw is properly attached."""
        while True:
            is_vertical, screw_on_head = self.orientation_checker.check_attachment(
                debug=debug
            )
            
            if screw_on_head and not is_vertical:
                print("[WARN] Screw not properly attached")
                if self.voice_client:
                    msg = self.config.get('voice', 'screw_not_attached',
                                         default="Screw not attached properly. Fix and press trigger.")
                    self.voice_client.speak_openai(msg)
                self.wait_for_trigger()
            elif not screw_on_head:
                return False
            else:
                return True
    
    def run_screwdriver(self):
        """Run the screwdriver, then return to start after a terminal result."""
        if self.screwdriver_client is None:
            return False
        
        print("[STEP] Running screwdriver...")
        self.screwdriver_client.run_screw_async(debug=False)
        
        poll_interval = self.config.get('hardware', 'timing', 'screwdriver_poll_interval', default=0.5)
        
        while True:
            if self.cancel_check is not None and self.cancel_check():
                print("[STEP] Cancellation requested while screwdriver was running.")
                try:
                    self.screwdriver_client.stop()
                except Exception as e:
                    print(f"[WARN] Failed to stop screwdriver after cancellation: {e}")
                return False
            status = self.screwdriver_client.get_status()
            if status.state in ("completed", "error"):                
                break
            time.sleep(poll_interval)

        succeeded = status.state == "completed"
        print("[STEP] Screwdriver finished; moving back to start_joint...")
        self.move_to_start_position()
        return succeeded
    
    def  wait_for_trigger(self,timeout_s=30.0):
        """Wait for tool button press."""
        poll_interval = 0.1
        start_time = time.time()
        
        while True:
            if self.cancel_check is not None and self.cancel_check():
                print("[WARN] Trigger wait cancelled by stop input.")
                return False
            if self.handles.arm.get_tool_io():
                return True
            
            if time.time() - start_time > timeout_s:
                print(f"[WARN] Trigger wait timeout after {timeout_s}s")
                return False
            
            time.sleep(poll_interval)

class GenericCycleManager:
    """Manages complete screw pickup and installation cycles."""

    def __init__(
        self,
        handles,
        detector: Sam3Detector,
        voice_client: Optional[OpenAIRealtimeClient] = None,
        move_to_start: bool = False,
        debug: bool = False,        
    ):
    
        cfg_file = utils.check_path_exists(CFG_PATH,__file__)       
        config = Config(cfg_file) 
        self.handles = handles
        self.detector = detector
        self.config = config
        self.voice_client = voice_client
        self.debug = debug
        
        self.camera_helper = CameraHelper(handles, config)
        self.motion_planner = MotionPlanner(handles, config)
        self.orientation_checker = ScrewOrientationChecker(handles, detector, config)

        if move_to_start:
            self.move_to_start_position()

    def move_to_start_position(self):
        print("[STEP] Moving to start position...",self.config.get('start_joint'))
        self.handles.arm.moveJ(self.config.get('start_joint'))

    def move_to_screw_pickup_position(self):
        j = self.config.get('screw_pickup_joint')
        print('[STEP] Moving to screw_pickup position...',j)
        self.handles.arm.moveJ(self.config.get('screw_pickup_joint'))

    def detect_generic(
        self,
        prompt: str,
        color: np.ndarray,
        conf: float = 0.2,
        orientation_align: str = 'long'

    ) -> List[Dict[str, Any]]:
        if self.debug:     
            cv2.imshow("SAM3 Raw", color)
            cv2.waitKey(0)
            
        masks = self.detector.segment(
            color_bgr=color,
            text_prompt=prompt,
            confidence_threshold=conf,
            category=prompt,
            orientation_align=orientation_align,
        )
        
        if self.debug and masks:
            output_dir = self.config.get('debug', 'output_dir', default='data/image_samples')
            output_file = self.config.get('debug', 'images', 'generic_detect', default='generic_detect.png')
            colors = self.config.get('debug', 'colors', default={})
   
            vis = draw_mask_debug(
                color, masks,
                output_path=f"{output_dir}/{output_file}",
                category_colors={prompt: tuple(colors.get(prompt, [0, 0, 255]))}
            )
            cv2.imshow("SAM3 Debug Visualization", vis)
            cv2.waitKey(0)

        return masks
    
    def move_to_generic_prompt(
        self,
        prompt,
        camera_type,
        conf: float = 0.2,
        z_offset: float = 0.1,
        y_offset: float = 0.0,
        x_offset: float = 0.0,
        move: bool = False,
        image_path: str | None = None,
        use_servo: bool = False,
        speed: Optional[float] = None,
        orientation_align: str = 'long',
        ignore_rotation: bool = True,
        start_joint: Optional[List[float]] = None
    ):
        """Detect and move to generic prompt targets.

        If use_servo=True, uses visual servoing: each ServoL step recomputes detection
        and reselects a target pose instead of following a fixed pose.
        """
        print(f"[STEP] Moving to targets for prompt: {prompt}...Image: {image_path}")

        
        if start_joint:
            self.handles.arm.moveJ(start_joint)
            # self.handles.gripper.open(True)#open gripper
            time.sleep(2.0)

        # If an offline image is provided, we can't do visual servoing – just do one shot.
        if image_path is not None and use_servo:
            print("[WARN] image_path provided; visual servo with live feedback is not supported. Falling back to single-shot move.")
            use_servo = False

        # ---- Visual servoing path ----
                # ---- Visual servoing path ----
        if use_servo:
            servo_offset = z_offset
            print("[INFO] Using visual servo (detection + target selection in servo loop).")

            def target_pose_fn() -> Optional[List[float]]:
                # 1) Capture new RGBD + intrinsics
                depth, color, intr, T_cam = self.camera_helper.get_rgbd_and_intrinsics(camera_type)
                T_base_tcp = self.handles.arm.get_T_base_tcp()

                # 2) Run detection for this prompt
                masks = self.detect_generic(prompt, color, conf=conf ,orientation_align = orientation_align)
                if not masks:
                    print("[SERVO] No objects detected for prompt.")
                    return None

                # 3) Compute 3D targets
                targets = self.compute_targets(color, prompt, masks, depth, intr)
                if not targets:
                    print("[SERVO] No valid 3D targets computed.")
                    return None

                # 4) Build approach pose candidates
                print('targets',targets)
                candidates = self.motion_planner.compute_approach_poses(T_base_tcp,
                    targets, T_cam, camera_type, servo_offset, ignore_rotation=ignore_rotation
                )
                if not candidates:
                    print("[SERVO] No candidate poses from targets.")
                    return None

                # 5) Select best candidate given current robot state
                best_cand, best_cost = self.motion_planner.select_best_candidate(
                    candidates,
                    ignore_close=False,
                )
                if best_cand is None:
                    print("[SERVO] No reachable candidates.")
                    return None

                return best_cand["pose"]

            status = self.motion_planner.servo_towards_dynamic(
                target_pose_fn=target_pose_fn,
                speed=speed if speed is not None else self.config.default_speed,
            )

            # return True, status
        print('single shot movement')
        # ---- Original single-shot behavior ----
        depth, color, intr, T_cam = self.camera_helper.get_rgbd_and_intrinsics(camera_type)
        T_base_tcp = self.handles.arm.get_T_base_tcp()
        if image_path is not None:
            color = utils.read_image_file(image_path)
            
        masks = self.detect_generic(prompt, color, conf=conf,orientation_align=orientation_align)
        if not masks:
            print("[WARN] No objects detected for prompt")
            return False, 'No object Mask'
        
        if image_path is None:
            targets = self.compute_targets(color, prompt, masks, depth, intr)
            
            candidates = self.motion_planner.compute_approach_poses(
                T_base_tcp,
                targets, T_cam, camera_type, x_offset=x_offset, y_offset=y_offset, z_offset=z_offset,ignore_rotation=ignore_rotation
            )
            print('candidates',candidates)
            print('moving to best reachable candidate...')
            return self.motion_planner.move_to_best_reachable(
                candidates,
                speed=speed if speed is not None else self.config.default_speed,
                move=move,
                debug=self.debug,
            ),'Moving to best target'
        else:
            print("[INFO] image_path provided, skipping move")
            return True, 'Skipped due to Image Path'

    # def compute_targets(self,color, label, masks, depth, intr) -> List[Dict]:
    #     """Compute 3D targets from masks."""
    #     targets = []
    #     for mask in masks:
    #         target_info = self.detector.compute_target_from_mask(
    #             mask, depth, intr, depth_scale=0.001
    #         )
    #         if target_info is None:
    #             continue
    #         target_info["image"] = color
    #         target_info["prompt"] = label
    #         if target_info:
    #             targets.append(target_info)

    #     return targets
    def compute_targets(self, color, prompt_label, masks, depth, intr) -> List[Dict]:
        """Compute 3D targets from masks."""
        targets = []
        for mask in masks:
            target_info = self.detector.compute_target_from_mask(
                mask, depth, intr, depth_scale=0.001
            )
            if target_info is None:
                continue

            target_info["image"] = color
            target_info["prompt"] = prompt_label
            targets.append(target_info)

        return targets

    def run_servo_prompt(
        self,
        prompt,
        camera_type='arm',
        conf=0.2,
        z_offset=0.1,
        orientation_align='long'
    ):
        """
        Run a single visual-servo sequence towards the given prompt.

        Stops when:
          - Target is almost reached  -> returns "target_reached"
          - Motion reaches steady state -> returns "steady_state"
          - Target detection fails repeatedly -> returns "target_lost"
          - Timeout occurs -> returns "timeout"
        """
        success = False
        try:
            move_success, status = self.move_to_generic_prompt(
                prompt=prompt,
                camera_type=camera_type,
                conf=conf,
                z_offset=z_offset,
                move=False, 
                use_servo=True,
                speed=None,          # use default speed unless overridden via config
                orientation_align=orientation_align,
            )

            if status == "target_reached":
                success =  True
                print(f"[TEST] Visual servo completed: target reached for '{prompt}'.")
            elif status == "steady_state":
                print(f"[TEST] Visual servo stopped in steady state for '{prompt}'.")
            elif status == "target_lost":
                print(f"[TEST] Visual servo stopped: target lost for '{prompt}'.")
            elif status == "timeout":
                print(f"[TEST] Visual servo stopped: timeout for '{prompt}'.")
            else:
                print(f"[TEST] Visual servo finished with unknown status: {status}")

            return success , status

        except KeyboardInterrupt:
            print("[TEST] Prompt loop stopped by user (KeyboardInterrupt).")
            return success , "interrupted"

def test_prompt(args, handles, detector):
    # Testing Prompts  
    cycle_manager = GenericCycleManager(
        handles=handles,
        detector=detector,
        voice_client=None,
        move_to_start=False,
    )

    if args.prompt_servo:
        print(f"[TEST] Starting generic prompt loop for '{args.prompt}' (Ctrl-C to stop)...")
        cycle_manager.run_servo_prompt(
            prompt=args.prompt,
            prompt_camera_type=args.prompt_camera_type,
            prompt_conf=args.prompt_conf,
            prompt_z_offset=args.prompt_z_offset,
            debug=args.debug,
        )
    else:
        print(f"[TEST] Moving to generic prompt '{args.prompt}'... Image: {args.test_image}")
        result = cycle_manager.move_to_generic_prompt(
            prompt=args.prompt,
            camera_type=args.prompt_camera_type,
            conf=args.prompt_conf,
            z_offset=args.prompt_z_offset,
            move=args.prompt_move,
            image_path=args.test_image,
            debug=args.debug,
            use_servo=args.prompt_servo,
            speed=args.prompt_speed,
        )

        if result is None:
            print("[TEST] No movement executed (no valid target).")
            return False
        else:
            print("[TEST] Movement executed.",result)
            return result

def main() -> None:
    """
    Basic CLI to test ScrewCycleManager / ManualScrewCycleManager.

    This intentionally does NOT test Config directly; it just relies on the
    classes to load and use configuration on their own.
    """
    parser = argparse.ArgumentParser(description="Test screw cycle managers.")
    parser.add_argument(
        "--mode",
        choices=["auto", "manual"],
        default="auto",
        help="auto: ScrewCycleManager / manual: ManualScrewCycleManager",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug prints / visualizations where supported.",
    )
    parser.add_argument(
        "--no-screwdriver",
        action="store_true",
        help="Skip real screwdriver client (use None).",
    )
    parser.add_argument(
        "--no-voice",
        action="store_true",
        help="Disable voice feedback.",
    )


    # Testing generic object detection
    parser.add_argument("--prompt", type=str, default = None, help="Prompt for generic object detection")
    parser.add_argument("--prompt_conf", type=float, default=0.2, help="Confidence for generic object detection")
    parser.add_argument("--prompt_z_offset", type=float, default=0.1, help="Z offset for generic object detection")
    parser.add_argument("--prompt_camera_type", type=str, default="fixed", help="Camera type, 'fixed' or 'arm'")
    parser.add_argument("--prompt_move", action="store_true", help="Move to detected generic object")       
    parser.add_argument("--prompt_servo",action="store_true",help="Use ServoL instead of moveL when moving towards prompt")
    parser.add_argument("--prompt_speed", type=float,default=None,help="Override robot speed for prompt motion / ServoL")
    parser.add_argument("--test_image", type=str, default = None, help="Debug a image file for generic object detection")

    args = parser.parse_args()

    # Initialize hardware``
    hw = HardwareInitializer(
        camera_arm_name="camera_gripper",
        camera_fixed_name="camera_fixed",
        ignore_gripper=True,
        tool_name="tcp_drill",
        debug=args.debug,
    )

    handles = hw.initialize()

    detector = Sam3Detector()

    if args.no_screwdriver:
        screwdriver_client = None
    else:
        screwdriver_client = (
            ScrewdriverClient() if args.enable_screwdriver else None
        )

    # If prompt argument provided, test generic prompt detection and movement
    if args.prompt is not None:
        print(f"[TEST] Testing generic prompt detection and movement... {args.prompt} ... Image: {args.test_image}")
        test_prompt(args, handles, detector)
        return

    if args.no_voice:
        voice_client = None
    else:
        voice_client = OpenAIRealtimeClient.load()
        voice_client.speak_openai('Quendabot Online')
    
    # Choose which manager to run based on mode.
    if args.mode == "manual":
        print("[INFO] Running MANUAL mode test (ManualScrewCycleManager)")
        manual_manager = ManualScrewCycleManager(
            handles=handles,
            screwdriver_client=screwdriver_client,
            voice_client=voice_client,
        )
        manual_manager.run_cycle(debug=args.debug)
    else:
        print("[INFO] Running AUTO mode test (ScrewCycleManager)")
        cycle_manager = ScrewCycleManager(
            handles=handles,
            detector=detector,
            screwdriver_client=screwdriver_client,
            voice_client=voice_client,
        )
        
        success = cycle_manager.run_cycle(debug=args.debug)
        print(f"[RESULT] Screw cycle success: {success}")


if __name__ == "__main__":
    main()
