#!/usr/bin/env python3
"""
Hardware initializer that loads config, connects camera(s), robot, gripper,
and returns a `HardwareHandles` bundle for the pipeline core.

Usage:
    from hardware_init import HardwareInitializer
    hw = HardwareInitializer(debug=True).initialize()
"""

import time
from dataclasses import dataclass, field
from typing import Any, Optional, Dict

import numpy as np
import yaml
import os
import sys

# Local modules
# from hardware.camera_rs import RealSenseCamera
# from hardware.urarm_control import URArmControl
# from hardware.robotik_gripper import RobotiqOnUR

from hardware.camera_rs_client import NetworkRealSenseCamera
from hardware.urarm_control_client import URArmControlClient, URDashboardRemote
from hardware.robotik_gripper_client import RobotiqOnURClient

import commons.grasp_utils as utils
import commons.utils as pose_utils


@dataclass
class HardwareHandles:
    # Flags
    ignore_arm: bool
    ignore_gripper: bool

    # Raw config & sections
    raw_cfg: Dict[str, Any]
    cam_arm_cfg: Dict[str, Any]
    cam_fixed_cfg: Dict[str, Any]
    frames_cfg: Dict[str, Any]
    robot_cfg: Dict[str, Any]
    gripper_cfg: Dict[str, Any]

    # Transforms
    T_tcp_cam: np.ndarray
    T_tcp_to_gripper: np.ndarray
    T_base_fixed_camera: Optional[np.ndarray]
    camera_tool_name: Optional[str]
    tool_tcp_transforms: Dict[str, np.ndarray]

    # Devices (use Any to allow either local or network camera objects)
    cam_arm: Optional[Any] = None
    cam_fixed: Optional[Any] = None
    arm_dash: Optional[URDashboardRemote] = None
    arm: Optional[Any] = None
    gripper: Optional[Any] = None
    tool_name: Optional[str] = None
    tool_cameras: Dict[str, Any] = field(default_factory=dict)
    tool_camera_cfgs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    tool_camera_transforms: Dict[str, np.ndarray] = field(default_factory=dict)
    tool_camera_tool_names: Dict[str, str] = field(default_factory=dict)

    # Motion params / waypoints
    j_speed: float = 0.5
    j_acc: float = 2.0
    l_speed: float = 0.5
    l_acc: float = 2.0

    

class HardwareInitializer:
    def __init__(
        self,
        ignore_arm: bool = False,
        ignore_gripper: bool = False,
        camera_arm_name: str = 'camera_arm',
        camera_fixed_name: str = 'camera_fixed',
        camera_warmup_secs: int = 60,
        tool_name: Optional[str] = None,
        camera_tool_name: Optional[str] = None,
        extra_arm_cameras: Optional[Dict[str, str]] = None,
        debug: bool = False,
        stop_dashboard_program_on_init: bool = True,
    ):
        config_dir = os.path.join(os.path.dirname(__file__), "config")
        config_file = os.path.join(config_dir, "config.yaml")
        if not os.path.exists(config_dir) or not os.path.exists(config_file):
            print("[!] missing config.yaml.")
            sys.exit(1)
        self.camera_arm_name = camera_arm_name
        self.tool_name = tool_name
        self.camera_tool_name = camera_tool_name or tool_name
        self.extra_arm_cameras = dict(extra_arm_cameras or {})
        self.stop_dashboard_program_on_init = stop_dashboard_program_on_init
        self.camera_fixed_name = camera_fixed_name
        self.ignore_arm = ignore_arm
        self.ignore_gripper = ignore_gripper
        self.camera_warmup_secs = camera_warmup_secs
        self.debug = debug

        self.config_path = config_file
        self._handles: Optional[HardwareHandles] = None

    def _create_camera(self, cfg: Dict[str, Any]) -> Any:
        """Create either a local RealSenseCamera or a NetworkRealSenseCamera and warm it up."""
        cam = NetworkRealSenseCamera(
                intrinsics = cfg.get("intrinsics"),
                server_ip=cfg["network_camera_ip"],
                port=cfg["network_camera_port"],
                camera_id=cfg.get("serial"),
            )

        # Start & warmup (both camera classes are expected to implement these)
        cam.start()
        cam.warmup(self.camera_warmup_secs)

        return cam
    
    def initialize(self) -> HardwareHandles:
        print("initializing hardware")

        # Load YAML config
        with open(self.config_path, "r") as f:
            raw_cfg = yaml.safe_load(f)

        cam_arm_cfg = raw_cfg[self.camera_arm_name]
        cam_fixed_cfg = raw_cfg[self.camera_fixed_name]
        frames_cfg = raw_cfg["frames"]
        robot_cfg = raw_cfg["robot"]
        gripper_cfg = raw_cfg.get("gripper", {})
        tool_name = self.tool_name
        camera_tool_name = self.camera_tool_name

        arm_control_file = os.path.join(os.path.dirname(self.config_path), "arm_control.yaml")
        with open(arm_control_file, "r") as f:
            arm_control_cfg = yaml.safe_load(f) or {}
        tool_tcp_transforms: Dict[str, np.ndarray] = {}
        for name, tool_cfg in (arm_control_cfg.get("tool", {}) or {}).items():
            pose = (tool_cfg or {}).get("tcp_pose")
            if pose is None or len(pose) != 6:
                continue
            T_flange_tcp = np.eye(4)
            T_flange_tcp[:3, :3] = pose_utils.axis_angle_to_rot(
                np.asarray(pose[3:6], dtype=float)
            )
            T_flange_tcp[:3, 3] = np.asarray(pose[:3], dtype=float)
            tool_tcp_transforms[str(name)] = T_flange_tcp

        # Transforms (TCP-based)
        T_tcp_drill_cam = utils.load_yaml_pose(frames_cfg.get("T_tcp_drill_camera"), default_degrees=True)
        T_tcp_gripper_cam = utils.load_yaml_pose(frames_cfg.get("T_tcp_gripper_camera"), default_degrees=True)

        camera_calibrations = {
            "tcp_drill": T_tcp_drill_cam,
            "tcp_gripper": T_tcp_gripper_cam,
        }

        def transform_from_active_tcp(camera_tcp_name: str) -> np.ndarray:
            T_camera_tool_cam = camera_calibrations.get(camera_tcp_name)
            if T_camera_tool_cam is None:
                raise ValueError(f"Unknown camera TCP calibration: {camera_tcp_name!r}")
            if tool_name == camera_tcp_name:
                return T_camera_tool_cam
            try:
                T_flange_active = tool_tcp_transforms[str(tool_name)]
                T_flange_camera_tool = tool_tcp_transforms[str(camera_tcp_name)]
                return (
                    np.linalg.inv(T_flange_active)
                    @ T_flange_camera_tool
                    @ T_camera_tool_cam
                )
            except KeyError as e:
                raise ValueError(f"Missing TCP calibration for tool {e.args[0]!r}") from e

        T_tcp_cam = transform_from_active_tcp(str(camera_tool_name))
        print(
            f"[INFO] Active robot TCP: {tool_name}; "
            f"arm camera calibrated from: {camera_tool_name}."
        )

        T_tcp_to_gripper = utils.load_yaml_pose(frames_cfg.get("T_tcp_to_gripper"), default_degrees=True)
        T_base_fixed_camera = utils.load_yaml_pose(
            frames_cfg.get("T_base_fixed_camera"), default_degrees=True
        )

        # Cameras
        cam_arm = self._create_camera(cam_arm_cfg)
        cam_fixed = self._create_camera(cam_fixed_cfg)
        tool_cameras = {"arm": cam_arm}
        tool_camera_cfgs = {"arm": cam_arm_cfg}
        tool_camera_transforms = {"arm": T_tcp_cam}
        tool_camera_tool_names = {"arm": str(camera_tool_name)}
        for alias, camera_config_name in self.extra_arm_cameras.items():
            alias = str(alias).strip().lower()
            camera_cfg = raw_cfg[str(camera_config_name)]
            camera_tcp_name = f"tcp_{alias}"
            camera = cam_arm if camera_config_name == self.camera_arm_name else self._create_camera(camera_cfg)
            tool_cameras[alias] = camera
            tool_camera_cfgs[alias] = camera_cfg
            tool_camera_transforms[alias] = transform_from_active_tcp(camera_tcp_name)
            tool_camera_tool_names[alias] = camera_tcp_name
            print(
                f"[INFO] Extra arm camera {alias!r}: config={camera_config_name!r}, "
                f"calibration={camera_tcp_name!r}."
            )

        # Motion params
        j_speed = float(robot_cfg.get("j_speed", 0.5))
        j_acc = float(robot_cfg.get("j_acc", robot_cfg.get("j_accel", 2.0)))  # support both keys
        l_speed = float(robot_cfg.get("l_speed", 0.5))
        l_acc = float(robot_cfg.get("l_acc", robot_cfg.get("l_accel", 2.0)))  # support both keys

        # Robot & gripper
        arm_dash: Optional[URDashboardRemote] = None
        gripper: Optional[Any] = None
        arm: Optional[Any] = None

        if not self.ignore_arm:
            if "ur_ip" not in robot_cfg:
                raise ValueError("No 'ur_ip' in robot config; cannot connect to UR arm.")
            
            server_url = 'http://'+robot_cfg['network_robot_ip'] + ":" + str(robot_cfg['network_robot_port'])
            arm_dash = URDashboardRemote(
                server_url,
                stop_program_on_init=self.stop_dashboard_program_on_init,
            )
            arm = URArmControlClient(server_url=server_url,tool_name=tool_name)

            if not self.ignore_gripper and gripper_cfg:

                gripper = RobotiqOnURClient(server_host=gripper_cfg['network_gripper_ip'],server_port=gripper_cfg['network_gripper_port'])
             
                # gripper.activate()
                # time.sleep(1.0)
                # gripper.open(block=True)

        handles = HardwareHandles(
            ignore_arm=self.ignore_arm,
            ignore_gripper=self.ignore_gripper,
            raw_cfg=raw_cfg,
            cam_arm_cfg=cam_arm_cfg,
            cam_fixed_cfg=cam_fixed_cfg,
            frames_cfg=frames_cfg,
            robot_cfg=robot_cfg,
            gripper_cfg=gripper_cfg,
            T_tcp_cam=T_tcp_cam,
            T_tcp_to_gripper=T_tcp_to_gripper,
            T_base_fixed_camera=T_base_fixed_camera,
            camera_tool_name=camera_tool_name,
            tool_tcp_transforms=tool_tcp_transforms,
            cam_arm=cam_arm,
            cam_fixed=cam_fixed,
            arm=arm,
            arm_dash=arm_dash,
            gripper=gripper,
            tool_name=tool_name,
            tool_cameras=tool_cameras,
            tool_camera_cfgs=tool_camera_cfgs,
            tool_camera_transforms=tool_camera_transforms,
            tool_camera_tool_names=tool_camera_tool_names,
            j_speed=j_speed,
            j_acc=j_acc,
            l_speed=l_speed,
            l_acc=l_acc,
        )
        self._handles = handles
        return handles

    def shutdown(self) -> None:
        """Best-effort shutdown for all devices created in initialize()."""
        h = self._handles
        if not h:
            return
        stopped_camera_ids = set()
        for camera in [h.cam_arm, *(h.tool_cameras or {}).values()]:
            if camera is None or id(camera) in stopped_camera_ids:
                continue
            stopped_camera_ids.add(id(camera))
            try:
                camera.stop()
            except Exception:
                pass
        try:
            if h.cam_fixed is not None:
                h.cam_fixed.stop()
        except Exception:
            pass
        try:
            if h.gripper is not None and hasattr(h.gripper, "deactivate"):
                h.gripper.deactivate()
        except Exception:
            pass
        try:
            if h.arm is not None and hasattr(h.arm, "stop"):
                h.arm.stop()
        except Exception:
            pass
