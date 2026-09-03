#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import time
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import yaml

CONFIG_PATH = ROOT / "hardware" / "config" / "config.yaml"
DEFAULT_WINDOW_NAME = "Arm Jog"


def load_hardware_config() -> Dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_default_server_url(cfg: Dict[str, Any]) -> str:
    robot_cfg = cfg.get("robot", {}) or {}
    host = robot_cfg.get("network_robot_ip")
    port = robot_cfg.get("network_robot_port")
    if not host or not port:
        raise ValueError("robot.network_robot_ip or robot.network_robot_port is missing from hardware/config/config.yaml")
    return f"http://{host}:{port}"


def format_vector(values: Optional[list[float]], precision: int = 4) -> str:
    if values is None:
        return "unavailable"
    return "[" + ", ".join(f"{float(v):.{precision}f}" for v in values) + "]"


def fetch_state(arm: Any) -> Tuple[list[float], Optional[list[float]]]:
    pose = [float(v) for v in arm.get_tcp_pose_axis_angle()]
    joints: Optional[list[float]]
    try:
        joints = [float(v) for v in arm.get_actual_q()]
    except Exception:
        joints = None
    return pose, joints


def print_state(pose: list[float], joints: Optional[list[float]]) -> None:
    print(f"TCP pose  : {format_vector(pose, precision=4)}")
    if joints is None:
        print("Joints    : unavailable")
    else:
        print(f"Joints    : {format_vector(joints, precision=4)}")


def make_status_frame(width: int = 640, height: int = 480, message: str = "Waiting for camera frames...") -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(frame, message, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def draw_overlay(
    frame_bgr: np.ndarray,
    camera_name: str,
    pose: Optional[list[float]],
    joints: Optional[list[float]],
    step_xy: float,
    step_z: float,
    step_rz_deg: float,
) -> np.ndarray:
    frame = frame_bgr.copy()
    lines = [
        f"{camera_name}  WASD: XY  Q/E: yaw(Z)  R/F: Z",
        f"step_xy={step_xy:.4f} m  step_z={step_z:.4f} m  step_rz={step_rz_deg:.1f} deg  ESC/X: quit",
        f"pose: {format_vector(pose, precision=3)}",
        f"joints: {format_vector(joints, precision=3)}",
    ]
    y = 24
    for line in lines:
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 28
    return frame


def get_frame_bgr(camera: Any, camera_name: str) -> np.ndarray:
    try:
        _, color_rgb = camera.get_rgbd()
        return color_rgb[:, :, ::-1].copy()
    except Exception as exc:
        return make_status_frame(message=f"{camera_name}: {exc}")


def move_step(
    arm: Any,
    delta_xyz: Tuple[float, float, float],
    delta_rz_rad: float,
    speed: float,
    accel: float,
    camera: Any,
    camera_name: str,
    window_name: str,
    step_xy: float,
    step_z: float,
    step_rz_deg: float,
) -> Tuple[bool, list[float], Optional[list[float]]]:
    current_pose, current_joints = fetch_state(arm)
    target = list(current_pose)
    target[0] += float(delta_xyz[0])
    target[1] += float(delta_xyz[1])
    target[2] += float(delta_xyz[2])
    target[5] += float(delta_rz_rad)

    reachable, analysis = arm.check_pose_reachable(target)
    if not reachable:
        print(f"[WARN] Target pose unreachable: {target}")
        if analysis is not None:
            print(f"[WARN] Reachability detail: {analysis}")
        return False, current_pose, current_joints

    arm.moveL(target, speed=speed, accel=accel, async_=True)
    while arm.getIsMoving():
        frame_bgr = get_frame_bgr(camera, camera_name)
        frame_bgr = draw_overlay(
            frame_bgr,
            camera_name,
            current_pose,
            current_joints,
            step_xy,
            step_z,
            step_rz_deg,
        )
        cv2.imshow(window_name, frame_bgr)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("x"), ord("X")):
            arm.stop()
            return True, current_pose, current_joints
        time.sleep(0.01)

    new_pose, new_joints = fetch_state(arm)
    print_state(new_pose, new_joints)
    return False, new_pose, new_joints


def parse_args() -> argparse.Namespace:
    cfg = load_hardware_config()
    parser = argparse.ArgumentParser(
        description="Keyboard jog utility with live arm camera preview and robot state printouts."
    )
    parser.add_argument(
        "--server-url",
        default=build_default_server_url(cfg),
        help="Robot gateway base URL. Defaults to hardware/config/config.yaml.",
    )
    parser.add_argument(
        "--camera-name",
        default='camera_drill',
        help="camera_gripper, camera_drill, camera_fixed",
    )
    parser.add_argument(
        "--tool-name",
        default="tcp_gripper",
        help="Tool profile passed to URArmControlClient.",
    )
    parser.add_argument(
        "--step-xy",
        type=float,
        default=0.02,
        help="Jog step in metres for X/Y moves.",
    )
    parser.add_argument(
        "--step-z",
        type=float,
        default=0.005,
        help="Jog step in metres for Z moves.",
    )
    parser.add_argument(
        "--step-rz-deg",
        type=float,
        default=5.0,
        help="Jog step in degrees for rotation about tool Z.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=0.05,
        help="moveL speed in m/s.",
    )
    parser.add_argument(
        "--accel",
        type=float,
        default=0.20,
        help="moveL accel in m/s^2.",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=30,
        help="Camera warmup frame count.",
    )
    parser.add_argument(
        "--window-name",
        default=DEFAULT_WINDOW_NAME,
        help="OpenCV window title.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from hardware.camera_rs_client import NetworkRealSenseCamera
    except ModuleNotFoundError as exc:
        if exc.name == "zmq":
            raise SystemExit("[ERROR] Missing dependency 'pyzmq'. Install it to use the arm camera viewer.") from exc
        raise

    from hardware.urarm_control_client import URArmControlClient

    cfg = load_hardware_config()

    camera_cfg = cfg.get(args.camera_name)
    if not isinstance(camera_cfg, dict):
        raise SystemExit(f"[ERROR] Camera config '{args.camera_name}' not found in hardware/config/config.yaml")

    camera = NetworkRealSenseCamera(
        intrinsics=camera_cfg.get("intrinsics"),
        server_ip=camera_cfg["network_camera_ip"],
        port=int(camera_cfg["network_camera_port"]),
        camera_id=camera_cfg.get("serial"),
    )
    arm = URArmControlClient(server_url=args.server_url, tool_name=args.tool_name)

    print("Controls:")
    print("  W/S: +Y / -Y")
    print("  A/D: -X / +X")
    print(f"  Q/E: +Rz / -Rz ({args.step_rz_deg:.1f} deg)")
    print(f"  R/F: +Z / -Z ({args.step_z:.4f} m)")
    print("  P: refresh printed state")
    print("  ESC or X: quit")

    try:
        camera.start()
        camera.warmup(args.warmup_frames)

        pose, joints = fetch_state(arm)
        print_state(pose, joints)

        key_to_delta = {
            ord("w"): (0.0, +args.step_xy, 0.0),
            ord("W"): (0.0, +args.step_xy, 0.0),
            ord("s"): (0.0, -args.step_xy, 0.0),
            ord("S"): (0.0, -args.step_xy, 0.0),
            ord("a"): (-args.step_xy, 0.0, 0.0),
            ord("A"): (-args.step_xy, 0.0, 0.0),
            ord("d"): (+args.step_xy, 0.0, 0.0),
            ord("D"): (+args.step_xy, 0.0, 0.0),
            ord("r"): (0.0, 0.0, +args.step_z),
            ord("R"): (0.0, 0.0, +args.step_z),
            ord("f"): (0.0, 0.0, -args.step_z),
            ord("F"): (0.0, 0.0, -args.step_z),
        }
        key_to_rz = {
            ord("q"): +np.deg2rad(args.step_rz_deg),
            ord("Q"): +np.deg2rad(args.step_rz_deg),
            ord("e"): -np.deg2rad(args.step_rz_deg),
            ord("E"): -np.deg2rad(args.step_rz_deg),
        }

        while True:
            frame_bgr = get_frame_bgr(camera, args.camera_name)
            frame_bgr = draw_overlay(
                frame_bgr,
                args.camera_name,
                pose,
                joints,
                args.step_xy,
                args.step_z,
                args.step_rz_deg,
            )
            cv2.imshow(args.window_name, frame_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("x"), ord("X")):
                break
            if key in (ord("p"), ord("P")):
                pose, joints = fetch_state(arm)
                print_state(pose, joints)
                continue

            delta = key_to_delta.get(key)
            delta_rz = key_to_rz.get(key, 0.0)
            if delta is None and abs(delta_rz) < 1e-12:
                continue
            if delta is None:
                delta = (0.0, 0.0, 0.0)

            should_exit, pose, joints = move_step(
                arm=arm,
                delta_xyz=delta,
                delta_rz_rad=delta_rz,
                speed=args.speed,
                accel=args.accel,
                camera=camera,
                camera_name=args.camera_name,
                window_name=args.window_name,
                step_xy=args.step_xy,
                step_z=args.step_z,
                step_rz_deg=args.step_rz_deg,
            )
            if should_exit:
                break
    finally:
        try:
            camera.stop()
        except Exception:
            pass
        try:
            arm.close()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
