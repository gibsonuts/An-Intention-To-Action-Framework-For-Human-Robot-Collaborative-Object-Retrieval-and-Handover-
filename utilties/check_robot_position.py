#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from typing import Any, Dict

import yaml

from hardware.urarm_control_client import URArmControlClient


def load_hardware_config() -> Dict[str, Any]:
    config_path = ROOT / "hardware" / "config" / "config.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_default_server_url(cfg: Dict[str, Any]) -> str:
    robot_cfg = cfg.get("robot", {}) or {}
    host = robot_cfg.get("network_robot_ip")
    port = robot_cfg.get("network_robot_port")
    if not host or not port:
        raise ValueError("robot.network_robot_ip or robot.network_robot_port is missing from hardware/config/config.yaml")
    return f"http://{host}:{port}"


def parse_args() -> argparse.Namespace:
    cfg = load_hardware_config()
    parser = argparse.ArgumentParser(description="Check robot TCP position through URArmControlClient.")
    parser.add_argument(
        "--server-url",
        default=build_default_server_url(cfg),
        help="Robot gateway base URL. Defaults to the network robot config.",
    )
    parser.add_argument(
        "--tool-name",
        default="tcp_drill",
        help="Tool profile passed to URArmControlClient.",
    )
    return parser.parse_args()


def get_robot_position_from_client(server_url: str, tool_name: str) -> None:
    client = URArmControlClient(server_url=server_url, tool_name=tool_name)

    try:
        tcp_pose = client.get_tcp_pose_axis_angle()
        print("Current TCP Pose [x, y, z, rx, ry, rz]:", tcp_pose)

        try:
            tool_io = client.get_tool_io()
            print("Tool trigger state:", tool_io)
        except Exception as exc:
            print("Warning: failed to read tool IO:", exc)

        try:
            stop_io = client.get_stop_io()
            print("Stop button state:", stop_io)
        except Exception as exc:
            print("Warning: failed to read stop IO:", exc)

        print("Note: URArmControlClient currently exposes TCP pose, not raw joint positions.")
    finally:
        try:
            client.h.close()
        except Exception:
            pass


if __name__ == "__main__":
    args = parse_args()
    get_robot_position_from_client(args.server_url, args.tool_name)
