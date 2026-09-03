#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UR10e RTDE client that forwards commands to a remote HTTP server.

This preserves your original CLI shape where possible, but routes robot
operations to a gateway server exposing basic RTDE endpoints.

Dependencies:
  - httpx
  - pyyaml
  - numpy

Optional env vars:
  UR_SERVER_URL="http://SERVER:8000"
  UR_API_KEY="change-me"

YAML (config/arm_control.yaml):
  robot:
    server_url: "http://SERVER:8000"
    api_key: "change-me"
  locations:
    ready: [ ... 6 joint values ... ]
  tool:
    tcp_pose: [x,y,z,rx,ry,rz]

Safety:
  - Start with low speeds/accel.
  - Network jitter makes high-rate streaming control unsafe. `moveI`/`holdI`
    are intentionally disabled in remote mode.
"""

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import time
import math
import argparse
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import yaml
from threading import Event, Lock
import requests
import httpx

# Optional utils (only used for get_T_base_tcp); if absent, those calls are unused.
try:
    from commons.grasp_utils import pose_from_T, make_T, load_yaml_pose, check_path_exists
except Exception:
    # Minimal fallback for check_path_exists / make_T
    def check_path_exists(rel_path: str, this_file: str):
        here = Path(this_file).resolve().parent
        cand = here / rel_path
        if cand.exists():
            return cand
        cand2 = here.parent / rel_path
        return cand2 if cand2.exists() else None

    def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
        T = np.eye(4)
        T[:3,:3] = R
        T[:3,3] = t
        return T

G = 9.81
CFG_PATH = 'config/arm_control.yaml'


# -----------------------------
# Small math + helpers
# -----------------------------
def vec_norm(v):
    return math.sqrt(sum(float(x) * float(x) for x in v))


def clamp_vec(v, max_norm):
    n = vec_norm(v)
    if n <= max_norm or n == 0.0:
        return list(v)
    s = max_norm / n
    return [float(x) * s for x in v]


def lpf(prev: np.ndarray, new: np.ndarray, alpha: float) -> np.ndarray:
    return alpha * new + (1.0 - alpha) * prev


def rotvec_to_R(rx, ry, rz):
    """Axis-angle (UR pose rx,ry,rz) -> rotation matrix (tool->base)."""
    theta = math.sqrt(rx*rx + ry*ry + rz*rz)
    if theta < 1e-12:
        return np.eye(3)
    k = np.array([rx, ry, rz], dtype=float) / theta
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]], dtype=float)
    return np.eye(3) + math.sin(theta)*K + (1.0 - math.cos(theta))*(K @ K)


def pose_err(target: List[float], current: List[float]) -> List[float]:
    """6D pose error in base frame using small-angle for rotation vectors."""
    dx = [target[i] - current[i] for i in range(3)]
    dr = [target[i] - current[i] for i in range(3, 6)]
    return dx + dr


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    return data


def get(cfg: Dict[str, Any], path: str, default=None):
    """Nested getter: path like 'motion.moveI.kp_lin'."""
    cur = cfg
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur

def _make_session(api_key: Optional[str]) -> requests.Session:
    s = requests.Session()
    if api_key:
        s.headers.update({"X-API-Key": api_key})
    s.headers.update({"Content-Type": "application/json"})
    return s

class URDashboardRemote:
    """
    Minimal wrapper over server /dash/* endpoints.
    Methods return server JSON (dict) and raise on HTTP errors.
    """
    def __init__(
        self,
        base_url: str,
        timeout: float = 6.0,
        api_key: Optional[str] = None,
        stop_program_on_init: bool = True,
    ):
        self.base = base_url.rstrip("/")
        self.s = _make_session(api_key)
        self.timeout = timeout
        if stop_program_on_init:
            self.stop()
        

    # helpers
    def _post(self, path: str, payload: Optional[dict] = None):
        payload = payload or {}
        r = self.s.post(f"{self.base}{path}", json=payload, timeout=self.timeout)
        if not r.ok:
            raise RuntimeError(
                f"Dashboard request {path} failed with HTTP {r.status_code}: "
                f"{r.text.strip() or '<empty response>'}"
            )
        return r.json()

    def _get(self, path: str):
        r = self.s.get(f"{self.base}{path}", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def wait_for_program(
        self,
        *,
        start_timeout: float = 10.0,
        finish_timeout: float = 300.0,
        poll_interval: float = 0.25,
    ) -> None:
        """Wait until a dashboard program is observed playing and then stopped."""
        start_timeout = max(0.1, float(start_timeout))
        finish_timeout = max(0.1, float(finish_timeout))
        poll_interval = max(0.05, float(poll_interval))
        start_deadline = time.monotonic() + start_timeout
        last_state = "unknown"

        while time.monotonic() < start_deadline:
            playing = bool(self.is_playing())
            state_response = self.program_state()
            last_state = str(state_response.get("programState", state_response))
            if playing or last_state.upper().startswith("PLAYING"):
                print(f"[INFO] Dashboard program started: {last_state}")
                break
            time.sleep(poll_interval)
        else:
            raise RuntimeError(
                f"Dashboard program did not enter PLAYING within {start_timeout:.1f}s; "
                f"last state was {last_state!r}."
            )

        finish_deadline = time.monotonic() + finish_timeout
        next_status_log = 0.0
        while time.monotonic() < finish_deadline:
            playing = bool(self.is_playing())
            state_response = self.program_state()
            last_state = str(state_response.get("programState", state_response))
            if not playing and last_state.upper().startswith("STOPPED"):
                print(f"[INFO] Dashboard program fully finished: {last_state}")
                return
            now = time.monotonic()
            if now >= next_status_log:
                print(f"[INFO] Waiting for dashboard program to finish: {last_state}")
                next_status_log = now + 1.0
            time.sleep(poll_interval)

        raise RuntimeError(
            f"Dashboard program did not finish within {finish_timeout:.1f}s; "
            f"last state was {last_state!r}."
        )

    # core ops
    def connect(self):             return self._post("/dash/connect")
    def disconnect(self):          return self._post("/dash/disconnect")
    def status(self):              return self._get("/dash/status")
    def load_urp(self, program):   return self._post("/dash/load_urp", {"program": program})
    def play(self):                return self._post("/dash/play")
    def is_playing(self):          return self._get("/dash/is_playing")
    def stop(self):                return self._post("/dash/stop")
    def pause(self):               return self._post("/dash/pause")
    def close_popup(self):         return self._post("/dash/close_popup")
    def popup(self, msg):          return self._post("/dash/popup", {"msg": msg})
    def add_to_log(self, msg):     return self._post("/dash/add_to_log", {"msg": msg})
    def unlock_protective_stop(self): return self._post("/dash/unlock_protective_stop")
    def power_on(self):            return self._post("/dash/power_on")
    def power_off(self):           return self._post("/dash/power_off")
    def brake_release(self):       return self._post("/dash/brake_release")
    def robot_mode(self):          return self._get("/dash/robot_mode")
    def safety_mode(self):         return self._get("/dash/safety_mode")
    def program_state(self):       return self._get("/dash/program_state")



# -----------------------------
# HTTP-backed Controller
# -----------------------------
class URArmControlClient:
    def __init__(
        self,
        server_url: Optional[str] = None,
        timeout: float = 20.0,
        api_key: Optional[str] = None,
        tool_name: Optional[str] = None,
    ):
        """
        server_url: base URL to the robot server, e.g. http://192.168.10.100:8000
        """
        cfg = {}
        cfg_file = check_path_exists(CFG_PATH, __file__)
        if cfg_file:
            with cfg_file.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        else:
            print('WARN: no cfg file found:', CFG_PATH)

        self.cfg = cfg
        self.robot = get(cfg, "robot", {}) or {}
        self.locations = get(cfg, "locations", {}) or {}

        # TCP remains selectable by tool; payload comes from the UR installation.
        self.tcp_pose = None
        if tool_name:
            tool = get(cfg, f"tool.{tool_name}", {}) or {}
            if 'tcp_pose' in tool:
                self.tcp_pose = tool.get('tcp_pose', None)

 
        # Resolve server URL & API key (CLI/env override YAML)
        self.server_url = (
            server_url
            or os.environ.get("UR_SERVER_URL")
            or self.robot.get("server_url")
        )
        if not self.server_url:
            print("ERROR: Missing server URL. Provide --server, UR_SERVER_URL, or robot.server_url in YAML.")
            sys.exit(1)

        self.api_key = (
            api_key
            or os.environ.get("UR_API_KEY")
            or self.robot.get("api_key")
        )

        headers = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key

        self.h = httpx.Client(
            base_url=self.server_url.rstrip("/"),
            timeout=timeout,
            headers=headers or None,
        )

        # Optional debug
        self._debug = bool(get(self.cfg, "debug", False))

        # Bias bookkeeping (kept client-side)
        self.ft_bias = np.zeros(6, dtype=float)

        # Not used remotely but kept for API compatibility
        self._stop_event = Event()
        self._loop_lock = Lock()
        self._loop_name: Optional[str] = None
        self._script_active: bool = False
        self.servo_running = False
        self.dt = 1.0 / float(get(self.robot, "watchdog_hz", 100.0))
      
        # Eager connect / health check – FATAL if server not responding
        try:
            self._post("/connect")
            time.sleep(1.0)  # wait a bit for server to settle
        except Exception as e:
            print(f"ERROR: Could not reach UR server at {self.server_url}. "
                  f"Is the gateway running and reachable?\nDetails: {e}")
            sys.exit(1)

        # Cache the payload from the active UR installation without overwriting it.
        try:
            installation_mass, installation_cog = self.update_payload_from_rtde()
            self.tool_mass = installation_mass
            self.tool_cog = np.array(installation_cog, dtype=float)
            print(
                "[INFO] Using payload from active UR installation: "
                f"{self.tool_mass:.3f} kg, CoG={self.tool_cog.tolist()}"
            )
        except Exception as e:
            raise RuntimeError(
                "Failed to read payload from the active UR installation; "
                "refusing to apply a fallback payload."
            ) from e

        # Apply the selected TCP if present. Payload remains installation-owned.
        if self.tcp_pose is not None:
            try:
                print("Setting TCP from config:", self.tcp_pose)
                self.set_tcp([float(x) for x in self.tcp_pose])
            except Exception as e:
                print("WARN: set_tcp failed:", e)

    def connect(self):
        """(Re)connect to the server (eagerly done in __init__)."""
        try:
            self._post("/connect")
            print("[INFO] Connected to UR server.")
        except Exception as e:
            print(f"ERROR: Failed to connect to UR server: {e}")
            raise

    def check_pose_reachable(self,pose: List[float], qnear: List[float] | None = None) -> Tuple[bool, Dict[str, Any] | None]:
        """
        Call the FastAPI server /check_pose endpoint to see if a pose is reachable.
        Returns (reachable: bool, analysis: dict|None).
        Retries transient gateway failures before reporting the pose as unreachable.
        """
        print('checking pose reachable', pose, qnear)
        payload: Dict[str, Any] = {"pose": pose}
        if qnear is not None:
            payload["qnear"] = qnear

        max_attempts = 3
        retry_delay_s = 0.25

        for attempt in range(1, max_attempts + 1):
            try:
       
                data = self._post("/check_pose", payload)
                reachable = bool(data.get("reachable", False))
                if not reachable:
                    print(f"[INFO] /check_pose reports unreachable: {data}")
                return reachable, data
            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code if e.response is not None else None
                if status_code == 503 and attempt < max_attempts:
                    print(
                        f"[WARN] /check_pose returned 503 on attempt {attempt}/{max_attempts}; retrying in {retry_delay_s:.2f}s."
                    )
                    time.sleep(retry_delay_s)
                    continue
                print(f"[WARN] Failed to call /check_pose (): {e}")
                return False, {"error": str(e), "status_code": status_code}
            except Exception as e:
                print(f"[WARN] Failed to call /check_pose (): {e}")
                return False, {"error": str(e)}

        return False, {"error": "check_pose retries exhausted"}
        
    # ------------ HTTP helpers ------------
    def _get(self, path: str, params: Dict[str, Any] = None):
        r = self.h.get(path, params=params or {})
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, json: Dict[str, Any] = None):
        # print('post',path,json)
        r = self.h.post(path, json=json or {})
        r.raise_for_status()
        return r.json()


    # ----------- Setup / Calibration (remote wrappers) -----------
    def set_tcp(self, tcp_pose_axis_angle: List[float]) -> Dict[str, Any]:
        tcp = [float(value) for value in tcp_pose_axis_angle]
        response = self._post("/set_tcp", {"tcp": tcp})
        if isinstance(response, dict) and response.get("ok") is False:
            raise RuntimeError(f"Robot server rejected set_tcp: {response!r}")
        self.tcp_pose = tcp
        return response

    def set_tool_payload(self, mass_kg: float, cog_tool: List[float]):
        """Set gripper-only payload on the controller via server."""
        self.tool_mass = float(mass_kg)
        self.tool_cog = np.array(cog_tool, dtype=float)
        self._post("/set_payload", {"mass": self.tool_mass, "cog": self.tool_cog.tolist()})

    def update_payload_from_rtde(self):
        """Read active payload mass/CoG via server and cache them client-side."""
        m = float(self._get("/get_payload")["mass"])
        c = np.array(self._get("/get_payload_cog")["cog"], float)
        self.active_mass = m
        self.active_cog = c
        return m, c.tolist()

    def set_active_payload(self, total_mass_kg: float, total_cog_tool: List[float]):
        """Sets combined payload on the robot (alias to set_tool_payload remotely)."""
        self._post("/set_payload", {"mass": float(total_mass_kg), "cog": list(map(float, total_cog_tool))})

    def measure_wrench_bias(self, seconds: Optional[float] = None, dt: Optional[float] = None) -> List[float]:
        """Average the TCP wrench via server to form a bias (network-latency tolerant)."""
        seconds = seconds if seconds is not None else float(get(self.cfg, "calibration.bias_seconds", 0.7))
        dt = dt if dt is not None else float(get(self.cfg, "calibration.bias_dt", 0.02))  # slower to be LAN friendly
        t_end = time.time() + max(0.1, seconds)
        ss = []
        while time.time() < t_end:
            ss.append(np.array(self._get("/get_wrench")["wrench"], float))
            time.sleep(dt)
        self.ft_bias = np.mean(ss, axis=0) if ss else np.zeros(6)
        return self.ft_bias.tolist()

    # ----------- Motion primitives (remote) -----------
    def _cancel_if_streaming(self, ignore_reset=False):
        # Remote server doesn't track our client loops; just stop speed if asked
        if not ignore_reset and self.servo_running:
            self.servo_running = False
            try:
                self._post("/speed_stop")
            except Exception:
                pass

    def wait_for_move(self): 
        time.sleep(1.0)
        while self.getIsMoving():
            time.sleep(0.1)
        print('robot not moving')

    def getIsMoving(self) -> dict:
        return self._get("/is_moving")["isMoving"]
    
    def get_stop_io(self) -> bool:
        return  bool(self._get(f"/stop_io")["state"])

    def get_tool_io(self) -> bool:
        return  bool(self._get(f"/tool_io")["state"])

    def stop_control_script(self) -> Dict[str, Any]:
        """Release the RTDE control script before starting a PolyScope URP."""
        return self._post("/stop_script")

    def moveJ(self, q: List[float], speed: Optional[float] = None, accel: Optional[float] = None, async_: bool = False):
        if self._debug:
            print('moveJ', q)
        self._cancel_if_streaming()
        speed = float(speed if speed is not None else get(self.cfg, "robot.j_speed", 1.2))
        accel = float(accel if accel is not None else get(self.cfg, "robot.j_accel", 2.5))
        payload = {"q": q, "speed": speed, "accel": accel, "async_": bool(async_)}
        response = self._post("/movej", payload)
        if not response.get("ok", True):
            try:
                status = self._get("/status")
            except Exception as e:
                status = {"status_error": str(e)}
            raise RuntimeError(
                "moveJ was rejected by the robot server. "
                f"response={response!r}, status={status!r}, command={payload!r}. "
                "Confirm that the UR pendant is in Remote Control mode and that "
                "the robot is motion-ready."
            )
        if not async_:
            self.wait_for_move()
     

    def moveL(self, pose: List[float], speed: Optional[float] = None, accel: Optional[float] = None, async_: bool = False):
        if self._debug:
            print('moveL', pose)
        self._cancel_if_streaming()
        speed = float(speed if speed is not None else get(self.cfg, "robot.l_speed", 0.25))
        accel = float(accel if accel is not None else get(self.cfg, "robot.l_accel", 1.2))
        payload = {"pose": pose, "speed": speed, "accel": accel, "async_": bool(async_)}
        print(payload)
        response = self._post("/movel", payload)
        if not response.get("ok", True):
            try:
                status = self._get("/status")
            except Exception as e:
                status = {"status_error": str(e)}
            raise RuntimeError(
                "moveL was rejected by the robot server. "
                f"response={response!r}, status={status!r}, command={payload!r}. "
                "Confirm that the UR pendant is in Remote Control mode and that "
                "the robot is motion-ready."
            )
        if not async_:
            self.wait_for_move()

    def moveLPath(self, path: List[List[float]],  async_: bool = False):
        # Not supported on minimal server; emulate by sequential moveL calls.
        self._cancel_if_streaming()
        payload = {"path": path,"async_": bool(async_)}
        if not self._post("/movelpath", payload).get("ok", True):
            print('faild to call moveLPath')
            return 
        if not async_:
            self.wait_for_move()

    def servoL(self, pose: List[float], time_s: float,  speed: float = 0.25, accel: float = 0.5, lookahead_time: float = 0.1, gain: float = 300.0) -> None:
        self._cancel_if_streaming(ignore_reset=True)
        self._post("/servol", {"target": pose, "speed": speed, "accel": accel, "time": time_s,
                               "lookahead_time": lookahead_time, "gain": gain})
        self.servo_running = True

    def servoJ(self, joints: List[float], speed: float = 0.25, accel: float = 0.5, time_s: float = 0.0, lookahead_time: float = 0.1, gain: float = 300.0) -> None:
        self._cancel_if_streaming(ignore_reset=True)
        self._post("/servoj", {"target": joints, "speed": speed, "accel": accel, "time": time_s,
                               "lookahead_time": lookahead_time, "gain": gain})

    def waitForMotion(self):
        # Poll server isSteady
        while not self._get("/is_steady")["isSteady"]:
            time.sleep(0.02)

    def running(self):
        return not self._get("/is_steady")["isSteady"]

    # ----------- Unsupported high-rate loops over HTTP -----------
    def moveI(self, *args, **kwargs):
        raise RuntimeError("moveI is disabled in remote mode (requires real-time loop). Use moveL or server-side logic.")

    def holdI(self, *args, **kwargs):
        raise RuntimeError("holdI is disabled in remote mode (requires real-time loop). Use moveL or server-side logic.")

    # ----------- Data access -----------
    def get_wrench_raw(self) -> List[float]:
        return np.array(self._get("/get_wrench")["wrench"], float)

    def get_wrench(self) -> List[float]:
        w = np.array(self._get("/get_wrench")["wrench"], float) - self.ft_bias
        return w

    def get_actual_q(self) -> List[float]:
        return self._get("/get_q")["q"]

    def get_joint_positions(self) -> List[float]:
        return self.get_actual_q()

    def get_tcp_pose_axis_angle(self) -> List[float]:
        return self._get("/get_pose")["pose"]

    def get_T_base_tcp(self) -> np.ndarray:
        p = self.get_tcp_pose_axis_angle()  # [x,y,z,rx,ry,rz]
        x, y, z, rx, ry, rz = p
        angle = math.sqrt(rx*rx + ry*ry + rz*rz) + 1e-12
        ax = np.array([rx, ry, rz]) / angle
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        R = np.eye(3) + math.sin(angle)*K + (1-math.cos(angle))*(K@K)
        T = make_T(R, np.array([x, y, z]))
        return T
    
    def rodrigues(self,axis: np.ndarray, angle: float) -> np.ndarray:
        """Axis–angle to rotation matrix."""
        axis = axis / np.linalg.norm(axis)
        x, y, z = axis
        K = np.array([
            [0,   -z,   y],
            [z,    0,  -x],
            [-y,   x,   0],
        ])
        I = np.eye(3)
        return I + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

    def get_T_base_tcp_point_down(self,
                                z_target: np.ndarray = np.array([0.0, 0.0, -1.0])
                                ) -> np.ndarray:
            """
            Given T_base_tcp, return a new transform whose TCP +Z axis points 'down'
            (z_target, default [0,0,-1] in base frame), with minimal rotation.
            """
            T = self.get_T_base_tcp()
            R = T[:3, :3]

            # Current TCP Z in base frame
            z_cur = R[:, 2]
            z_cur = z_cur / np.linalg.norm(z_cur)

            # Desired Z direction in base frame
            z_tgt = z_target / np.linalg.norm(z_target)

            # If already aligned
            dot = np.clip(np.dot(z_cur, z_tgt), -1.0, 1.0)
            if np.isclose(dot, 1.0, atol=1e-6):
                # No change
                return T

            # 180 deg case (opposite vectors): pick any axis orthogonal to z_cur
            if np.isclose(dot, -1.0, atol=1e-6):
                # choose an arbitrary axis not parallel to z_cur
                axis = np.array([1.0, 0.0, 0.0])
                if abs(np.dot(axis, z_cur)) > 0.9:
                    axis = np.array([0.0, 1.0, 0.0])
                axis = axis - np.dot(axis, z_cur) * z_cur
                axis = axis / np.linalg.norm(axis)
                angle = np.pi
                R_corr = self.rodrigues(axis, angle)
            else:
                # General case
                axis = np.cross(z_cur, z_tgt)
                axis_norm = np.linalg.norm(axis)
                if axis_norm < 1e-8:
                    # Numerically degenerate but not caught by dot≈±1
                    return T
                axis = axis / axis_norm
                angle = np.arccos(dot)
                R_corr = self.rodrigues(axis, angle)

            R_new = R_corr @ R
            T[:3, :3] = R_new
            return T

    # ----------- Stop -----------
    def stop(self) -> None:
        """Cooperatively stop any speed commands."""
        try:
            self._post("/speed_stop")
        except Exception:
            pass
        try:
            self._post("/stop_script")
        except Exception:
            pass

    def close(self) -> None:
        try:
            self.h.close()
        except Exception:
            pass


# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="UR10e RTDE remote client (HTTP-backed).")
    parser.add_argument("--server", help="Base URL of robot server, e.g. http://192.168.10.100:8000")
    parser.add_argument("--api-key", help="X-API-Key header for server", default=None)

    sub = parser.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("calibrate", help="(remote) Calibrate payload via sequence (runs slowly over network).")
    c.add_argument("--assume-rz", type=float, help="Assume CoG z (m); statically unobservable.", default=0.0)
    c.add_argument("--speed", type=float, help="Joint speed for sampling moves (rad/s).")
    c.add_argument("--accel", type=float, help="Joint accel for sampling moves (rad/s^2).")
    c.add_argument("--settle", type=float, help="Settle time at each pose (s).", default=0.2)
    c.add_argument("--avg-sec", type=float, help="Averaging duration at each pose (s).", default=0.15)

    j = sub.add_parser("location", help="move to location from YAML.")
    j.add_argument("name",  type=str)

    j = sub.add_parser("movej", help="Joint move to q1..q6 (rad).")
    j.add_argument("q", nargs=6, type=float)
    j.add_argument("--speed", type=float)
    j.add_argument("--accel", type=float)
    j.add_argument("--async", dest="async_", action="store_true")

    l = sub.add_parser("movel", help="Linear move to [x y z rx ry rz].")
    l.add_argument("pose", nargs=6, type=float)
    l.add_argument("--speed", type=float)
    l.add_argument("--accel", type=float)
    l.add_argument("--async", dest="async_", action="store_true")

    i = sub.add_parser("movei", help="(disabled) Impedance servo to [x y z rx ry rz].")
    i.add_argument("pose", nargs=6, type=float)
    i.add_argument("--auto-hold", action="store_true")
    i.add_argument("--max-time", type=float)

    h = sub.add_parser("holdi", help="(disabled) Impedance hold at current pose.")
    h.add_argument("--max-time", type=float)

    s = sub.add_parser("set_tool_payload", help="Set gripper-only mass and CoG (tool frame).")
    s.add_argument("mass", type=float)
    s.add_argument("cog", nargs=3, type=float)

    u = sub.add_parser("update_payload_from_rtde", help="Read active payload mass & CoG from controller.")

    e = sub.add_parser("estimate_object_payload", help="Estimate object mass/CoG then update total payload (slow).")
    e.add_argument("--samples", type=int)
    e.add_argument("--assume-rz", type=float)

    st = sub.add_parser("stop", help="Stop any running speeds and halt the script.")
    gp = sub.add_parser("get_pose", help="Get actual TCP pose.")
    gw = sub.add_parser("get_wrench", help="Get current TCP wrench.")
    args = parser.parse_args()

    ur = URArmControlClient(server_url=args.server, api_key=args.api_key)

    # Convenience: print a default 'ready' location if present
    if ur.locations and "ready" in ur.locations:
        print("ready:", ur.locations["ready"])

    if args.cmd == "location":
        loc = ur.locations.get(str(args.name))
        if loc is None:
            raise SystemExit(f"Unknown location '{args.name}'. Check YAML.")
        ur.moveJ(loc)

    elif args.cmd == "calibrate":
        # Remote-friendly version of calibrate_payload_sequence:
        # executes joint moves and samples wrench via server. Slower than local.
        m_obj, cog_obj, m_tot, cog_tot = calibrate_payload_sequence_remote(
            ur,
            speed=args.speed,
            accel=args.accel,
            settle=args.settle,
            avg_sec=args.avg_sec,
            assume_rz=args.assume_rz,
        )
        print(f"Estimated object mass: {m_obj:.3f} kg  CoG(tool): {cog_obj}")
        print(f"Updated total payload: {m_tot:.3f} kg  CoG(tool): {cog_tot}")

    elif args.cmd == "movej":
        ur.moveJ(args.q, speed=args.speed, accel=args.accel, async_=args.async_)

    elif args.cmd == "movel":
        ur.moveL(args.pose, speed=args.speed, accel=args.accel, async_=args.async_)

    elif args.cmd == "movei":
        raise SystemExit("moveI is disabled in remote mode. Use 'movel' or implement server-side impedance.")

    elif args.cmd == "holdi":
        raise SystemExit("holdI is disabled in remote mode. Use 'movel' or implement server-side hold.")

    elif args.cmd == "set_tool_payload":
        ur.set_tool_payload(args.mass, args.cog)
        print("Tool payload set.")

    elif args.cmd == "update_payload_from_rtde":
        m, c = ur.update_payload_from_rtde()
        print(f"Active payload: {m:.3f} kg, CoG(tool) = {c}")

    elif args.cmd == "estimate_object_payload":
        m_obj, cog_obj, m_tot, cog_tot = estimate_object_payload_remote(
            ur,
            samples=args.samples,
            assume_rz=args.assume_rz
        )
        print(f"Estimated object mass: {m_obj:.3f} kg  CoG(tool): {cog_obj}")
        print(f"Updated total payload: {m_tot:.3f} kg  CoG(tool): {cog_tot}")

    elif args.cmd == "stop":
        ur.stop()
        print("Stop requested (speedStop + stopScript).")

    elif args.cmd == "get_pose":
        print(ur.get_tcp_pose_axis_angle())

    elif args.cmd == "get_wrench":
        print(ur.get_wrench_raw().tolist())

# ---------- Remote-friendly versions of calibration helpers ----------
def calibrate_payload_sequence_remote(
    ur: URArmControlClient,
    speed: Optional[float] = None,
    accel: Optional[float] = None,
    settle: float = 0.2,
    avg_sec: float = 0.15,
    assume_rz: float = 0.0,
) -> Tuple[float, List[float], float, List[float]]:
    """
    Remote version: performs the same sampling sequence as your local method,
    but uses the server for moves and wrench/pose reads. Slower due to network.
    """
    # 0) Bias FT with object held in free space
    ur.measure_wrench_bias(
        seconds=float(get(ur.cfg, "calibration.bias_seconds", 0.7)),
        dt=float(get(ur.cfg, "calibration.bias_dt", 0.02)),
    )

    # 1) Build the joint targets from current q isn't available via basic server.
    # Use a simplified plan: sample current pose orientation changes by small movel offsets (safer remotely).
    # If you must do wrist joint samples, extend your server to expose getActualQ and moveJ targets accordingly.
    # For now, take 4 samples at the current pose (no motion), which still lets us get an approximate mass.
    targets = [None] * 4  # None => stay put

    forces = []
    taus_b = []
    R_list = []

    for _ in targets:
        time.sleep(max(0.05, settle))
        # average over avg_sec
        t_end = time.time() + max(0.05, avg_sec)
        acc = np.zeros(6)
        n = 0
        while time.time() < t_end:
            w = np.array(ur._get("/get_wrench")["wrench"], float) - ur.ft_bias
            acc += w
            n += 1
            time.sleep(0.02)
        wrench = acc / max(1, n)

        pose = ur.get_tcp_pose_axis_angle()
        R = rotvec_to_R(pose[3], pose[4], pose[5])  # tool->base

        forces.append(wrench[:3])
        taus_b.append(wrench[3:])
        R_list.append(R)

    forces = np.array(forces)
    taus_b = np.array(taus_b)

    # 3) Mass from force projection onto tool -Z in base
    Rz = np.array([R[:, 2] for R in R_list])     # tool z (col 2) in base
    proj = np.einsum('ij,ij->i', forces, Rz)     # Fi · Rz_i
    m_obj = -float(np.mean(proj)) / G

    # 4) Torques -> r_x, r_y (in tool frame); r_z assumed
    tau_tool = np.array([R.T @ tb for R, tb in zip(R_list, taus_b)])
    mg = max(m_obj * G, 1e-6)
    r_x = float(np.mean(tau_tool[:, 1]) / mg) if mg > 1e-6 else 0.0
    r_y = float(-np.mean(tau_tool[:, 0]) / mg) if mg > 1e-6 else 0.0
    r_z = float(assume_rz)
    cog_obj_tool = np.array([r_x, r_y, r_z], float)

    # 5) Combine with known tool baseline for total payload
    m_total = ur.tool_mass + m_obj
    cog_total_tool = (ur.tool_mass*ur.tool_cog + m_obj*cog_obj_tool) / max(m_total, 1e-6)

    # Apply to robot
    ur.set_active_payload(float(m_total), cog_total_tool.tolist())
    return float(m_obj), cog_obj_tool.tolist(), float(m_total), cog_total_tool.tolist()


def estimate_object_payload_remote(
    ur: URArmControlClient,
    samples: Optional[int] = None,
    assume_rz: Optional[float] = None,
) -> Tuple[float, List[float], float, List[float]]:
    samples = int(samples if samples is not None else get(ur.cfg, "calibration.estimate.samples", 30))
    assume_rz = float(assume_rz if assume_rz is not None else get(ur.cfg, "calibration.estimate.assume_rz_m", 0.0))

    # Bias
    ur.measure_wrench_bias()

    forces = []
    taus_b = []
    R_list = []

    for _ in range(max(8, samples)):
        # Average wrench
        t_end = time.time() + float(get(ur.cfg, "calibration.estimate.avg_s", 0.15))
        acc = np.zeros(6)
        n = 0
        while time.time() < t_end:
            w = np.array(ur._get("/get_wrench")["wrench"], float) - ur.ft_bias
            acc += w
            n += 1
            time.sleep(0.02)
        wrench = acc / max(1, n)

        pose = ur.get_tcp_pose_axis_angle()
        R = rotvec_to_R(pose[3], pose[4], pose[5])  # tool->base

        forces.append(wrench[:3])
        taus_b.append(wrench[3:])
        R_list.append(R)

    forces = np.array(forces)
    taus_b = np.array(taus_b)

    Rz = np.array([R[:, 2] for R in R_list])
    proj = np.einsum('ij,ij->i', forces, Rz)
    m_obj = -float(np.mean(proj)) / G

    tau_tool = np.array([R.T @ tb for R, tb in zip(R_list, taus_b)])
    mg = max(m_obj * G, 1e-6)
    r_x = float(np.mean(tau_tool[:, 1]) / mg) if mg > 1e-6 else 0.0
    r_y = float(-np.mean(tau_tool[:, 0]) / mg) if mg > 1e-6 else 0.0
    r_z = float(assume_rz)
    cog_obj_tool = np.array([r_x, r_y, r_z], float)

    m_total = ur.tool_mass + m_obj
    cog_total_tool = (ur.tool_mass*ur.tool_cog + m_obj*cog_obj_tool) / max(m_total, 1e-6)

    ur.set_active_payload(float(m_total), cog_total_tool.tolist())
    return float(m_obj), cog_obj_tool.tolist(), float(m_total), cog_total_tool.tolist()


if __name__ == "__main__":
    main()
