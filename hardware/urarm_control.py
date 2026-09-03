#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UR10e RTDE controller with impedance servo, YAML-configurable parameters, and
payload calibration (object mass/CoG estimation).

Dependencies:
  - ur_rtde (pip install ur_rtde)
  - numpy
  - pyyaml

Safety:
  - Test with low speeds/gains and generous clearance.
  - Real robots can cause injury and damage.

Notes in this version:
  - Fixes a race where unconditional stop requests prevented motion.
  - We only cancel streaming loops when actually active.
  - "Heavy" stopScript() is called only if we know our velocity loop was running.
"""

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import time
import math
import argparse
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import yaml
from threading import Event, Lock

from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive
from commons.grasp_utils import pose_from_T, make_T, load_yaml_pose, check_path_exists


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


# -----------------------------
# Main Controller
# -----------------------------
class URArmControl:
    def __init__(self):
        """
        cfg: full dictionary from YAML (see example at bottom).
        Required minimal keys:
          - robot.host
        """
        cfg = {}
        # Try several ways to open the config for robustness.
        cfg_file = check_path_exists(CFG_PATH, __file__)
        if cfg_file:
            with cfg_file.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        else:
            print('ERROR no cfg file', cfg_file)
            sys.exit(1)

        self.cfg = cfg
        self.robot = get(cfg, "robot")
        self.servo_running = False;

        self.host = get(self.robot, "host")
        if not self.host:
            raise ValueError("robot.host (UR controller IP) must be set in YAML.")

        self.locations = get(cfg, "locations")

        self.rtde_c = RTDEControl(self.host)
        self.rtde_r = RTDEReceive(self.host)

        # Watchdog/loop rate
        self.dt = 1.0 / float(get(self.robot, "watchdog_hz", 100.0))

        # Use the payload stored in the active UR installation as the baseline.
        self.active_mass = float(self.rtde_r.getPayload())
        self.active_cog = np.array(self.rtde_r.getPayloadCog(), dtype=float)
        self.tool_mass = self.active_mass
        self.tool_cog = self.active_cog.copy()
        print(
            "[INFO] Using payload from active UR installation: "
            f"{self.tool_mass:.3f} kg, CoG={self.tool_cog.tolist()}"
        )

        # TCP
        self.tcp = get(cfg, "tool.tcp_pose", None)
        if self.tcp is not None:
            if len(self.tcp) != 6:
                raise ValueError("tool.tcp_pose must be [x,y,z,rx,ry,rz].")
            self.rtde_c.setTcp(list(map(float, self.tcp)))

        # Wrench bias (estimated in free space)
        self.ft_bias = np.zeros(6, dtype=float)

        # --- Cooperative cancellation controls ---
        self._stop_event = Event()      # set() to request cancellation of any running loop
        self._loop_lock = Lock()        # guards loop start/stop transitions
        self._loop_name: Optional[str] = None  # "moveI", "holdI", or None
        self._script_active: bool = False      # True while our speedL loop is running

        # Optional debug
        self._debug = bool(get(self.cfg, "debug", False))

    # ----------- Setup / Calibration -----------
    def set_tcp(self, tcp_pose_axis_angle: List[float]) -> None:
        self.rtde_c.setTcp(tcp_pose_axis_angle)

    def set_tool_payload(self, mass_kg: float, cog_tool: List[float]):
        """Set gripper-only payload; also becomes the active payload."""
        self.tool_mass = float(mass_kg)
        self.tool_cog = np.array(cog_tool, dtype=float)
        self.rtde_c.setPayload(self.tool_mass, self.tool_cog.tolist())
        self.active_mass = self.tool_mass
        self.active_cog = self.tool_cog.copy()

    def update_payload_from_rtde(self):
        """Read active payload mass/CoG currently on the controller and cache them."""
        m = float(self.rtde_r.getPayload())           # kg
        c = np.array(self.rtde_r.getPayloadCog(), float)  # [x,y,z] m (tool frame)
        self.active_mass = m
        self.active_cog = c
        return m, c.tolist()

    def set_active_payload(self, total_mass_kg: float, total_cog_tool: List[float]):
        """Set combined payload (tool + object) on the robot and cache."""
        self.rtde_c.setPayload(float(total_mass_kg), list(total_cog_tool))
        self.active_mass = float(total_mass_kg)
        self.active_cog = np.array(total_cog_tool, float)

    def measure_wrench_bias(self, seconds: Optional[float] = None, dt: Optional[float] = None) -> List[float]:
        """Average the current TCP wrench (object in free space) to form a bias."""
        seconds = seconds if seconds is not None else float(get(self.cfg, "calibration.bias_seconds", 0.7))
        dt = dt if dt is not None else float(get(self.cfg, "calibration.bias_dt", 0.005))
        t_end = time.time() + max(0.1, seconds)
        ss = []
        while time.time() < t_end:
            ss.append(np.array(self.rtde_r.getActualTCPForce(), float))
            time.sleep(dt)
        self.ft_bias = np.mean(ss, axis=0) if ss else np.zeros(6)
        return self.ft_bias.tolist()
    
    def calibrate_payload_sequence(
        self,
        speed: Optional[float] = None,
        accel: Optional[float] = None,
        settle: float = 0.2,
        avg_sec: float = 0.15,
        assume_rz: float = 0.0,
    ) -> Tuple[float, List[float], float, List[float]]:
        """
        Take 4 samples at specific orientations:
          - current pose
          - wrist3 (joint 5) at +20°, +40°, +60°
          - wrist1 (joint 3) at +45°
        Then estimate object mass and CoG(x,y) in tool frame and apply the new total payload.
        Returns: (m_obj, cog_obj_tool[3], m_total, cog_total_tool[3])
        """
        # Gentle motion defaults
        speed = float(speed if speed is not None else get(self.cfg, "robot.j_speed", 0.6))
        accel = float(accel if accel is not None else get(self.cfg, "robot.j_accel", 1.2))

        # 0) Bias FT with object held in free space
        self.measure_wrench_bias(
            seconds=float(get(self.cfg, "calibration.bias_seconds", 0.7)),
            dt=float(get(self.cfg, "calibration.bias_dt", 0.005)),
        )

        # 1) Build the joint targets
        q0 = list(self.rtde_r.getActualQ())
        targets: List[List[float]] = []
        # current
        targets.append(q0.copy())
        # wrist3 +20, +40, +60  (joint index 5)
        for deg in (20.0, 40.0, 60.0):
            q = q0.copy()
            q[5] = q[5] + math.radians(deg)
            targets.append(q)
        # wrist1 +45 (joint index 3)
        qw1 = q0.copy()
        qw1[3] = qw1[3] + math.radians(45.0)
        targets.append(qw1)

        forces = []
        taus_b = []
        R_list = []

        # 2) Visit each target, settle, then average FT
        for q in targets:
            self.moveJ(q, speed=speed, accel=accel, async_=False)
            # small settle
            time.sleep(max(0.05, settle))
            # average over avg_sec
            t_end = time.time() + max(0.05, avg_sec)
            acc = np.zeros(6)
            n = 0
            while time.time() < t_end:
                w = np.array(self.rtde_r.getActualTCPForce(), float) - self.ft_bias
                acc += w
                n += 1
                time.sleep(0.005)
            wrench = acc / max(1, n)

            pose = self.rtde_r.getActualTCPPose()
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

        # 4) Torques -> r_x, r_y (in tool frame); r_z assumed/unobservable statically
        tau_tool = np.array([R.T @ tb for R, tb in zip(R_list, taus_b)])
        mg = max(m_obj * G, 1e-6)
        r_x = float(np.mean(tau_tool[:, 1]) / mg)    # tau_y ≈ +m g r_x
        r_y = float(-np.mean(tau_tool[:, 0]) / mg)   # tau_x ≈ -m g r_y
        r_z = float(assume_rz)
        cog_obj_tool = np.array([r_x, r_y, r_z], float)

        # 5) Combine with known tool baseline for total payload
        m_total = self.tool_mass + m_obj
        cog_total_tool = (self.tool_mass*self.tool_cog + m_obj*cog_obj_tool) / max(m_total, 1e-6)
        print('new values',m_total,cog_total_tool)
        # Apply to robot and cache
        self.set_active_payload(float(m_total), cog_total_tool.tolist())

        return float(m_obj), cog_obj_tool.tolist(), float(m_total), cog_total_tool.tolist()

    def estimate_object_payload(
        self,
        samples: Optional[int] = None,
        settle: Optional[float] = None,
        avg_sec: Optional[float] = None,
        assume_rz: Optional[float] = None,
    ) -> Tuple[float, List[float], float, List[float]]:
        """
        Identify object mass and CoG(x,y) in tool frame via gravity across orientations.
        Assumes active payload is tool-only before calling (or bias removes the rest).
        Returns: (m_obj, cog_obj_tool[3], m_total, cog_total_tool[3])
        """
        samples = int(samples if samples is not None else get(self.cfg, "calibration.estimate.samples", 30))
        settle = float(settle if settle is not None else get(self.cfg, "calibration.estimate.settle_s", 0.2))
        avg_sec = float(avg_sec if avg_sec is not None else get(self.cfg, "calibration.estimate.avg_s", 0.15))
        assume_rz = float(assume_rz if assume_rz is not None else get(self.cfg, "calibration.estimate.assume_rz_m", 0.0))

        # 1) Bias with the object held free in space
        self.measure_wrench_bias()

        forces = []
        taus_b = []
        R_list = []

        for _ in range(max(8, samples)):
            time.sleep(settle)
            # Average wrench at this pose
            t_end = time.time() + avg_sec
            acc = np.zeros(6)
            n = 0
            while time.time() < t_end:
                w = np.array(self.rtde_r.getActualTCPForce(), float) - self.ft_bias
                acc += w
                n += 1
                time.sleep(0.005)
            wrench = acc / max(1, n)

            pose = self.rtde_r.getActualTCPPose()
            R = rotvec_to_R(pose[3], pose[4], pose[5])  # tool->base

            forces.append(wrench[:3])
            taus_b.append(wrench[3:])
            R_list.append(R)

        forces = np.array(forces)
        taus_b = np.array(taus_b)

        # 2) Mass from force projection onto tool -Z in base
        Rz = np.array([R[:, 2] for R in R_list])          # tool z in base
        proj = np.einsum('ij,ij->i', forces, Rz)          # Fi · Rz_i
        m_obj = -float(np.mean(proj)) / G

        # 3) Torques -> r_x, r_y (in tool frame)
        tau_tool = np.array([R.T @ tb for R, tb in zip(R_list, taus_b)])
        mg = max(m_obj * G, 1e-6)
        r_x = float(np.mean(tau_tool[:, 1]) / mg)   # tau_y ≈ +m g r_x
        r_y = float(-np.mean(tau_tool[:, 0]) / mg)  # tau_x ≈ -m g r_y
        r_z = float(assume_rz)                      # unobservable statically
        cog_obj_tool = np.array([r_x, r_y, r_z], float)

        # 4) Combine with known tool baseline for total payload
        m_total = self.tool_mass + m_obj
        cog_total_tool = (self.tool_mass*self.tool_cog + m_obj*cog_obj_tool) / max(m_total, 1e-6)

        # Apply to robot
        self.set_active_payload(float(m_total), cog_total_tool.tolist())
        return float(m_obj), cog_obj_tool.tolist(), float(m_total), cog_total_tool.tolist()

    # ----------- Wrench compensation used by impedance -----------
    def _compensated_wrench(self) -> np.ndarray:
        """
        Return TCP wrench with (a) bias removed and (optionally) (b) gravity removed.
        Optional final sign flip if controller reports the opposite convention.
        """
        w = np.array(self.rtde_r.getActualTCPForce(), float) - self.ft_bias

        # Optional gravity removal in software (default off).
        if bool(get(self.cfg, "motion.software_gravity_comp", False)):
            pose = self.rtde_r.getActualTCPPose()
            R_tb = rotvec_to_R(pose[3], pose[4], pose[5])
            F_tool = np.array([0.0, 0.0, -self.active_mass * G], float)
            tau_tool = np.cross(self.active_cog, F_tool)
            grav = np.concatenate([R_tb @ F_tool, R_tb @ tau_tool])
            w = w - grav

        # Optional sign flip if your stack reports the opposite (environment vs robot) sign.
        if bool(get(self.cfg, "motion.flip_ft_sign", False)):
            w = -w

        return w

    # ----------- Cooperative cancellation helpers -----------
    def _begin_loop(self, name: str):
        """Mark the start of a long-running loop (moveI/holdI)."""
        with self._loop_lock:
            self._stop_event.clear()
            self._loop_name = name
            self._script_active = True
            if self._debug:
                print(f"[DEBUG] begin_loop: {name}")

    def _end_loop(self):
        """Mark the end of a long-running loop."""
        with self._loop_lock:
            if self._debug:
                print(f"[DEBUG] end_loop: {self._loop_name}")
            self._loop_name = None
            self._stop_event.clear()
            self._script_active = False

    def is_loop_active(self) -> bool:
        return (self._loop_name is not None) and (not self._stop_event.is_set())

    def stop_gracefully(self):
        self.rtde_c.stopScript()

    def request_stop(self, force: bool = False):
        self._stop_event.set()
        if self._debug:
            print(f"[DEBUG] request_stop: loop_name={self._loop_name} script_active={self._script_active}")
        try:
            # Gentle stop of any current speed move
            if force or self._loop_name is not None or self._script_active:
                self.rtde_c.speedStop()
        except Exception as e:
            print(e)
        

        # Only bring out the hammer if explicitly forced
        # if force:
        #     try:
        #         if hasattr(self.rtde_c, "isProgramRunning"):
        #             try:
        #                 if self.rtde_c.isProgramRunning():
        #                     self.rtde_c.stopScript()
        #             except Exception:
        #                 self.rtde_c.stopScript()
        #         else:
        #             self.rtde_c.stopScript()
        #     except Exception:
        #         pass

        self._script_active = False

    # ----------- Motion primitives -----------
    def _cancel_if_streaming(self,ignore_reset=False):
        if not ignore_reset and self.servo_running:
            print('SERVO was running, must do a reset before running new commands')
            self.servo_running = False
            self.reset()
            
        # Only cancel if our streaming loop is/was active
        if self.is_loop_active() or self._script_active:
            self.request_stop()

    def moveJ(self, q: List[float], speed: Optional[float] = None, accel: Optional[float] = None, async_: bool = False):
        """Joint move to q (rad). Cancels impedance/hold loop first IF active."""
        print('moveJ',q)
        self._cancel_if_streaming()
        speed = float(speed if speed is not None else get(self.cfg, "robot.j_speed", 1.2))
        accel = float(accel if accel is not None else get(self.cfg, "robot.j_accel", 2.5))
        return self.rtde_c.moveJ(q, speed, accel, async_)

    def moveL(self, pose: List[float], speed: Optional[float] = None, accel: Optional[float] = None, async_: bool = False):
        print('moveL',pose)
        """Linear move to pose [x,y,z,rx,ry,rz] in base frame. Cancels loop first IF active."""
        self._cancel_if_streaming()
        speed = float(speed if speed is not None else get(self.cfg, "robot.l_speed", 0.25))
        accel = float(accel if accel is not None else get(self.cfg, "robot.l_accel", 1.2))
        return self.rtde_c.moveL(pose, speed, accel, async_)

    def moveLPath(self, path: List[Dict], speed: Optional[float] = None, accel: Optional[float] = None, async_: bool = False):
        print('moveLPath',len(path))
        """Linear move to pose [x,y,z,rx,ry,rz] in base frame. Cancels loop first IF active."""
        self._cancel_if_streaming()
        speed = float(speed if speed is not None else get(self.cfg, "robot.l_speed", 0.25))
        accel = float(accel if accel is not None else get(self.cfg, "robot.l_accel", 1.2))
        return self.rtde_c.moveL(path,async_)


    def servoL(self, pose: List[float], speed: float = 0.25, accel: float = 0.5, time: float = 0.0, lookahead_time: float = 0.1, gain: float = 300.0) -> None:
        """Cancels loop IF active and sends a servoL."""
        self._cancel_if_streaming(ignore_reset=True)
        if self.rtde_c:
            self.rtde_c.servoL(pose, speed, accel, time, lookahead_time, gain)
            self.servo_running = True

    # def servoL(
    #     self,
    #     pose: List[float],
    #     speed: float = 0.25,
    #     accel: float = 0.5,
    #     time: float = 0.0,
    #     lookahead_time: float = 0.1,
    #     gain: float = 300.0,
    # ) -> None:
    #     """Simple safety: refuse large single-step jumps; otherwise forward to RTDE."""
    #     self._cancel_if_streaming()

    #     # Defaults if YAML not present
    #     g = get(self.cfg, "motion.servoL_guard", {}) or {}
    #     if g.get("enabled", True) is False:
    #         return self.rtde_c.servoL(pose, speed, accel, time, lookahead_time, gain)

    #     # Limits
    #     max_lin = float(g.get("max_step_lin_m", 0.02))      # 2 cm
    #     max_rot = float(g.get("max_step_rot_rad", 0.10))    # ~5.7°

    #     # Basic sanity
    #     if not all(np.isfinite(pose)):
    #         raise ValueError("servoL refused: non-finite target pose.")

    #     cur = list(self.rtde_r.getActualTCPPose())
    #     dx = np.array(pose[:3], float) - np.array(cur[:3], float)
    #     dr = np.array(pose[3:6], float) - np.array(cur[3:6], float)
    #     lin = float(np.linalg.norm(dx))
    #     rot = float(np.linalg.norm(dr))

    #     if lin > max_lin or rot > max_rot:
    #         raise ValueError(
    #             f"servoL refused: jump too large (Δlin={lin:.4f} m, Δrot={rot:.3f} rad; "
    #             f"limits {max_lin:.4f} m, {max_rot:.3f} rad)."
    #         )
        
    #     # Safe to execute
    #     self.rtde_c.servoL(pose, speed, accel, time, lookahead_time, gain)
        
    def servoJ(self, joints: List[float], speed: float = 0.25, accel: float = 0.5, time: float = 0.0, lookahead_time: float = 0.1, gain: float = 300.0) -> None:
        """Cancels loop IF active and sends a servoJ."""
        self._cancel_if_streaming(ignore_reset=True)
        if self.rtde_c:
            self.rtde_c.servoJ(joints, speed, accel, time, lookahead_time, gain)

    def waitForMotion(self):
       while not self.rtde_c.isSteady():                # or: np.linalg.norm(rtde_r.getActualQd()) < 1e-3
            print('waiting for arm to stop')
            time.sleep(0.02)

    def running(self):
        return not self.rtde_c.isSteady()

    def moveI(
        self,
        target_pose: List[float],
        loop_hz: Optional[float] = None,
        kp_lin: Optional[float] = None,
        kp_rot: Optional[float] = None,
        max_lin_speed: Optional[float] = None,
        max_rot_speed: Optional[float] = None,
        force_threshold: Optional[float] = None,
        k_imp: Optional[float] = None,
        imp_max: Optional[float] = None,
        ft_alpha: Optional[float] = None,
        settle_tol_lin: Optional[float] = None,
        settle_tol_rot: Optional[float] = None,
        settle_time: Optional[float] = None,
        max_time: Optional[float] = None,
        remeasure_bias: Optional[bool] = None,
        downweight_z: Optional[float] = None,
        auto_hold: bool = False,
    ):
        """
        Impedance (velocity) servo to target pose with contact yielding at a fixed rate.
        All parameters default from YAML unless explicitly overridden.
        Cancellation: This loop will exit if `request_stop()` is called or if you
        start any other move command (moveJ/moveL/servoJ/servoL), which internally calls `request_stop()`.
        """
        # YAML defaults
        mcfg = self.cfg.get("motion", {})
        icfg = mcfg.get("moveI", {})

        loop_hz = float(loop_hz if loop_hz is not None else icfg.get("loop_hz", 100.0))
        kp_lin = float(kp_lin if kp_lin is not None else icfg.get("kp_lin", 1.2))
        kp_rot = float(kp_rot if kp_rot is not None else icfg.get("kp_rot", 2.0))
        max_lin_speed = float(max_lin_speed if max_lin_speed is not None else icfg.get("max_lin_speed", 0.12))
        max_rot_speed = float(max_rot_speed if max_rot_speed is not None else icfg.get("max_rot_speed", 0.7))
        force_threshold = float(force_threshold if force_threshold is not None else icfg.get("force_threshold", 10.0))
        k_imp = float(k_imp if k_imp is not None else icfg.get("k_imp", 0.0035))
        imp_max = float(imp_max if imp_max is not None else icfg.get("imp_max", 0.05))
        ft_alpha = float(ft_alpha if ft_alpha is not None else icfg.get("ft_alpha", 0.25))
        settle_tol_lin = float(settle_tol_lin if settle_tol_lin is not None else icfg.get("settle_tol_lin", 0.002))
        settle_tol_rot = float(settle_tol_rot if settle_tol_rot is not None else icfg.get("settle_tol_rot", 0.01))
        settle_time = float(settle_time if settle_time is not None else icfg.get("settle_time", 0.5))
        max_time = float(max_time if max_time is not None else icfg.get("max_time", 30.0))
        remeasure_bias = bool(remeasure_bias if remeasure_bias is not None else icfg.get("remeasure_bias", True))
        downweight_z = float(downweight_z if downweight_z is not None else icfg.get("downweight_z", 1.0))

        dt = 1.0 / loop_hz
        self.dt = dt
        self._cancel_if_streaming()
        # Mark loop started
        self._begin_loop("moveI")

        if remeasure_bias:
            self.measure_wrench_bias(seconds=float(get(self.cfg, "calibration.bias_seconds", 0.6)))

        ft_filt = np.zeros(6, float)
        settled_since = None
        t0 = time.time()

        try:
            while not self._stop_event.is_set():
                t_loop = time.time()

                # Task velocity from pose error
                cur = self.rtde_r.getActualTCPPose()
                err = pose_err(target_pose, cur)
                v_lin = [kp_lin*e for e in err[:3]]
                v_rot = [kp_rot*e for e in err[3:]]
                v_lin = clamp_vec(v_lin, max_lin_speed)
                v_rot = clamp_vec(v_rot, max_rot_speed)

                # Impedance relief from compensated force
                ft = self._compensated_wrench()
                ft_filt = lpf(ft_filt, ft, ft_alpha)
                F = ft_filt[:3].copy()
                F[2] *= float(downweight_z)

                fmag = vec_norm(F)
                if fmag > force_threshold:
                    direction = np.array(F, float) / (fmag + 1e-9)
                    v_relief_lin = (k_imp * (fmag - force_threshold)) * direction
                    v_relief_lin = clamp_vec(v_relief_lin.tolist(), imp_max)
                else:
                    v_relief_lin = [0.0, 0.0, 0.0]

                v_cmd = [v_lin[0] + v_relief_lin[0],
                         v_lin[1] + v_relief_lin[1],
                         v_lin[2] + v_relief_lin[2],
                         v_rot[0], v_rot[1], v_rot[2]]

                # Final caps
                v_cmd[:3] = clamp_vec(v_cmd[:3], max_lin_speed)
                v_cmd[3:] = clamp_vec(v_cmd[3:], max_rot_speed)

                # Send
                self.rtde_c.speedL(v_cmd, 0.5, dt)

                # Settle detection
                lin_err = vec_norm(err[:3])
                rot_err = vec_norm(err[3:])
                if lin_err < settle_tol_lin and rot_err < settle_tol_rot:
                    if settled_since is None:
                        settled_since = t_loop
                    elif (t_loop - settled_since) >= settle_time:
                        self.rtde_c.speedStop()
                        if auto_hold and not self._stop_event.is_set():
                            # Transition into hold control; if cancelled, holdI will honor the stop event.
                            self.holdI(loop_hz=loop_hz,
                                       kp_lin=kp_lin, kp_rot=kp_rot,
                                       max_lin_speed=max_lin_speed, max_rot_speed=max_rot_speed,
                                       force_threshold=force_threshold, k_imp=k_imp,
                                       imp_max=imp_max, ft_alpha=ft_alpha,
                                       downweight_z=downweight_z, remeasure_bias=False,
                                       max_time=None)
                        return True
                else:
                    settled_since = None

                # Timeout
                if (max_time is not None) and ((t_loop - t0) > max_time):
                    self.rtde_c.speedStop()
                    return False

                # Keep loop timing
                elapsed = time.time() - t_loop
                if (dt - elapsed) > 0:
                    time.sleep(dt - elapsed)
            # stop_event was set: cooperative cancel
            self.rtde_c.speedStop()
            return False
        except KeyboardInterrupt:
            self.rtde_c.speedStop()
            return False
        finally:
            try:
                self._end_loop()
                # if self._script_active:
                #     self.rtde_c.stopScript()
            except Exception:
                pass
            self._end_loop()

    def holdI(
        self,
        loop_hz: Optional[float] = None,
        kp_lin: Optional[float] = None,
        kp_rot: Optional[float] = None,
        max_lin_speed: Optional[float] = None,
        max_rot_speed: Optional[float] = None,
        force_threshold: Optional[float] = None,
        k_imp: Optional[float] = None,
        imp_max: Optional[float] = None,
        ft_alpha: Optional[float] = None,
        downweight_z: Optional[float] = None,
        remeasure_bias: Optional[bool] = None,
        max_time: Optional[float] = None,
    ):
        """
        Hold current TCP pose using impedance control at loop_hz (virtual spring + yielding).
        All parameters default from YAML unless explicitly overridden.
        Cancellation: Exits if `request_stop()` is called or another move is issued.
        """
        mcfg = self.cfg.get("motion", {})
        hcfg = mcfg.get("holdI", {})

        loop_hz = float(loop_hz if loop_hz is not None else hcfg.get("loop_hz", 100.0))
        kp_lin = float(kp_lin if kp_lin is not None else hcfg.get("kp_lin", 2.0))
        kp_rot = float(kp_rot if kp_rot is not None else hcfg.get("kp_rot", 3.0))
        max_lin_speed = float(max_lin_speed if max_lin_speed is not None else hcfg.get("max_lin_speed", 0.10))
        max_rot_speed = float(max_rot_speed if max_rot_speed is not None else hcfg.get("max_rot_speed", 0.6))
        force_threshold = float(force_threshold if force_threshold is not None else hcfg.get("force_threshold", 12.0))
        k_imp = float(k_imp if k_imp is not None else hcfg.get("k_imp", 0.0030))
        imp_max = float(imp_max if imp_max is not None else hcfg.get("imp_max", 0.04))
        ft_alpha = float(ft_alpha if ft_alpha is not None else hcfg.get("ft_alpha", 0.25))
        downweight_z = float(downweight_z if downweight_z is not None else hcfg.get("downweight_z", 1.0))
        remeasure_bias = bool(remeasure_bias if remeasure_bias is not None else hcfg.get("remeasure_bias", False))
        max_time = None if max_time is None else float(max_time)
        self._cancel_if_streaming()
        
        dt = 1.0 / loop_hz
        self.dt = dt
        # Reference pose = pose when entering hold
        ref = list(self.rtde_r.getActualTCPPose())

        # Mark loop started
        self._begin_loop("holdI")

        if remeasure_bias:
            self.measure_wrench_bias(seconds=float(get(self.cfg, "calibration.bias_seconds", 0.6)))

        ft_filt = np.zeros(6, float)
        t0 = time.time()

        try:
            while not self._stop_event.is_set():
                t_loop = time.time()

                # Pose error to fixed reference (virtual spring)
                cur = self.rtde_r.getActualTCPPose()
                err = [ref[i] - cur[i] for i in range(6)]

                v_lin = [kp_lin*e for e in err[:3]]
                v_rot = [kp_rot*e for e in err[3:]]
                v_lin = clamp_vec(v_lin, max_lin_speed)
                v_rot = clamp_vec(v_rot, max_rot_speed)

                # Impedance relief from compensated wrench (yields on contact)
                ft = self._compensated_wrench()
                ft_filt = lpf(ft_filt, ft, ft_alpha)
                F = ft_filt[:3].copy()
                F[2] *= float(downweight_z)

                fmag = vec_norm(F)
                if fmag > force_threshold:
                    direction = np.array(F, float) / (fmag + 1e-9)
                    v_relief_lin = (-k_imp * (fmag - force_threshold)) * direction
                    v_relief_lin = clamp_vec(v_relief_lin.tolist(), imp_max)
                else:
                    v_relief_lin = [0.0, 0.0, 0.0]

                v_cmd = [v_lin[0] + v_relief_lin[0],
                         v_lin[1] + v_relief_lin[1],
                         v_lin[2] + v_relief_lin[2],
                         v_rot[0], v_rot[1], v_rot[2]]

                # Final caps
                v_cmd[:3] = clamp_vec(v_cmd[:3], max_lin_speed)
                v_cmd[3:] = clamp_vec(v_cmd[3:], max_rot_speed)

                # Send at loop rate
                self.rtde_c.speedL(v_cmd, 0.5, dt)

                # Optional timeout
                if max_time is not None and (t_loop - t0) > max_time:
                    self.rtde_c.speedStop()
                    return True

                # Loop timing
                elapsed = time.time() - t_loop
                if (dt - elapsed) > 0:
                    time.sleep(dt - elapsed)

            # stop_event was set: cooperative cancel
            self.rtde_c.speedStop()
            return True
        except KeyboardInterrupt:
            self.rtde_c.speedStop()
            return True
        
        finally:
            try:
                self._end_loop()
                # if self._script_active:
                #     self.rtde_c.stopScript()
            except Exception:
                pass
            self._end_loop()

    def get_wrench_raw(self) -> List[float]:
        w = np.array(self.rtde_r.getActualTCPForce(), float)
        return w

    def get_wrench(self) -> List[float]:
        w = np.array(self.rtde_r.getActualTCPForce(), float) - self.ft_bias
        return w

    def get_tcp_pose_axis_angle(self) -> List[float]:
        return self.rtde_r.getActualTCPPose()

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


    # Backwards-compatible name; stop() now performs a cooperative cancel
    def stop(self) -> None:
        """Explicitly stop any active loop and halt the current script (if ours)."""
        self.request_stop()

    def reset(self) -> bool:
            """
            Reset RTDE control/receive interfaces and reapply current state.
            - Cooperatively cancels any streaming loops.
            - Disconnects existing RTDE interfaces (if supported by the version).
            - Recreates RTDEControl/RTDEReceive with the same host.
            - Reapplies TCP and the *current active payload* (mass/CoG).
            Returns True on success, False otherwise.
            """
            # Stop any velocity loops and scripts we own
            try:
                self.request_stop()  # gentle; respects our cooperative flags
            except Exception as e:
                print(f"[WARN] request_stop during reset: {e}")

            # Best-effort disconnect on existing interfaces
            for name, iface in (("rtde_c", getattr(self, "rtde_c", None)),
                                ("rtde_r", getattr(self, "rtde_r", None))):
                try:
                    if iface is not None and hasattr(iface, "disconnect"):
                        iface.disconnect()
                except Exception as e:
                    print(f"[WARN] {name}.disconnect() failed: {e}")

            # Recreate interfaces
            try:
                self.rtde_c = RTDEControl(self.host)
                self.rtde_r = RTDEReceive(self.host)
            except Exception as e:
                print(f"[ERROR] Recreating RTDE interfaces failed: {e}")
                return False

            # Reapply TCP if we had one
            try:
                if self.tcp is not None:
                    self.rtde_c.setTcp(list(map(float, self.tcp)))
            except Exception as e:
                print(f"[WARN] Reapplying TCP failed: {e}")

            # Reapply *active* payload (what we were using before reset)
            try:
                self.rtde_c.setPayload(float(self.active_mass), self.active_cog.tolist())
            except Exception as e:
                print(f"[WARN] Reapplying payload failed: {e}")

            # Keep previously computed bias; user can re-bias if desired
            # Reset loop flags
            self._stop_event.clear()
            self._loop_name = None
            self._script_active = False

            if self._debug:
                print("[DEBUG] RTDE interfaces reset and state reapplied.")
            return True
# -----------------------------
# CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="UR10e RTDE impedance controller (YAML-configurable).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("calibrate", help="Calibrate payload using 4 predefined orientations.")
    c.add_argument("--assume-rz", type=float, help="Assume CoG z (m); statically unobservable.", default=0.0)
    c.add_argument("--speed", type=float, help="Joint speed for sampling moves (rad/s).")
    c.add_argument("--accel", type=float, help="Joint accel for sampling moves (rad/s^2).")
    c.add_argument("--settle", type=float, help="Settle time at each pose (s).", default=0.2)
    c.add_argument("--avg-sec", type=float, help="Averaging duration at each pose (s).", default=0.15)
    
    j = sub.add_parser("location", help="move to location.")
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

    i = sub.add_parser("movei", help="Impedance servo to [x y z rx ry rz].")
    i.add_argument("pose", nargs=6, type=float)
    i.add_argument("--auto-hold", action="store_true")
    i.add_argument("--max-time", type=float)

    h = sub.add_parser("holdi", help="Impedance hold at current pose (Ctrl+C to stop).")
    h.add_argument("--max-time", type=float)

    s = sub.add_parser("set_tool_payload", help="Set gripper-only mass and CoG (tool frame).")
    s.add_argument("mass", type=float)
    s.add_argument("cog", nargs=3, type=float)

    u = sub.add_parser("update_payload_from_rtde", help="Read active payload mass & CoG from controller.")

    e = sub.add_parser("estimate_object_payload", help="Estimate object mass/CoG then update total payload.")
    e.add_argument("--samples", type=int)
    e.add_argument("--assume-rz", type=float)

    st = sub.add_parser("stop", help="Cooperatively stop any running impedance/hold loop and halt the script.")

    args = parser.parse_args()

    ur = URArmControl()
    print(ur.locations['ready'])
    if args.cmd == "location":
        ur.moveJ(ur.locations[str(args.name)])

    elif args.cmd == "calibrate":
        m_obj, cog_obj, m_tot, cog_tot = ur.calibrate_payload_sequence(
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
        ur.moveI(args.pose, auto_hold=args.auto_hold, max_time=args.max_time)
    elif args.cmd == "holdi":
        ur.holdI(max_time=args.max_time)
    elif args.cmd == "set_tool_payload":
        ur.set_tool_payload(args.mass, args.cog)
        print("Tool payload set.")
    elif args.cmd == "update_payload_from_rtde":
        m, c = ur.update_payload_from_rtde()
        print(f"Active payload: {m:.3f} kg, CoG(tool) = {c}")
    elif args.cmd == "estimate_object_payload":
        m_obj, cog_obj, m_tot, cog_tot = ur.estimate_object_payload(
            samples=args.samples, assume_rz=args.assume_rz
        )
        print(f"Estimated object mass: {m_obj:.3f} kg  CoG(tool): {cog_obj}")
        print(f"Updated total payload: {m_tot:.3f} kg  CoG(tool): {cog_tot}")
    elif args.cmd == "stop":
        ur.stop()
        print("Stop requested (loops cancelled, script halted if active).")


if __name__ == "__main__":
    main()
