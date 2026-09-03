# behaviours/idle_lookaround.py
from __future__ import annotations
import math, time, threading
import numpy as np
from .base_action import BaseAction, register_action
from commons.grasp_utils import clamp, as_np_bounds, as_tuple2

@register_action("look")
@register_action("idle_lookaround")
class IdleLookaroundAction(BaseAction):
    def run(self, stop_event: threading.Event, **kwargs):
        cfgl = dict(self.cfg.get("idle_lookaround", {}))
        controlcfg = dict(self.cfg.get("control", {}))
        yaw_amplitude_deg = float(cfgl.get("yaw_amplitude_deg", 30.0))
        pitch_amplitude_deg = float(cfgl.get("pitch_amplitude_deg", 15.0))
        period_s = float(cfgl.get("period_s", 6.0))
        pitch_freq_mult = float(cfgl.get("pitch_freq_mult", 1.0))
        servo_hz = int(controlcfg.get("servo_hz", 100))
        servo_lookahead = float(controlcfg.get("servo_lookahead", 0.03))
        servo_gain = float(controlcfg.get("servo_gain", 300.0))
        xy_radius_m = float(cfgl.get("xy_radius_m", 0.05))
        z_amplitude_m = float(cfgl.get("z_amplitude_m", 0.02))
        xy_bounds = controlcfg.get("xy_bounds")
        z_bounds = controlcfg.get("z_bounds")
        lp_alpha_pos = float(controlcfg.get("lp_alpha_pos", 0.25))
        lp_alpha_yaw = float(cfgl.get("lp_alpha_yaw", 0.25))
        lp_alpha_pitch = float(cfgl.get("lp_alpha_pitch", 0.25))
        look_from_ready_pose = bool(cfgl.get("look_from_ready_pose", True))
        ready_location = cfgl.get("ready_location")
    
        # Auto-switch options
        auto_track = bool(cfgl.get("auto_track_on_person", True))
        visibility_threshold = float(cfgl.get("visibility_threshold", 0.5))
        presence_min_ticks = int(cfgl.get("presence_min_ticks", 8))
        presence_check_every = int(cfgl.get("presence_check_every", 2))

        if look_from_ready_pose:       
            self.arm.moveJ(ready_location)
        
        self.gripper.open()


        # capture neutral pose
        T0 = self.arm.get_T_base_tcp()
        p0 = T0[:3, 3].copy()
        R0 = T0[:3, :3].copy()

        dt = 1.0 / max(1, servo_hz)
        t0 = time.perf_counter()
        tick = 0
        p_cmd = p0.copy()
        yaw_cmd = 0.0
        pitch_cmd = 0.0

        def rotz(a):
            c, s = math.cos(a), math.sin(a)
            return np.array([[c, -s, 0.0],[s, c, 0.0],[0.0,0.0,1.0]], dtype=float)
        def rotx(a):
            c, s = math.cos(a), math.sin(a)
            return np.array([[1.0,0.0,0.0],[0.0,c,-s],[0.0,s,c]], dtype=float)

        use_servo = hasattr(self.arm, "servoL")
        presence_ticks = 0
        
        if self.debug:
            print("starting look around")

        # Ensure tracker is running if we want auto-track
        if auto_track:
            try:
                self.manager.start_tracker()
            except Exception:
                pass

        while not stop_event.is_set():
            target_time = t0 + tick * dt
            now = time.perf_counter()
            if (target_time - now) > 0:
                time.sleep(target_time - now)
            tick += 1
            t = time.perf_counter() - t0

            # oscillation targets
            phase = 2.0 * math.pi * (t / max(1e-6, period_s))
            yaw_target = math.radians(yaw_amplitude_deg) * math.sin(phase)
            phase_p = phase * max(1e-6, pitch_freq_mult) + math.pi * 0.5
            pitch_target = math.radians(pitch_amplitude_deg) * math.sin(phase_p)
            dx = xy_radius_m * math.sin(phase)
            dy = (xy_radius_m * 0.6) * math.sin(2.0 * phase)
            dz = z_amplitude_m * math.sin(1.5 * phase)
            p_target = p0 + np.array([dx, dy, dz], dtype=float)

            # smoothing
            yaw_cmd += lp_alpha_yaw * (yaw_target - yaw_cmd)
            pitch_cmd += lp_alpha_pitch * (pitch_target - pitch_cmd)
            p_cmd += lp_alpha_pos * (p_target - p_cmd)

            # desired pose
            R_des = rotz(yaw_cmd) @ R0 @ rotx(pitch_cmd)
            T_des = np.eye(4, dtype=float); T_des[:3,:3] = R_des; T_des[:3,3] = p_cmd

            try:
                from commons.grasp_utils import pose_from_T
                pose6 = pose_from_T(T_des)


                # clamps
                if xy_bounds is not None:
                    pose6[0] = clamp(pose6[0], float(xy_bounds[0][0]), float(xy_bounds[0][1]))
                    pose6[1] = clamp(pose6[1], float(xy_bounds[1][0]), float(xy_bounds[1][1]))
                if z_bounds is not None:
                    pose6[2] = clamp(pose6[2], float(z_bounds[0]), float(z_bounds[1]))

                # if self.debug:
                #     print("executing",pose6,'use_servo',use_servo)

                if use_servo:
                    self.arm.servoL(pose=pose6, time_s=dt, lookahead_time=servo_lookahead, gain=servo_gain)
                else:
                    self.arm.moveL(pose=pose6, 
                                   speed=float(self.cfg.get("motion",{}).get("move_speed",0.20)),
                                   accel=float(self.cfg.get("motion",{}).get("move_accel",0.60)))
            except Exception as e:
                print(f"[idle_lookaround] servo/move error: {e}")

            # --- auto switch if people seen ---
            if auto_track and (tick % max(1, presence_check_every) == 0):
                try:
                    body = self.tracker.get_body_positions(
                        transform_4x4=self.manager.T_base_fixed_camera,
                        filter_visible=True,
                        visibility_threshold=visibility_threshold,
                    )
                    if body:
                        presence_ticks += 1
                    else:
                        presence_ticks = 0
                except Exception:
                    presence_ticks = 0

                if presence_ticks >= presence_min_ticks:
                    # Trigger switch on a separate thread to avoid join() issues
                    def _switch():
                        try:
                            self.manager.start_action("point_at_person")
                        except Exception as e:
                            print("[idle_lookaround] switch error:", e)
                    threading.Thread(target=_switch, daemon=True).start()
                    stop_event.set()
                    break
