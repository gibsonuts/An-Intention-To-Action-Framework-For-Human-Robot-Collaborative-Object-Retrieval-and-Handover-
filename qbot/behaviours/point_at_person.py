# behaviours/point_at_person.py
from __future__ import annotations
import math, time, threading
import numpy as np
from .base_action import BaseAction, register_action
from commons.grasp_utils import  norm, clamp, slerp_R, log_so3, exp_so3, look_at_relative, has_any


@register_action("point")
@register_action("point_at_person")
class PointAtPersonAction(BaseAction):
    def run(self, stop_event: threading.Event, **kwargs):
        cfg = self.cfg.get("point_at_person", {})
        controlcfg = self.cfg.get("control", {})
        
        distance_m = cfg.get("distance_m", 0.6)
        target = str(cfg.get("target", "nose"))
        rate_hz = float(controlcfg.get("rate_hz", 10.0))
        visibility_threshold = float(cfg.get("visibility_threshold", 0.5))
        xy_bounds = controlcfg.get("xy_bounds")
        z_bounds = controlcfg.get("z_bounds")
        hold_position = bool(cfg.get("hold_position", False))
        preserve_roll = bool(cfg.get("preserve_roll", True))
        slow_factor = float(controlcfg.get("slow_factor", 1.5))       # >=1.0; 1.5 ≈ 33% slower
        pos_step_scale = float(controlcfg.get("pos_step_scale", 1.0 / max(slow_factor, 1e-6)))  # default ties to slow_factor

        servo_hz = int(controlcfg.get("servo_hz", 100))
        servo_lookahead = float(controlcfg.get("servo_lookahead", 0.08))
        servo_gain = float(controlcfg.get("servo_gain", 300.0))
        lp_alpha_pos = float(controlcfg.get("lp_alpha_pos", 0.05))
        lp_alpha_dir = float(controlcfg.get("lp_alpha_dir", 0.05))
        max_step_m = float(controlcfg.get("max_step_m", 0.002))
        max_ang_deg_per_s = float(cfg.get("max_ang_deg_per_s", 45.0))
        warm_start_s = float(cfg.get("warm_start_s", 0.8))
        lost_hold_s = float(cfg.get("lost_target_hold_s", 0.3))
        hold_orientation = bool(cfg.get("hold_orientation", True))
        keep_z = bool(cfg.get("keep_z", True))
        force_moveL = bool(cfg.get("force_moveL", False))
        debug_print_every = int(cfg.get("debug_print_every", 20))

        auto_idle_on_loss = bool(cfg.get("auto_idle_on_loss", True))
        loss_timeout_s = float(cfg.get("loss_timeout_s", 2.0))
        pitch_offset_deg = float(cfg.get("pitch_offset_deg", 0.0))

        # ensure tracker is running
        self.manager.start_tracker()

        use_servo = hasattr(self.arm, "servoL") and not force_moveL
        # dt = 1.0 / max(1, int(servo_hz if use_servo else max(1, int(rate_hz))))
        base_rate = int(servo_hz if use_servo else max(1, int(rate_hz)))
        dt_base = 1.0 / max(1, base_rate)
        dt = dt_base * max(1.0, slow_factor)  # larger dt => slower, keeps math consisten

        t0 = time.perf_counter()
        tick = 0

        T0 = self.arm.get_T_base_tcp(); p0 = T0[:3,3].copy(); R0 = T0[:3,:3].copy()
        p_cmd = p0.copy(); R_cmd = R0.copy(); R_prev = R0.copy(); z_cmd_dir = R0[:,2].copy()
        last_seen_t = time.perf_counter()

        landmark_map = self.manager.landmark_map

        try:
            while not stop_event.is_set():
                target_time = t0 + tick * dt
                now = time.perf_counter()
                if (target_time - now) > 0:
                    time.sleep(target_time - now)
                tick += 1
                t = time.perf_counter() - t0

                body = self.tracker.get_body_positions(
                    transform_4x4=self.manager.T_base_fixed_camera,
                    filter_visible=True,
                    visibility_threshold=visibility_threshold,
                )
                have_target = False
                p_target = None
                if has_any(body):
                    p_target = self.select_target_point(body, target, landmark_map)
                    if p_target is not None and not np.any(np.isnan(p_target)):
                        have_target = True
                        last_seen_t = time.perf_counter()

                T_now = self.arm.get_T_base_tcp(); p_now = T_now[:3,3].copy()

                if hold_position:
                    p_cmd = p0.copy()
                else:
                    if have_target:
                        z_dir_now = norm(p_target - p_now)
                        p_des = p_target - z_dir_now * float(distance_m)
                        if keep_z:
                            p_des[2] = p0[2]
                        # step_vec = lp_alpha_pos * (p_des - p_cmd)
                        step_vec = (lp_alpha_pos * pos_step_scale) * (p_des - p_cmd)

                        # step_len = float(np.linalg.norm(step_vec))
                        # if step_len > max_step_m:
                        #     step_vec *= (max_step_m / (step_len + 1e-9))
                        step_len = float(np.linalg.norm(step_vec))
                        effective_max_step = max_step_m * pos_step_scale
                        if step_len > effective_max_step:
                            step_vec *= (effective_max_step / (step_len + 1e-9))
                        p_cmd += step_vec


                if hold_orientation:
                    R_des = R0
                else:
                    if have_target:
                        z_des = norm((p_target - p_cmd))
                        z_cmd_dir = norm(z_cmd_dir + lp_alpha_dir * (z_des - z_cmd_dir))
                    else:
                        if (time.perf_counter() - last_seen_t) > lost_hold_s:
                            z_back = R0[:,2]
                            z_cmd_dir = norm(z_cmd_dir + lp_alpha_dir * (z_back - z_cmd_dir))
                    R_target = look_at_relative(
                        from_p=p_cmd,
                        to_p=(p_cmd + z_cmd_dir),
                        R_ref=R0,
                        up_hint=np.array([0.0,0.0,1.0]),
                        preserve_roll=preserve_roll,
                    )

                    # --- NEW: apply pitch offset in TCP frame (positive = tilt up) ---
                    if abs(pitch_offset_deg) > 1e-6:
                        pitch_rad = math.radians(pitch_offset_deg)
                        y_axis_local = R_target[:, 1]                    # TCP Y-axis
                        R_target = R_target @ exp_so3(y_axis_local * pitch_rad)
                    # -----------------------------------------------------------------

                    warm_a = 1.0 if warm_start_s <= 1e-3 else min(1.0, (time.perf_counter()-t0)/warm_start_s)
                    R_des = slerp_R(R0, R_target, warm_a)

                # limit angular rate
                R_rel = R_prev.T @ R_des
                w = log_so3(R_rel)
                ang = float(np.linalg.norm(w))
                max_step = math.radians(max_ang_deg_per_s) * dt
                if ang > 1e-9:
                    scale = min(1.0, max_step/ang)
                    R_cmd = R_prev @ exp_so3(w * scale)
                else:
                    R_cmd = R_des.copy()
                R_prev = R_cmd.copy()
    
                T_cmd = np.eye(4, dtype=float); T_cmd[:3,:3] = R_cmd; T_cmd[:3,3] = p_cmd
                try:
                    from commons.grasp_utils import pose_from_T
                    pose6 = pose_from_T(T_cmd)
        
                    if xy_bounds is not None:
                        pose6[0] = clamp(pose6[0], float(xy_bounds[0][0]), float(xy_bounds[0][1]))
                        pose6[1] = clamp(pose6[1], float(xy_bounds[1][0]), float(xy_bounds[1][1]))
                    if z_bounds is not None:
                        pose6[2] = clamp(pose6[2], float(z_bounds[0]), float(z_bounds[1]))

                    if use_servo:
                        self.arm.servoL(pose=pose6, time_s=dt, lookahead_time=servo_lookahead, gain=servo_gain)
                    else:
                        ms = float(self.cfg.get("motion",{}).get("move_speed",0.20))
                        ma = float(self.cfg.get("motion",{}).get("move_accel",0.60))
                        self.arm.moveL(pose=pose6, speed=ms, accel=ma)
              
                except Exception as e:
                    print(f"[point_at_person] servo/move error: {e}")


                if debug_print_every>0 and (tick % debug_print_every==0) and self.debug:
                    dist = float(np.linalg.norm((p_target - p_cmd))) if (p_target is not None) else float('nan')
                    print(f"[point] t={t:.3f}s have={have_target} p_cmd={np.round(p_cmd,3)} d≈{dist:.3f}")
     
                if auto_idle_on_loss and (not have_target) and ((time.perf_counter() - last_seen_t) > loss_timeout_s):
                    def _switch():
                        self.manager.start_action("idle_lookaround")
                    threading.Thread(target=_switch, daemon=True).start()
                    stop_event.set()
                    break


        except Exception as e:
            print('[point_at_person] error', e)
