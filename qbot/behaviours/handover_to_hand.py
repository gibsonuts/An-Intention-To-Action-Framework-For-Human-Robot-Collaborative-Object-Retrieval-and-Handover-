# behaviours/handover_to_hand.py
from __future__ import annotations
import math, time, threading
import numpy as np
from .base_action import BaseAction, register_action
from commons.grasp_utils import norm, clamp, look_at_relative, log_so3, exp_so3, pose_from_T

@register_action("handover")
@register_action("handover_to_hand")
class HandoverToHandAction(BaseAction):
    def run(self, stop_event: threading.Event, **kwargs):
        cfgh = self.cfg.get("handover", {})
        controlcfg = self.cfg.get("control", {})

        # --- hand selection mode ------------------------------------------------
        hand_cfg = (kwargs.get("hand") or cfgh.get("hand", "either")).lower().strip()
        # accepted modes: 'left', 'right' (fixed), or 'either'/'auto'/'presented'/'any' (auto-select)
        auto_modes = {"either", "auto", "presented", "any"}
        hand_mode = hand_cfg if hand_cfg in ({"left", "right"} | auto_modes) else "right"
        # keep track of which hand we're currently targeting to avoid jitter
        current_hand = None  # 'left' or 'right'
        # -----------------------------------------------------------------------

        # --- control/servo and filters -----------------------------------------
        servo_hz = int(controlcfg.get("servo_hz", 100))
        servo_lookahead = float(controlcfg.get("servo_lookahead", 0.06))
        servo_gain = float(controlcfg.get("servo_gain", 300.0))
        lp_alpha_pos = float(controlcfg.get("lp_alpha_pos", 0.08))
        lp_alpha_dir = float(controlcfg.get("lp_alpha_dir", 0.08))
        max_step_m = float(controlcfg.get("max_step_m", 0.003))
        max_ang_deg = float(controlcfg.get("max_ang_deg", 45.0))  # approach cap before scaling
        xy_bounds = controlcfg.get("xy_bounds")
        max_xy_bounds = controlcfg.get("xy_bounds")
        z_bounds = controlcfg.get("z_bounds")
        rate_hz = float(controlcfg.get("rate_hz", 10.0))
        # slow purely by time (dt) and/or by smaller positional steps (no gain changes)
        slow_factor = float(controlcfg.get("slow_factor", 1.5))       # >=1.0; 1.5 ≈ 33% slower
        pos_step_scale = float(controlcfg.get("pos_step_scale", 1.0 / max(slow_factor, 1e-6)))  # default ties to slow_factor
        ang_step_scale = float(controlcfg.get("ang_step_scale", pos_step_scale))
       
        # --- behavior thresholds ------------------------------------------------
        standoff_m = float(cfgh.get("standoff_m", 0.10))
        far_threshold_m = float(cfgh.get("far_threshold_m", 1.0))
        keep_z = bool(cfgh.get("keep_z", True))
        ready_location = cfgh.get("ready_location")
        preserve_roll = bool(cfgh.get("preserve_roll", True))
        visibility_threshold = float(cfgh.get("visibility_threshold", 0.5))
        rotation_mode = str(cfgh.get("rotation_mode", "none")).lower().strip()
        force_total_thresh_N = float(cfgh.get("force_total_thresh_N", 10.0))
        force_pull_z_thresh_N = float(cfgh.get("force_pull_z_thresh_N", 6.0))
        grabbed_min_ticks = int(cfgh.get("grabbed_min_ticks", 5))
        max_wait_s = float(cfgh.get("max_wait_s", 15.0))
        hold_time_s = float(cfgh.get("hold_time_s", 3.0))
        post_open_wait_s = float(cfgh.get("post_open_wait_s", 0.3))

        # --- return-to-start speedups (only used when no hand is visible) ---
        # >1.0 = faster than normal stepping; kept separate to avoid affecting hand approach
        return_speed_mult = float(cfgh.get("return_speed_mult", 3.0))          # positional step multiplier
        return_ang_deg_per_s = float(cfgh.get("return_ang_deg_per_s", 120.0))  # angular cap during return
        return_near_m = float(cfgh.get("return_near_m", 0.08))                 # begin taper within this distance
        return_move_speed_mult = float(cfgh.get("return_move_speed_mult", 2.0))# for moveL fallback

        # --- presentation (stretch) gating -------------------------------------
        stretch_min_extension_m = float(cfgh.get("stretch_min_extension_m", 0.25))   # min wrist-shoulder distance
        stretch_max_bend_deg    = float(cfgh.get("stretch_max_bend_deg", 20.0))      # elbow angle threshold
        not_stretched_return_delay_s = float(cfgh.get("not_stretched_return_delay_s", 0.3))  # small grace

        # --- NEW: sticky hand selection knobs ----------------------------------
        switch_hysteresis_m   = float(cfgh.get("switch_hysteresis_m", 0.08))   # other hand must be this much closer
        switch_min_lock_s     = float(cfgh.get("switch_min_lock_s", 0.8))      # refuse to switch for this long
        hand_loss_grace_s     = float(cfgh.get("hand_loss_grace_s", 0.4))      # tolerate brief occlusion
        prefer_stretched_only = bool(cfgh.get("prefer_stretched_only", True))  # only switch to "presented" hand

        # --- HANDS-ONLY FALLBACK OPTIONS --------------------------------------
        # If the pose/body isn't visible, optionally fall back to MediaPipe Hands only.
        allow_hands_only = bool(cfgh.get("allow_hands_only", True))
        # Where on the hand to target when using hand-only: "wrist" (LM 0) or "palm_center" (avg of palm keypoints)
        hand_target_mode = str(cfgh.get("hand_target_mode", "wrist")).lower().strip()  # 'wrist'|'palm_center'
        # Minimum number of known 3D hand landmarks required to consider a hand valid in hands-only mode
        min_hand_points = int(cfgh.get("min_hand_points", 5))

        # ensure wrists/elbows/shoulders are known to the tracker
        try:
            pose = self.manager.pose_module
            lm = self.manager.landmark_map
            lm.setdefault("left_wrist",  pose.PoseLandmark.LEFT_WRIST.value)
            lm.setdefault("right_wrist", pose.PoseLandmark.RIGHT_WRIST.value)
            lm.setdefault("left_elbow",  pose.PoseLandmark.LEFT_ELBOW.value)
            lm.setdefault("right_elbow", pose.PoseLandmark.RIGHT_ELBOW.value)
            lm.setdefault("left_shoulder",  pose.PoseLandmark.LEFT_SHOULDER.value)
            lm.setdefault("right_shoulder", pose.PoseLandmark.RIGHT_SHOULDER.value)
        except Exception:
            pass

        # tracker
        self.manager.start_tracker()
        
        
        #move to read location firsts
        print('moving to ready location')
        self.arm.moveJ(ready_location)

        # gripper set to close
        # self.hw.gripper.close()

        use_servo = hasattr(self.arm, "servoL")
        base_rate = int(servo_hz if use_servo else max(1, int(rate_hz)))
        dt_base = 1.0 / max(1, base_rate)
        dt = dt_base * max(1.0, slow_factor)  # larger dt => slower, keeps math consistent

        t0 = time.perf_counter(); tick = 0

        # T0 = self.arm.get_T_base_tcp(); p0 = T0[:3,3].copy(); R0 = T0[:3,:3].copy()
        T0 = self.arm.get_T_base_tcp() @ self.hw.T_tcp_to_gripper; p0 = T0[:3,3].copy(); R0 = T0[:3,:3].copy()
        p_cmd = p0.copy(); z_cmd_dir = R0[:,2].copy(); R_prev = R0.copy()

        landmark_map = self.manager.landmark_map
        last_seen = time.perf_counter()
        grabbed_ticks = 0

        last_stretched_time = time.perf_counter()
        force_return = False
        hold_deadline = None  # when to resume motion after pausing near the hand

        # sticky selection state
        last_switch_time = time.perf_counter()
        last_seen_hand = {"left": -1e9, "right": -1e9}


        def _get_T_base_tip():
            T = self.arm.get_T_base_tcp()
            return T @ self.hw.T_tcp_to_gripper

        def _get_T_tip_to_tcp(T):
            return T @ np.linalg.inv(self.hw.T_tcp_to_gripper)

            # return self.arm.get_T_base_tcp()
        
        # ---------------------------- helpers ---------------------------------
        def _pt(body, name: str):
            return self.select_target_point(body, name, landmark_map)

        # elbow angle + extension check → "presented hand"
        def _hand_stretched_out(body, which: str) -> bool:
            if which not in ("left", "right"):
                return False
            wrist  = _pt(body, f"{which}_wrist")
            elbow  = _pt(body, f"{which}_elbow")
            shoulder = _pt(body, f"{which}_shoulder")
            if wrist is None or elbow is None or shoulder is None:
                return False
            if np.any(np.isnan(wrist)) or np.any(np.isnan(elbow)) or np.any(np.isnan(shoulder)):
                return False

            v_ew = wrist - elbow      # elbow -> wrist
            v_es = shoulder - elbow   # elbow -> shoulder
            a = np.linalg.norm(v_ew); b = np.linalg.norm(v_es)
            if a < 1e-6 or b < 1e-6:
                return False

            cosang = float(np.clip(np.dot(v_ew, v_es) / (a*b), -1.0, 1.0))
            angle_deg = math.degrees(math.acos(cosang))  # ~180 when straight
            bend_deg  = 180.0 - angle_deg                # 0 when straight

            extension = float(np.linalg.norm(wrist - shoulder))
            if self.debug:
                print("[handover] ext=%.3f(m) thr=%.3f | bend=%.1f(°) max=%.1f"
                      % (extension, stretch_min_extension_m, bend_deg, stretch_max_bend_deg))

            extension_ok = extension >= stretch_min_extension_m
            bend_ok = bend_deg >= stretch_max_bend_deg
            return bool(extension_ok and bend_ok)

        def _apply_clamps(p):
            if xy_bounds is not None:
                p[0] = clamp(p[0], float(xy_bounds[0][0]), float(xy_bounds[0][1]))
                p[1] = clamp(p[1], float(xy_bounds[1][0]), float(xy_bounds[1][1]))
            if z_bounds is not None:
                p[2] = clamp(p[2], float(z_bounds[0]), float(z_bounds[1]))
            return p

        def _read_force_tool(R_prev = None):
            FxFyFz = np.zeros(3, dtype=float)
            wrench = self.arm.get_wrench()  # [Fx,Fy,Fz,Mx,My,Mz] in BASE
            FxFyFz = np.array(wrench[:3], dtype=float).reshape(3)
        
            if R_prev is not None:
                FxFyFz = R_prev.T @ FxFyFz
            return FxFyFz, float(np.linalg.norm(FxFyFz))

        def _near_target(p_hand, thresh=0.02):
            return float(np.linalg.norm((p_hand - p_cmd))) <= float(thresh)

        def _far_target(p_hand, thresh=0.02):
            return float(np.linalg.norm((p_hand - p_cmd))) >= float(thresh)

        def _make_T(R: np.ndarray, p: np.ndarray) -> np.ndarray:
            T = np.eye(4, dtype=float); T[:3,:3] = R; T[:3,3] = p
            return T

        def _command_step(R_cmd: np.ndarray, p_cmd_now: np.ndarray, *, dt_: float,transform: np.ndarray=None, allow_servo: bool=True):
            """Single place to actually send motion to the robot (servoL or moveL).
            Applies clamps and forms pose6 from R,p. Used by both 'go to hand' and 'go back to start'."""
            T = _make_T(R_cmd, p_cmd_now)
            if transform is not None:
               T = T@transform

            pose6 = pose_from_T(T)
           

            pose6 = _apply_clamps(pose6)
            if not allow_servo:
                # force moveL
                ms = float(self.cfg.get("motion",{}).get("move_speed",0.20))
                ma = float(self.cfg.get("motion",{}).get("move_accel",0.60))
                self.arm.moveL(pose=pose6, speed=ms, accel=ma)
            else:
                if use_servo:
                    self.arm.servoL(pose=pose6, time_s=dt_, lookahead_time=servo_lookahead, gain=servo_gain)
                else:
                    ms = float(self.cfg.get("motion",{}).get("move_speed",0.20))
                    ma = float(self.cfg.get("motion",{}).get("move_accel",0.60))
                    self.arm.moveL(pose=pose6, speed=ms, accel=ma)

        def _return_step(p_now: np.ndarray, R_now: np.ndarray):
            """Compute one step toward home (p0/R0) with return multipliers. Returns (p_cmd_next, R_cmd_next)."""
            # Position: step toward start (optionally keep current Z), faster when farther
            p_des = p0.copy()
            if keep_z:
                p_des[2] = p_now[2]  # glide in XY only
            to_home = (p_des - p_now)
            dist_home = float(np.linalg.norm(to_home))
            # farther => closer to 1.0; near home => taper but keep ≥0.5 to avoid crawl
            taper = float(np.clip(dist_home / max(1e-6, return_near_m), 0.5, 1.0))
            step_vec = (lp_alpha_pos * pos_step_scale) * to_home
            step_vec *= (return_speed_mult * taper)
            step_len = float(np.linalg.norm(step_vec))
            effective_max_step = max_step_m * pos_step_scale * return_speed_mult
            if step_len > effective_max_step:
                step_vec *= (effective_max_step / (step_len + 1e-9))
            p_next = _apply_clamps(p_now + step_vec)

            # Orientation: slerp toward R0 with a larger (but capped) angular step
            R_rel = R_now.T @ R0
            w = log_so3(R_rel)
            max_ang_step = math.radians(min(return_ang_deg_per_s, 180.0)) * dt * ang_step_scale
            ang = float(np.linalg.norm(w))
            if ang > 1e-9:
                scale = min(1.0, max_ang_step / ang)
                R_next = R_now @ exp_so3(w * scale)
            else:
                R_next = R0.copy()
            return p_next, R_next

        def _go_back_to_start():
            """After handover, go back to (p0,R0) using the same stepper. Obeys stop_event."""
            try:
                # speed up the discrete moveL path a bit if no servo
                force_moveL_speed_mult = return_move_speed_mult
                while not stop_event.is_set():
                    # T_now = self.arm.get_T_base_tcp(); p_now = T_now[:3,3].copy(); R_now = T_now[:3,:3].copy()
                    T_now =  _get_T_base_tip(); p_now = T_now[:3,3].copy(); R_now = T_now[:3,:3].copy()
            
                    to_home_full = np.linalg.norm(np.hstack([
                        (p_now - (p0 if not keep_z else np.array([p0[0], p0[1], p_now[2]]))),
                        log_so3(R_now.T @ R0)
                    ]))
                    if to_home_full < 1e-2:  # small pos+ang error => done
                        break
                    p_n, R_n = _return_step(p_now, R_now)
                    if use_servo:
                        _command_step(R_n, p_n, transform=np.linalg.inv(self.hw.T_tcp_to_gripper), dt_=dt, allow_servo=True)
                    else:
                        ms = float(self.cfg.get("motion",{}).get("move_speed",0.20)) * force_moveL_speed_mult
                        ma = float(self.cfg.get("motion",{}).get("move_accel",0.60)) * force_moveL_speed_mult
                        pose6 = pose_from_T(_make_T(R_n, p_n))
                        pose6 = _apply_clamps(pose6)
                        self.arm.moveL(pose=pose6, speed=ms, accel=ma)
                    time.sleep(dt)
            except Exception:
                pass

        # --- sticky hand selector (BODY-BASED) --------------------------------
        def _select_presented_hand(body, p_ref):
            """
            Return (p_hand, chosen_hand_str) with hysteresis and lockout to avoid flip-flopping.
            Keeps current_hand unless clearly worse for long enough or truly lost.
            """
            nonlocal current_hand, last_switch_time, last_seen_hand

            def _wrist(name):
                p = self.select_target_point(body, name, landmark_map)
                return None if (p is None or np.any(np.isnan(p))) else p

            def _first_non_none(a, b):
                return a if a is not None else b

            # fixed modes: respect user's choice
            if hand_mode in ("left", "right"):
                name = "left_wrist" if hand_mode == "left" else "right_wrist"
                p = _wrist(name)
                if p is not None:
                    last_seen_hand[hand_mode] = time.perf_counter()
                    return p, hand_mode
                return None, None

            # auto modes
            pL, pR = _wrist("left_wrist"), _wrist("right_wrist")
            vis = {"left": pL is not None, "right": pR is not None}
            now = time.perf_counter()

            # update last seen stamps
            if vis["left"]:  last_seen_hand["left"]  = now
            if vis["right"]: last_seen_hand["right"] = now

            if current_hand in ("left","right"):
                cur = current_hand
                still_ok = vis[cur] or (now - last_seen_hand[cur] <= hand_loss_grace_s)
                if still_ok:
                    other = "right" if cur == "left" else "left"
                    can_consider = (now - last_switch_time) >= switch_min_lock_s and vis[other]
                    if can_consider:
                        p_cur   = pL if cur   == "left"  else pR
                        p_other = pR if other == "right" else pL
                        if p_cur is None:
                            current_hand = other
                            last_switch_time = now
                            return p_other, other
                        d_cur   = float(np.linalg.norm(p_cur - p_ref))
                        d_other = float(np.linalg.norm(p_other - p_ref))
                        closer_enough = (d_cur - d_other) >= switch_hysteresis_m
                        presented_ok = (not prefer_stretched_only) or _hand_stretched_out(body, other)
                        if closer_enough and presented_ok:
                            current_hand = other
                            last_switch_time = now
                            return p_other, other

                    # stick with current; if its point is None (grace case), fall back to the other
                    p_cur = pL if cur=="left" else pR
                    p_fallback = pR if cur=="left" else pL
                    return _first_non_none(p_cur, p_fallback), cur

            # no current hand yet (or truly lost): pick start hand
            candidates = []
            if vis["left"]:
                scoreL = float(np.linalg.norm(pL - p_ref))
                prefL  = _hand_stretched_out(body, "left") if prefer_stretched_only else True
                candidates.append(("left", pL, scoreL, prefL))
            if vis["right"]:
                scoreR = float(np.linalg.norm(pR - p_ref))
                prefR  = _hand_stretched_out(body, "right") if prefer_stretched_only else True
                candidates.append(("right", pR, scoreR, prefR))

            if not candidates:
                return None, None

            preferred = [c for c in candidates if c[3]]
            pool = preferred if preferred else candidates
            hand, p, _score, _ = min(pool, key=lambda c: c[2])
            current_hand = hand
            last_switch_time = time.perf_counter()
            return p, hand

        # --- HANDS-ONLY SELECTOR ----------------------------------------------
        def _hand_palm_center(hand_xyz: dict[int, np.ndarray]) -> np.ndarray | None:
            """
            Compute a robust palm/centroid point from a dict of 21 MediaPipe Hands landmarks in 3D.
            Uses wrist (0) + MCPs (5,9,13,17) by default; falls back to average of all valid if needed.
            """
            if not hand_xyz:
                return None
            # Try preferred subset
            idxs = [0, 5, 9, 13, 17] if hand_target_mode != "wrist" else [0]
            pts = []
            for i in idxs:
                if i in hand_xyz:
                    p = hand_xyz[i]
                    if p is not None and not np.isnan(p).any():
                        pts.append(p)
            if not pts and hand_target_mode == "wrist":
                return None
            if not pts:
                # fallback: average of all valid points
                for i, p in hand_xyz.items():
                    if p is not None and not np.isnan(p).any():
                        pts.append(p)
            if len(pts) == 0:
                return None
            return np.mean(np.vstack(pts), axis=0)

        def _select_hand_only(p_ref: np.ndarray):
            """
            Choose the closest hand (from MediaPipe Hands 3D) to the tool, independent of left/right.
            Returns (p_hand, idx) where idx is the index in hands array (for debugging).
            """
            try:
                hands_all = self.tracker.get_all_hands_positions(
                    transform_4x4=self.manager.T_base_fixed_camera
                )
            except Exception:
                hands_all = []
            if not hands_all:
                return None, None
            candidates = []
            for hi, hdict in enumerate(hands_all):
                # Discard if too few valid 3D points
                valid = [p for p in hdict.values() if p is not None and not np.isnan(p).any()]
                if len(valid) < min_hand_points:
                    continue
                p = _hand_palm_center(hdict)
                if p is None or np.isnan(p).any():
                    continue
                score = float(np.linalg.norm(p - p_ref))
                candidates.append((score, hi, p))
            if not candidates:
                return None, None
            _, idx, p_sel = min(candidates, key=lambda c: c[0])
            return p_sel, idx

        # -------------------------- main loop ---------------------------------
        end_status, end_reason = 'unknown','unknown'
        pause_motion = False
        Start_F_tool, Start_Fmag = _read_force_tool()
        try:
            while not stop_event.is_set():
                target_time = t0 + tick * dt
                now = time.perf_counter()
                if (target_time - now) > 0:
                    time.sleep(target_time - now)
                tick += 1

                # read current pose early so we can choose the closest wrist/hand
                T_now_for_choice = _get_T_base_tip()
                p_now_for_choice = T_now_for_choice[:3,3].copy()

                # Try BODY first
                body = self.tracker.get_body_positions(
                    transform_4x4=self.manager.T_base_fixed_camera,
                    filter_visible=True,
                    visibility_threshold=visibility_threshold,
                )

                have_target = False; p_hand = None
                used_hands_only = False

                if body:
                    # BODY-BASED selection (left/right + presentation heuristics)
                    p_sel, chosen = _select_presented_hand(body, p_ref=p_now_for_choice)
                    if p_sel is not None:
                        have_target = True
                        p_hand = p_sel
                        last_seen = time.perf_counter()
                        if chosen is not None:
                            # current_hand is set inside chooser when a switch happens
                            if self.debug and (tick % 10 == 0):
                                print(f"[handover] tracking {current_hand} hand")
                            which = current_hand if current_hand in ("left", "right") else (chosen if chosen in ("left", "right") else None)
                            if which is not None and _hand_stretched_out(body, which):
                                last_stretched_time = time.perf_counter()
                                force_return = False     # good presentation => we can approach
                            else:
                                # Not stretched: optionally wait a small grace, then start returning to start
                                if (time.perf_counter() - last_stretched_time) >= not_stretched_return_delay_s:
                                    if self.debug and (tick % 10 == 0):
                                        print(f"[handover] hand not sufficiently presented; returning to start")
                                    force_return = True

                # If no BODY target and allowed, FALL BACK to HANDS-ONLY
                if (not have_target) and allow_hands_only:
                    p_sel, idx = _select_hand_only(p_now_for_choice)
                    if p_sel is not None:
                        have_target = True
                        p_hand = p_sel
                        used_hands_only = True
                        last_seen = time.perf_counter()
                        # When using hands-only, disable "return due to not presented" logic
                        force_return = False
                        # Do not try to decide left/right; keep current_hand as-is (None)
                        # if self.debug and (tick % 10 == 0):
                            # print(f"[handover] hands-only fallback active (hand #{idx})")

                # --- No target path (works whether no body and/or no hands) ---
                if not have_target:
                    if (time.perf_counter() - last_seen) > max_wait_s:
                        if self.debug:
                            print("[handover] No hand visible; returning to start.")
                        # Current state
                        # T_now = self.arm.get_T_base_tcp()
                        T_now =  _get_T_base_tip()
                        p_now = T_now[:3,3].copy()
                        R_now = T_now[:3,:3].copy()
                        # Compute one return step and command
                        p_cmd, R_prev = _return_step(p_now, R_now)
                        _command_step(R_prev, p_cmd, transform=np.linalg.inv(self.hw.T_tcp_to_gripper), dt_=dt, allow_servo=True)

                    # Hover (no target): keep current pose as reference and continue
                    # T_now = self.arm.get_T_base_tcp()
                    T_now =  _get_T_base_tip()
                    p_cmd = T_now[:3,3].copy()
                    R_prev = T_now[:3,:3].copy()
                    continue

                # if not pause_motion:
                #     if force_return and (not used_hands_only):
                #         # Only enforce "return" behavior when using BODY gating logic.
                #         T_now = self.arm.get_T_base_tcp()
                #         p_now = T_now[:3,3].copy()
                #         R_now = T_now[:3,:3].copy()
                #         p_cmd, R_prev = _return_step(p_now, R_now)
                #         _command_step(R_prev, p_cmd, dt_=dt, allow_servo=True)
                #         # also keep p_cmd/R_prev fresh for the next iteration
                #         continue

                #     # --- Targeted approach path (common to body or hands-only) ---
                #     T_now = _get_T_base_tip(); p_now = T_now[:3,3].copy()
                #     print(p_now,self.arm.get_T_base_tcp()[:3,3])
                    
                #     z_dir_now = norm(p_hand - p_now)
                if not pause_motion:
                    if force_return and (not used_hands_only):
                        # Only enforce "return" behavior when using BODY gating logic.
                        # T_now = self.arm.get_T_base_tcp()
                        T_now =  _get_T_base_tip()
                        p_now = T_now[:3,3].copy()
                        R_now = T_now[:3,:3].copy()
                        p_cmd, R_prev = _return_step(p_now, R_now)
                        _command_step(R_prev, p_cmd, transform=np.linalg.inv(self.hw.T_tcp_to_gripper), dt_=dt, allow_servo=True)
                        # also keep p_cmd/R_prev fresh for the next iteration
                        continue

                    # --- Targeted approach path (common to body or hands-only) ---
                    T_now = _get_T_base_tip(); p_now = T_now[:3,3].copy()
                    # T_now = self.arm.get_T_base_tcp();p_now = T_now[:3,3].copy()
                    # print(p_now, self.arm.get_T_base_tcp()[:3,3])

                    # --- dynamic y-bound: keep TCP at least 10 mm from hand in Y ---
                    # safe_offset = 0.1  # this includes the length of gripper
                    # y_robot = float(p_now[1])
                    # y_hand  = float(p_hand[1])

                    # # initialise xy_bounds if not configured
                    # if xy_bounds is None:
                    #     # very loose X bounds, Y will be controlled by the hand
                    #     xy_bounds = [[-float("inf"), float("inf")],
                    #                  [-float("inf"), float("inf")]]
    
                    # y_min, y_max = xy_bounds[1]
                    # y_max_safe = y_hand + safe_offset 
                    # new_y_min = max(y_max_safe,max_xy_bounds[1][0])
                    # if (new_y_min-y_robot > 0.2):#prevents jerkyness
                    #     print('too jerky',y_min ,'->',y_max_safe,' xy_bounds[1][0] ', xy_bounds[1][0] )
                    # else:
                    #     xy_bounds[1][0] = max(y_max_safe,max_xy_bounds[1][0])
                    #     print('y_min',y_min ,'->',y_max_safe,' xy_bounds[1][0] ', xy_bounds[1][0] )
                    # ----------------------------------------------------------------

                    z_dir_now = norm(p_hand - p_now)
                    p_des = p_hand - z_dir_now * float(standoff_m)
                    if keep_z:
                        p_des[2] = p0[2]
                    step_vec = (lp_alpha_pos * pos_step_scale) * (p_des - p_cmd)

                    # approach slowdown based on distance to hand
                    dist_to_hand = float(np.linalg.norm(p_des - p_now))
                    near_m = max(0.02, 0.5 * float(standoff_m))  # taper within ~standoff/2 (≥2 cm)
                    speed_scale = float(np.clip(dist_to_hand / near_m, 0.25, 1.0))  # never fully stop
                    step_vec *= speed_scale

                    step_len = float(np.linalg.norm(step_vec))
                    effective_max_step = max_step_m * pos_step_scale
                    if step_len > effective_max_step:
                        step_vec *= (effective_max_step / (step_len + 1e-9))
                    p_cmd += step_vec
                    
                    
                    z_des = norm(p_hand - p_cmd)
                    z_cmd_dir = norm(z_cmd_dir + lp_alpha_dir * (z_des - z_cmd_dir))
                    R_target = look_at_relative(
                        from_p=p_cmd,
                        to_p=(p_cmd + z_cmd_dir),
                        R_ref=R0,
                        up_hint=np.array([0.0, 0.0, 1.0]),
                        preserve_roll=preserve_roll,
                    )

                    # --------- orientation update selectable by rotation_mode ----------
                    R_rel = R_prev.T @ R_target
                    max_ang_step = math.radians(max_ang_deg) * dt * ang_step_scale

                    if rotation_mode == "none":
                        # keep previous orientation, no change
                        R_cmd = R_prev.copy()

                    elif rotation_mode == "yaw":
                        # YAW-ONLY: rotate about base Z toward the target heading
                        # yaw angle from relative rotation
                        yaw = math.atan2(R_rel[1, 0], R_rel[0, 0])
                        # clamp yaw step
                        yaw_step = float(np.clip(yaw, -max_ang_step, max_ang_step))

                        cy, sy = math.cos(yaw_step), math.sin(yaw_step)
                        R_step = np.array(
                            [
                                [cy, -sy, 0.0],
                                [sy,  cy, 0.0],
                                [0.0, 0.0, 1.0],
                            ],
                            dtype=float,
                        )
                        # apply yaw step in base frame
                        R_cmd = R_prev @ R_step

                    else:
                        # DEFAULT / 'full': full 3D shortest rotation
                        w = log_so3(R_rel)
                        ang = float(np.linalg.norm(w))
                        if ang > 1e-9:
                            scale = min(1.0, max_ang_step / ang)
                            R_cmd = R_prev @ exp_so3(w * scale)
                        else:
                            R_cmd = R_target.copy()

                    R_prev = R_cmd.copy()


                    # R_rel = R_prev.T @ R_target
                    # w = log_so3(R_rel)
                    # max_ang_step = math.radians(max_ang_deg) * dt * ang_step_scale
                    # ang = float(np.linalg.norm(w))
                    # if ang > 1e-9:
                    #     scale = min(1.0, max_ang_step/ang)
                    #     R_cmd = R_prev @ exp_so3(w * scale)
                    # else:
                    #     R_cmd = R_target.copy()
                    # R_prev = R_cmd.copy()

                    _command_step(R_cmd, p_cmd, transform=np.linalg.inv(self.hw.T_tcp_to_gripper), dt_=dt, allow_servo=True)

                F_tool, Fmag = _read_force_tool()
                ForceDiff = abs(Start_Fmag - Fmag)
                if ForceDiff > 9 or not self.hw.gripper.is_holding_object(): #check if object is being held
                    grabbed_ticks+=1
                    if grabbed_ticks >= int(grabbed_min_ticks):
                        print('ForceDiff',ForceDiff)
                        end_status, end_reason = "completed", "success"
                        if self.debug: print("[handover] Grab detected!")
                        break
                else:
                    grabbed_ticks = 0

                # pause_motion = True
                # grab detection + timed hold/resume
                # if _near_target(p_hand, thresh=max(0.015, standoff_m)):
                #     if not pause_motion:
                #         # we just entered the near zone ⇒ start the hold timer and (optionally) re-bias
                #         hold_deadline = time.perf_counter() + hold_time_s
                #         try:
                #             self.arm.measure_wrench_bias()
                #         except Exception:
                #             pass

                #     F_tool, Fmag = _read_force_tool(R_prev)
                #     pull_ok  = (F_tool[2] >= float(force_pull_z_thresh_N))
                #     total_ok = (Fmag >= float(force_total_thresh_N))
                #     grabbed  = bool(pull_ok or total_ok)
                #     grabbed_ticks = grabbed_ticks + 1 if grabbed else 0

                #     pause_motion = True  # stay paused while in near zone
                #     if self.debug and (tick % 10 == 0):
                #         print(f"[handover] near wrist | F_tool={np.round(F_tool,2)} | Fmag={Fmag:.2f} | grabbed_ticks={grabbed_ticks}")

                #     if grabbed_ticks >= int(grabbed_min_ticks):
                #         end_status, end_reason = "completed", "success"
                #         if self.debug: print("[handover] Grab detected!")
                #         break

                # else:
                #     # not near target → if we had started a timed hold, release it when the deadline passes
                #     if hold_deadline is not None and time.perf_counter() >= hold_deadline:
                #         pause_motion = False
                #         hold_deadline = None

            # post-grab: stabilize + gripper + go back to start/ready
            try:
                self.manager.stop_all_actions()
                if end_status == "completed":
                    # optional holdI and dwell
                    # try:
                    #     self.hw.arm.holdI()
                    # except Exception:
                    #     end_status, end_reason = "fail", "holdi"
                    # time.sleep(max(0.0, float(hold_time_s)))
                    try:
                        self.hw.gripper.open()
                    except Exception:
                        end_status, end_reason = "fail", "gripper close"
                    time.sleep(max(0.0, float(post_open_wait_s)))

                    # Either step home or go to a named ready pose
                    try:
                        self.arm.moveJ(ready_location)
                    except Exception:
                        # fall back to stepped return if named pose missing
                        _go_back_to_start()

            finally:
                print('action completed')
                if end_status == "completed":
                    print('action success')
                    # try:
                    #     self.manager.start_action("idle_lookaround")
                    # except Exception:
                    #     pass
                self.complete(end_status, reason=end_reason,info='no errors')

        except Exception as e:
            print("[handover] error:", e)
            self.complete("error", reason="exception", info=str(e))
