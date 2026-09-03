import argparse
import base64
import io
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from detectors.sam3_object_detection import draw_mask_debug
from detectors.sam3_object_detection import masks_overlap
from detectors.sam3_object_detection import Sam3Detector
from hardware.hardware_init import HardwareInitializer
from hardware.screwdriver_client import ScrewdriverClient
from llm.openai_query_client import OpenAiQueryClient
from llm.openai_stt_llm_tts_client import StreamedSpeechPipelineClient
from qbot.qbot_cycles import ManualScrewCycleManager, ScrewCycleManager


ACTIVE_SCREWDRIVER_CLIENT = None
ACTIVE_VOICE_CLIENT = None


def handle_sigint(signum, frame):
    print("\n[CTRL-C] Cleaning up...")
    global ACTIVE_SCREWDRIVER_CLIENT, ACTIVE_VOICE_CLIENT

    if ACTIVE_SCREWDRIVER_CLIENT is not None:
        try:
            ACTIVE_SCREWDRIVER_CLIENT.stop()
        except Exception as e:
            print(f"[WARN] Failed to stop screwdriver client: {e}")

    if ACTIVE_VOICE_CLIENT is not None:
        try:
            ACTIVE_VOICE_CLIENT.stop()
        except Exception as e:
            print(f"[WARN] Failed to stop voice client: {e}")

    sys.exit(0)


signal.signal(signal.SIGINT, handle_sigint)


def read_stop_button_pressed(handles, *, active_high: bool = True) -> bool:
    """Normalize the raw stop input for active-high or active-low wiring."""
    raw_state = bool(handles.arm.get_stop_io())
    return raw_state if active_high else not raw_state


def wait_stop_button_released(
    handles,
    voice_client,
    poll: float = 0.05,
    *,
    active_high: bool = True,
):
    pressed = read_stop_button_pressed(handles, active_high=active_high)
    if pressed:
        message = "Please release the stop button to start operation."
        print(f"[STEP] {message}")
        if voice_client is not None:
            voice_client.speak_openai(message)

    while True:
        pressed = read_stop_button_pressed(handles, active_high=active_high)
        if not pressed:
            print("[STEP] Stop button released – starting operation.")
            return
        time.sleep(poll)


def check_tool_button_event(
    handles,
    *,
    listen_window_s: float = 0.5,
    long_press_sec: float = 2.5,
    poll: float = 0.05,
) -> Optional[str]:
    """
    Listen briefly for a tool-button interaction.

    Returns:
      - "short" when the trigger is pressed and released quickly
      - "long" when the trigger is held for at least `long_press_sec`
      - None when no press is observed within the listen window
    """
    deadline = time.time() + max(0.0, listen_window_s)
    pressed_start = None

    while time.time() < deadline or pressed_start is not None:
        pressed = bool(handles.arm.get_tool_io())
        now = time.time()

        if pressed:
            if pressed_start is None:
                pressed_start = now
            elif now - pressed_start >= long_press_sec:
                print(f"[EVENT] Long trigger hold detected (>{long_press_sec:.1f}s).")
                return "long"
        elif pressed_start is not None:
            held = now - pressed_start
            if held < long_press_sec:
                print(f"[EVENT] Short trigger press detected ({held:.2f}s).")
                return "short"
            pressed_start = None

        time.sleep(poll)

    return None


class ProcedureCancelled(RuntimeError):
    """Raised when the stop input requests cancellation of the active procedure."""


class StopIOCoordinator:
    def __init__(
        self,
        handles,
        *,
        voice_client: Optional[StreamedSpeechPipelineClient] = None,
        screwdriver_client: Optional[ScrewdriverClient] = None,
        poll: float = 0.05,
        stop_active_high: bool = True,
    ) -> None:
        self.handles = handles
        self.voice_client = voice_client
        self.screwdriver_client = screwdriver_client
        self.poll = poll
        self.stop_active_high = bool(stop_active_high)
        self._cancel_event = threading.Event()
        self._operation_active = threading.Event()
        self._shutdown_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._stop_latched = False
        self._cancel_notice_sent = False
        self._cancel_reason = "Operation cancelled by stop request."
        self._stop_released_handler = None
        self._central_reset_enabled = False
        self._interrupt_complete = threading.Event()
        self._interrupt_complete.set()
        self._interrupt_lock = threading.Lock()
        self._state_lock = threading.Lock()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._monitor_loop,
            name="stop-io-monitor",
            daemon=True,
        )
        self._thread.start()

    def shutdown(self) -> None:
        self._shutdown_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def begin_operation(self) -> None:
        with self._state_lock:
            self._operation_active.set()
            self._cancel_event.clear()
            self._cancel_notice_sent = False
            self._cancel_reason = "Operation cancelled by stop request."
            try:
                self._stop_latched = read_stop_button_pressed(
                    self.handles,
                    active_high=self.stop_active_high,
                )
            except Exception as e:
                print(f"[WARN] Failed to read stop button state at operation start: {e}")
                self._stop_latched = False
            if self._stop_latched:
                self._cancel_event.set()
                self._cancel_notice_sent = True
                self._interrupt_hardware_async()

    def end_operation(self) -> None:
        with self._state_lock:
            self._operation_active.clear()
            if not (self._central_reset_enabled and self._cancel_event.is_set()):
                self._cancel_event.clear()
                self._cancel_notice_sent = False
                self._cancel_reason = "Operation cancelled by stop request."

    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def raise_if_cancelled(self) -> None:
        if self._cancel_event.is_set():
            raise ProcedureCancelled(self._cancel_reason)

    def is_stop_held(self) -> bool:
        """Return True while the physical stop input is held down."""
        try:
            pressed = read_stop_button_pressed(
                self.handles,
                active_high=self.stop_active_high,
            )
        except Exception:
            with self._state_lock:
                return bool(self._stop_latched)
        with self._state_lock:
            self._stop_latched = pressed
        return pressed

    def request_cancel(self, reason: str = "System reset requested.") -> None:
        """Cancel current work and interrupt hardware, including from an LLM tool."""
        with self._state_lock:
            already_cancelled = self._cancel_event.is_set()
            self._cancel_reason = str(reason or "System reset requested.")
            self._cancel_event.set()
        print(f"[EVENT] {self._cancel_reason}")
        if not already_cancelled:
            self._interrupt_hardware_async()

    def wait_for_hardware_interrupt(self, timeout: float) -> bool:
        """Wait until asynchronous stop commands have returned."""
        return self._interrupt_complete.wait(timeout=max(0.1, float(timeout)))

    def set_stop_released_handler(self, handler) -> None:
        """Register a callback that resets the robot after the stop input is released."""
        self._stop_released_handler = handler

    def enable_central_reset(self) -> None:
        """Let an external reset controller exclusively own return-to-start motion."""
        self._central_reset_enabled = True

    def central_reset_enabled(self) -> bool:
        return self._central_reset_enabled

    def complete_reset(self) -> None:
        """Clear cancellation only after reset motion completes and stop is released."""
        with self._state_lock:
            if not self._stop_latched:
                self._cancel_event.clear()
                self._cancel_notice_sent = False
                self._cancel_reason = "Operation cancelled by stop request."

    def wait_until_stop_released(self, *, clear_cancel: bool = True) -> None:
        print("[STEP] Waiting for stop button release before cancellation recovery...")
        wait_stop_button_released(
            self.handles,
            self.voice_client,
            poll=self.poll,
            active_high=self.stop_active_high,
        )
        with self._state_lock:
            self._stop_latched = False
            if clear_cancel:
                self._cancel_event.clear()
                self._cancel_notice_sent = False
        print("[STEP] Stop button released; cancellation recovery may continue.")

    def _monitor_loop(self) -> None:
        prev_pressed = False
        while not self._shutdown_event.is_set():
            pressed = False
            try:
                pressed = read_stop_button_pressed(
                    self.handles,
                    active_high=self.stop_active_high,
                )
            except Exception as e:
                print(f"[WARN] Failed to poll stop button state: {e}")
                time.sleep(max(self.poll, 0.2))
                continue

            if pressed and not prev_pressed:
                self._handle_stop_pressed()
            elif not pressed and prev_pressed:
                with self._state_lock:
                    self._stop_latched = False
                print("[EVENT] Stop button released. Starting central reset.")
                handler = self._stop_released_handler
                if handler is not None:
                    threading.Thread(
                        target=handler,
                        name="stop-release-reset",
                        daemon=True,
                    ).start()

            prev_pressed = pressed
            time.sleep(self.poll)

    def _handle_stop_pressed(self) -> None:
        announce = False
        with self._state_lock:
            self._stop_latched = True
            self._cancel_reason = "Operation cancelled by physical stop button."
            self._cancel_event.set()
            if self._operation_active.is_set():
                if not self._cancel_notice_sent:
                    self._cancel_notice_sent = True
                    announce = True

        print("[EVENT] Stop button pressed. System is locked until release.")
        self._interrupt_hardware_async()
        if announce and self.voice_client is not None:
            try:
                self.voice_client.speak_openai(
                    "Stop button pressed. Cancelling the current procedure."
                )
            except Exception as e:
                print(f"[WARN] Failed to announce stop-button cancellation: {e}")

    def _interrupt_hardware_async(self) -> None:
        if not self._interrupt_lock.acquire(blocking=False):
            return
        self._interrupt_complete.clear()

        def worker() -> None:
            try:
                self._interrupt_active_hardware()
            finally:
                self._interrupt_complete.set()
                self._interrupt_lock.release()

        threading.Thread(
            target=worker,
            name="stop-hardware-interrupt",
            daemon=True,
        ).start()

    def _interrupt_active_hardware(self) -> None:
        screwdriver_running = False
        if self.screwdriver_client is not None:
            try:
                screwdriver_running = bool(self.screwdriver_client.is_running())
            except Exception as e:
                print(f"[WARN] Failed to read screwdriver state during cancellation: {e}")

        try:
            if self.screwdriver_client is not None and screwdriver_running:
                self.screwdriver_client.stop()
        except Exception as e:
            print(f"[WARN] Failed to stop screwdriver during cancellation: {e}")

        if screwdriver_running:
            print(
                "[STEP] Screwdriver cancellation requested; preserving arm control "
                "for return-to-start recovery."
            )
            return

        try:
            self.handles.arm.stop()
        except Exception as e:
            print(f"[WARN] Failed to stop arm during cancellation: {e}")

        try:
            self.handles.arm_dash.stop()
        except Exception as e:
            print(f"[WARN] Failed to stop dashboard program during cancellation: {e}")


class LLMScrewPicker:
    """Adds screw-type selection using SAM candidate regions + OpenAI vision ranking."""

    def __init__(
        self,
        cycle_manager: ScrewCycleManager,
        manual_cycle_manager: Optional[ManualScrewCycleManager] = None,
        stop_coordinator: Optional[StopIOCoordinator] = None,
    ):
        self.cycle = cycle_manager
        self.manual_cycle_manager = manual_cycle_manager
        self.stop_coordinator = stop_coordinator
        self._busy_lock = threading.Lock()

    def _notify(self, voice_client: Optional[StreamedSpeechPipelineClient],  message: str, *, role: str = 'assistant',speak: bool = True) -> None:
        if voice_client is None:
            return
        try:
            # voice_client.send_text(message, role=role, speak=speak)
            voice_client.speak_openai('looking for the screw now')
        except Exception as e:
            print(f"[WARN] send_text status update failed, falling back to speak_openai: {e}")
            try:
                voice_client.speak_openai(message)
            except Exception as e2:
                print(f"[WARN] speak_openai status update failed: {e2}")

    def _nudge_after_failed_pickup(self, attempt_idx: int, *, debug: bool = False) -> Dict[str, Any]:
        verify_cfg = self.cycle.config.get('pickup_verification', default={}) or {}
        xy_step = float(verify_cfg.get("retry_pickup_nudge_xy_m", 0.004))
        z_step = float(verify_cfg.get("retry_pickup_nudge_z_m", -0.003))
        speed = float(verify_cfg.get("retry_pickup_nudge_speed", self.cycle.config.default_speed))

        # Walk around the target slightly across retries instead of repeating the same offset.
        xy_pattern = [
            (1.0, 1.0),
            (-1.0, 1.0),
            (1.0, -1.0),
            (-1.0, -1.0),
        ]
        dir_x, dir_y = xy_pattern[max(0, attempt_idx) % len(xy_pattern)]
        dx = dir_x * xy_step
        dy = dir_y * xy_step
        dz = z_step

        pose = list(self.cycle.handles.arm.get_tcp_pose_axis_angle())
        if len(pose) < 6:
            raise RuntimeError(f"Expected 6D TCP pose, got: {pose}")

        nudged_pose = list(pose[:6])
        nudged_pose[0] += dx
        nudged_pose[1] += dy
        nudged_pose[2] += dz

        print(
            "[STEP] Nudging arm before pickup retry "
            f"(dx={dx:.4f}, dy={dy:.4f}, dz={dz:.4f}, speed={speed:.3f})"
        )
        if debug:
            print(f"[DEBUG] Pickup retry nudge pose: {nudged_pose}")
        self.cycle.handles.arm.moveL(nudged_pose, speed=speed)
        return {
            "ok": True,
            "dx_m": dx,
            "dy_m": dy,
            "dz_m": dz,
            "speed": speed,
            "pose": nudged_pose,
        }

    def installation_requested_screw_async(
        self,
        request: str,
        *,
        voice_client: Optional[StreamedSpeechPipelineClient] = None,
        debug: bool = False,
        pickup_debug_gui: bool = False,
        pickup_debug_gui_wait_ms: int = 1,
    ) -> Dict[str, Any]:
        if not self._busy_lock.acquire(blocking=False):
            return {"status": "busy", "message": "Robot is already executing a screw installation."}

        def _worker():
            try:
                result: Dict[str, Any]
                try:
                    if self.stop_coordinator is not None:
                        self.stop_coordinator.begin_operation()
                    result = self._installation_requested_screw_impl(
                        request,
                        voice_client=voice_client,
                        debug=debug,
                        pickup_debug_gui=pickup_debug_gui,
                        pickup_debug_gui_wait_ms=pickup_debug_gui_wait_ms,
                    )
                except ProcedureCancelled as e:
                    result = self._handle_cancelled_operation(
                        request,
                        voice_client=voice_client,
                        reason=str(e),
                    )
                except Exception as e:
                    if self.stop_coordinator is not None and self.stop_coordinator.is_cancelled():
                        result = self._handle_cancelled_operation(
                            request,
                            voice_client=voice_client,
                            reason=f"Operation interrupted after stop request: {e}",
                        )
                    else:
                        raise

                if result.get("status") == "ok":
                    # self._notify(voice_client, f"Completed screw installation request: {request}", speak=True)
                    if voice_client is not None:
                        voice_client.speak_openai('screw has been installed')
                elif result.get("status") not in ("cancelled",):
                    reason = str(result.get("reason", "unknown error"))
                    self._notify(voice_client, f"Screw installation failed. Reason: {reason}", speak=True)
                    # voice_client.speak_openai('screw has failed install')
            except Exception as e:
                print(f"[ERROR] pickup_requested_screw worker crashed: {e}")
                # self._notify(voice_client, f"Screw installation crashed. Reason: {e}", speak=True)
                if voice_client is not None:
                    voice_client.speak_openai('error occured during screw installation')
            finally:
                if self.stop_coordinator is not None:
                    self.stop_coordinator.end_operation()
                self._busy_lock.release()
                print("[READY] Screw installation worker is ready for the next command.")

        threading.Thread(target=_worker, name="screw-install-worker", daemon=True).start()
        return {"status": "ok", "action": "started", "request": request}

    def pickup_requested_screw(
        self,
        request: str,
        *,
        voice_client: Optional[StreamedSpeechPipelineClient] = None,
        debug: bool = False,
        pickup_debug_gui: bool = False,
        pickup_debug_gui_wait_ms: int = 1,
    ) -> Dict[str, Any]:
        if not self._busy_lock.acquire(blocking=False):
            return {"status": "busy", "message": "Robot is already executing a screw installation."}

        try:
            return self._pickup_requested_screw_impl(
                request,
                voice_client=voice_client,
                debug=debug,
                pickup_debug_gui=pickup_debug_gui,
                pickup_debug_gui_wait_ms=pickup_debug_gui_wait_ms,
            )
        finally:
            self._busy_lock.release()

    def _installation_requested_screw_impl(
        self,
        request: str,
        *,
        voice_client: Optional[StreamedSpeechPipelineClient] = None,
        debug: bool = False,
        pickup_debug_gui: bool = False,
        pickup_debug_gui_wait_ms: int = 1,
    ) -> Dict[str, Any]:
        self._raise_if_cancelled()

        offsets = self.cycle.config.offsets
        observation_offset = offsets["observation_offset"]
        pickup_clearance = offsets["pickup_clearance"]
        ok = False
        max_detection_attempts = 3

        #Use Fixed Camera to detect screw and move above it before pickup,
        # for attempt in range(max_detection_attempts):
        #     ok = self.cycle.detect_and_move_to_screw(
        #         debug=debug,
        #         camera="fixed",
        #         z_offset=observation_offset,
        #         target_mode="centre",
        #         y_offset=0.0,
        #         x_offset=0.15,
        #     )
        #     if ok:
        #         break

        # if not ok:
        #     return {
        #         "status": "failed",
        #         "reason": f"Failed to detect screw with fixed camera after {max_detection_attempts} attempts.",
        #     }
        

        verify_cfg = self.cycle.config.get('pickup_verification', default={}) or {}
        retry_pickup_attempts = max(0, int(verify_cfg.get("retry_pickup_attempts", 1)))
        enable_local_refine = bool(verify_cfg.get("enable_local_refine", True))
        max_pickup_attempts = 1 + retry_pickup_attempts
        pickup_attempt_history = []
        pickup_verified = False
        pickup_verify_info: Dict[str, Any] = {}
        info: Dict[str, Any] = {}
        retry_anchor_target_base: Optional[Any] = None
        retry_ready_from_local_refine = False

        for pickup_attempt_idx in range(max_pickup_attempts):
            attempt_no = pickup_attempt_idx + 1
            if attempt_no == 1:
                print("[STEP] Moving to pickup position...")
                self.cycle.move_to_screw_pickup_position()
                self._raise_if_cancelled()
            else:
                print(f"[WARN] Pickup was not verified. Retrying pickup ({attempt_no}/{max_pickup_attempts})...")
                if voice_client is not None:
                    voice_client.speak_openai("Pickup was not confirmed. Trying the pickup again.")
                if retry_ready_from_local_refine:
                    print("[STEP] Reusing post-nudge local refinement for the next pickup attempt...")
                else:
                    print("[STEP] Re-detecting screw from nudged pose...")

            if retry_ready_from_local_refine:
                ok = True
                retry_ready_from_local_refine = False
            else:
                ok, info = self.cycle.llm_detect_and_move_to_screw(
                    request,
                    camera="arm",
                    llm_target_mode="head",
                    z_offset=pickup_clearance,
                    detection_mode=self.cycle.config.get(
                        "screwpickup_detection", "mode", default="head_and_box"
                    ),
                    llm_birdseye=True,
                    debug=debug,
                )
                self._raise_if_cancelled()
                if not ok:
                    pickup_attempt_history.append(
                        {
                            "attempt": attempt_no,
                            "detect_ok": False,
                            "detect_info": info,
                        }
                    )
                    if attempt_no >= max_pickup_attempts:
                        return {
                            "status": "failed",
                            "reason": str(info.get("reason", "Failed to detect the requested screw for pickup.")),
                            "coarse": info,
                            "pickup_attempts": pickup_attempt_history,
                        }
                    continue

                selected_target_base = info.get("selected_target_base")
                if enable_local_refine and selected_target_base:
                    print("[STEP] Refining pickup pose using local screw-head detection near the projected pickup point...")
                    refine_ok, refine_info = self.cycle.refine_pickup_target_locally(
                        target_base=selected_target_base,
                        camera="arm",
                        z_offset=pickup_clearance,
                        debug=debug,
                        debug_gui=debug,
                    )
                    self._raise_if_cancelled()
                    info["pickup_refine"] = refine_info
                    if not refine_ok:
                        pickup_attempt_history.append(
                            {
                                "attempt": attempt_no,
                                "detect_ok": False,
                                "detect_info": info,
                                "pickup_refine_failed": True,
                            }
                        )
                        if attempt_no >= max_pickup_attempts:
                            return {
                                "status": "failed",
                                "reason": str(refine_info.get("reason", "Failed to refine the requested screw position for pickup.")),
                                "coarse": info,
                                "pickup_attempts": pickup_attempt_history,
                            }
                        continue
                    retry_anchor_target_base = refine_info.get("refined_target_base") or selected_target_base
                elif selected_target_base:
                    retry_anchor_target_base = selected_target_base
                    info["pickup_refine"] = {
                        "skipped": True,
                        "reason": "Local pickup refinement is disabled by pickup_verification.enable_local_refine.",
                    }

            print(f"[STEP] Running screw pickup program (attempt {attempt_no}/{max_pickup_attempts})...")
            self.cycle.run_pickup_program()
            self._raise_if_cancelled()

            time.sleep(2)
            pickup_verified, pickup_verify_info = self.cycle.verify_pickup_arm_camera(
                debug=debug,
                debug_gui=pickup_debug_gui,
                debug_gui_wait_ms=pickup_debug_gui_wait_ms,
            )
            pickup_attempt_history.append(
                {
                    "attempt": attempt_no,
                    "detect_ok": True,
                    "detect_info": info,
                    "pickup_verified": bool(pickup_verified),
                    "pickup_verify_info": pickup_verify_info,
                }
            )
            if pickup_verified:
                break
            if attempt_no < max_pickup_attempts:
                try:
                    print("[STEP] Returning to pickup position before retry nudge...")
                    self.cycle.move_to_screw_pickup_position()
                    self._raise_if_cancelled()
                    nudge_info = self._nudge_after_failed_pickup(pickup_attempt_idx, debug=debug)
                    pickup_attempt_history[-1]["retry_nudge"] = nudge_info
                except Exception as e:
                    pickup_attempt_history[-1]["retry_nudge"] = {
                        "ok": False,
                        "error": str(e),
                    }
                    print(f"[WARN] Failed to nudge arm before pickup retry: {e}")
                    continue

                try:
                    if enable_local_refine:
                        if retry_anchor_target_base is not None:
                            print("[STEP] Refining pickup pose again after nudge...")
                            retry_refine_ok, retry_refine_info = self.cycle.refine_pickup_target_locally(
                                target_base=retry_anchor_target_base,
                                camera="arm",
                                z_offset=pickup_clearance,
                                debug=debug,
                                debug_gui=debug,
                            )
                            self._raise_if_cancelled()
                            pickup_attempt_history[-1]["retry_refine"] = {
                                "ok": bool(retry_refine_ok),
                                **(retry_refine_info or {}),
                            }
                            if retry_refine_ok and isinstance(retry_refine_info, dict):
                                retry_ready_from_local_refine = True
                                retry_anchor_target_base = (
                                    retry_refine_info.get("refined_target_base") or retry_anchor_target_base
                                )
                                if isinstance(info, dict):
                                    info["pickup_refine_after_nudge"] = retry_refine_info
                            else:
                                retry_ready_from_local_refine = False
                        else:
                            pickup_attempt_history[-1]["retry_refine"] = {
                                "ok": False,
                                "reason": "No prior pickup target was available for post-nudge local refinement.",
                            }
                    else:
                        pickup_attempt_history[-1]["retry_refine"] = {
                            "ok": False,
                            "skipped": True,
                            "reason": "Local pickup refinement is disabled by pickup_verification.enable_local_refine.",
                        }
                except Exception as e:
                    pickup_attempt_history[-1]["retry_refine"] = {
                        "ok": False,
                        "error": str(e),
                    }
                    print(f"[WARN] Failed to refine pickup pose after nudge: {e}")

        if not pickup_verified:
            if voice_client is not None:
                voice_client.speak_openai("I could not confirm the screw pickup after retrying.")
            return {
                "status": "failed",
                "reason": f"Screw pickup was not verified for: {request}",
                "coarse": info,
                "pickup_verification": pickup_verify_info,
                "pickup_attempts": pickup_attempt_history,
            }

        # self._notify(voice_client, "Pickup program completed. Looking for installation marker.", speak=True)
        print("[STEP] Moving to Target Mark after pickup...")
        self._raise_if_cancelled()
        self.cycle.move_to_screw_target_position()
        self._raise_if_cancelled()
        if voice_client is not None:
            voice_client.speak_openai('show me where to install the screw')
        target_mark_fixed_acquired = False
        manual_override_hold_s = float(
            self.cycle.config.get(
                'hardware', 'timing', 'manual_override_hold_s', default=3.0
            )
        )
        ok_target_mark, target_mark_status = self.cycle.move_to_target_mark_with_status(
            debug=debug,
            use_fixed_camera=True,
        )
        target_mark_fixed_acquired = bool(target_mark_status.get("fixed_camera_success", False))
        while not ok_target_mark:
            self._raise_if_cancelled()
            trigger_event = check_tool_button_event(
                self.cycle.handles,
                listen_window_s=0.6,
                long_press_sec=manual_override_hold_s,
                poll=0.05,
            )
            if trigger_event == "long":
                if self.manual_cycle_manager is None:
                    if voice_client is not None:
                        voice_client.speak_openai("Manual screw mode is not available.")
                    return {
                        "status": "failed",
                        "reason": "Long trigger hold requested manual screw mode, but no manual cycle manager is configured.",
                        "coarse": info,
                    }

                if voice_client is not None:
                    voice_client.speak_openai("Manual mode enabled. Guide the screw to the desired location and press the trigger again when ready to install.")
                if not self.manual_cycle_manager.manual_position(
                    debug=debug,
                    trigger_hold_sec=manual_override_hold_s,
                ):
                    self._raise_if_cancelled()
                    if voice_client is not None:
                        voice_client.speak_openai('manual_position failed ')
                    trigger_event = "short"
                else:
                    ok_target_mark = True
                    target_mark_status = {
                        "manual_override": True,
                        "green_target_skipped": True,
                        "admittance_completed": True,
                    }
                    break

            if trigger_event == "short":
                if voice_client is not None:
                    voice_client.speak_openai('screw installation cancelled')
                self.cycle.move_to_start_position()
                start_move_info = {"selected_prompt": "start_joint", "ok": True}
                return {
                    "status": "cancelled",
                    "reason": "Operation cancelled by short robot trigger press while searching for the Target Mark.",
                    "coarse": info,
                    "return_to_start": start_move_info,
                }
            ok_target_mark, target_mark_status = self.cycle.move_to_target_mark_with_status(
                debug=debug,
                use_fixed_camera=not target_mark_fixed_acquired,
            )
            self._raise_if_cancelled()
            target_mark_fixed_acquired = target_mark_fixed_acquired or bool(
                target_mark_status.get("fixed_camera_success", False)
            )
            time.sleep(0.5)


        target_mark_info = {
            "selected_prompt": "target_mark",
            "fixed_camera_acquired": target_mark_fixed_acquired,
            "status": target_mark_status,
        }
        if not ok_target_mark:
            # self._notify(voice_client, "Failed to find installation mark.", speak=True)
            return {
                "status": "failed",
                "reason": "Picked up screw, but failed to move to the Target Mark.",
                "coarse": info,
                "target_mark": target_mark_info,
            }

        # if debug:
        #     input("Reached Target Mark. Press Enter to continue to screwdriver step...")
        
        screwdriver_info = {"enabled": bool(self.cycle.screwdriver_client)}
        start_move_info = {"selected_prompt": "start_joint", "ok": False}
        if self.cycle.screwdriver_client is not None:
            # self._notify(voice_client, "Reached installation mark. Attempting screwdriver operation now.", speak=True)
            print("[STEP] Running screwdriver after reaching Target Mark...")
            ok_screwdriver = self.cycle.run_screwdriver()
            self._raise_if_cancelled()
            screwdriver_info["ran"] = True
            screwdriver_info["ok"] = bool(ok_screwdriver)
            if not ok_screwdriver:
                # self._notify(voice_client, "Screwdriver operation failed.", speak=True)
                return {
                    "status": "failed",
                    "reason": "Reached Target Mark, but screwdriver run failed.",
                    "coarse": info,
                    "target_mark": target_mark_info,
                    "screwdriver": screwdriver_info,
                }
            
            start_move_info = {"selected_prompt": "start_joint", "ok": True}
            # self._notify(voice_client, "Screwdriver operation completed successfully.", speak=True)
        else:
            # self._notify(
            #     voice_client,
            #     "Reached installation mark. Screwdriver operation is not enabled, so skipping that step.",
            #     speak=True,
            # )
            #wait for entre key
            try:
                self.cycle.move_to_start_position()
                self._raise_if_cancelled()
                start_move_info["ok"] = True
            except Exception as e:
                return {
                    "status": "failed",
                    "reason": f"Install completed, but failed to move back to start location: {e}",
                    "coarse": info,
                    "target_mark": target_mark_info,
                    "screwdriver": screwdriver_info,
                "return_to_start": start_move_info,
            }
            screwdriver_info["ran"] = False



        return {
            "status": "ok",
            "message": f"Picked up requested screw: {request}",
            "coarse": info,
            "target_mark": target_mark_info,
            "screwdriver": screwdriver_info,
            "return_to_start": start_move_info,
        }

    def _raise_if_cancelled(self) -> None:
        if self.stop_coordinator is not None:
            self.stop_coordinator.raise_if_cancelled()

    def _handle_cancelled_operation(
        self,
        request: str,
        *,
        voice_client: Optional[StreamedSpeechPipelineClient] = None,
        reason: str,
    ) -> Dict[str, Any]:
        print(f"[STEP] Handling cancellation for request {request!r}: {reason}")
        if (
            self.stop_coordinator is not None
            and self.stop_coordinator.central_reset_enabled()
        ):
            print(
                "[STEP] Cancellation cleanup deferred to the central reset "
                "controller."
            )
            return {
                "status": "cancelled",
                "reason": reason,
                "return_to_start": {"selected_prompt": "start_joint", "deferred": True},
            }

        if voice_client is not None:
            try:
                voice_client.speak_openai(
                    "Cancelling the procedure. Release the stop button so I can return to the start position."
                )
            except Exception as e:
                print(f"[WARN] Failed to announce cancellation cleanup: {e}")

        if self.stop_coordinator is not None:
            try:
                self.stop_coordinator.wait_until_stop_released()
            except Exception as e:
                print(f"[WARN] Failed while waiting for stop release: {e}")

        start_move_info = {"selected_prompt": "start_joint", "ok": False}
        try:
            self.cycle.confirm_drill_tcp(
                context="screwdriver cancellation recovery"
            )
            print("[STEP] Returning to start_joint after screwdriver cancellation.")
            self.cycle.move_to_start_position()
            start_move_info["ok"] = True
            print("[STEP] Cancellation recovery reached start_joint.")
        except Exception as e:
            return {
                "status": "cancelled",
                "reason": f"{reason} Return-to-start failed: {e}",
                "return_to_start": start_move_info,
            }

        return {
            "status": "cancelled",
            "reason": reason,
            "return_to_start": start_move_info,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--debug-step",
        choices=("pickup", "target", "screw"),
        help="Run only one debug stage: pickup, target-mark detection, or screwdriver operation.",
    )
    parser.add_argument(
        "--debug-request",
        default="a screw",
        help="Screw description used by --debug-step pickup.",
    )
    parser.add_argument("--pickup-debug-gui", action="store_true")
    parser.add_argument("--pickup-debug-gui-wait-ms", type=int, default=1)
    parser.add_argument("--enable_screwdriver", action="store_true")
    return parser.parse_args()


def main():
    global ACTIVE_SCREWDRIVER_CLIENT, ACTIVE_VOICE_CLIENT

    args = parse_args()

    hardware_initializer = HardwareInitializer(
        camera_arm_name="camera_drill",
        camera_fixed_name="camera_fixed",
        ignore_gripper=True,
        tool_name="tcp_drill",
        stop_dashboard_program_on_init=False,
        debug=args.debug,
    )
    hw = hardware_initializer.initialize()

    detector = Sam3Detector()
    screwdriver_client = ScrewdriverClient() if args.enable_screwdriver else None
    ACTIVE_SCREWDRIVER_CLIENT = screwdriver_client

    voice_client = StreamedSpeechPipelineClient.load()
    ACTIVE_VOICE_CLIENT = voice_client
    stop_coordinator = StopIOCoordinator(
        hw,
        voice_client=voice_client,
        screwdriver_client=screwdriver_client,
    )
    stop_coordinator.start()

    cycle_manager = ScrewCycleManager(
        hw,
        detector,
        screwdriver_client=screwdriver_client,
        voice_client=voice_client,
        cancel_check=stop_coordinator.is_cancelled,
    )
    manual_cycle_manager = ManualScrewCycleManager(
        hw,
        screwdriver_client,
        voice_client,
        cancel_check=stop_coordinator.is_cancelled,
    )
    llm_picker = LLMScrewPicker(
        cycle_manager,
        manual_cycle_manager=manual_cycle_manager,
        stop_coordinator=stop_coordinator,
    )

    def run_debug_step() -> Dict[str, Any]:
        """Run one hardware stage without continuing into the full installation workflow."""
        step = str(args.debug_step)
        stop_coordinator.begin_operation()
        try:
            if step == "pickup":
                print(f"[DEBUG] Running pickup-only stage for request: {args.debug_request!r}")
                cycle_manager.move_to_screw_pickup_position()
                stop_coordinator.raise_if_cancelled()
                pickup_clearance = cycle_manager.config.offsets["pickup_clearance"]
                ok, info = cycle_manager.llm_detect_and_move_to_screw(
                    request=args.debug_request,
                    camera="arm",
                    llm_target_mode="head",
                    z_offset=pickup_clearance,
                    detection_mode=cycle_manager.config.get(
                        "screwpickup_detection", "mode", default="head_and_box"
                    ),
                    llm_birdseye=True,
                    debug=True,
                )
                if not ok:
                    return {"ok": False, "stage": step, "reason": info.get("reason", "Screw target was not reachable."), "details": info}

                selected_target_base = info.get("selected_target_base")
                verify_cfg = cycle_manager.config.get("pickup_verification", default={}) or {}
                if bool(verify_cfg.get("enable_local_refine", True)) and selected_target_base:
                    ok, refine_info = cycle_manager.refine_pickup_target_locally(
                        target_base=selected_target_base,
                        camera="arm",
                        z_offset=pickup_clearance,
                        debug=True,
                        debug_gui=True,
                    )
                    if not ok:
                        return {"ok": False, "stage": step, "reason": refine_info.get("reason", "Local pickup refinement failed."), "details": refine_info}

                print("[DEBUG] Running pickup program only; target-mark and screwing stages are skipped.")
                cycle_manager.run_pickup_program()
                stop_coordinator.raise_if_cancelled()
                return {"ok": True, "stage": step}

            if step == "target":
                print("[DEBUG] Repeatedly checking for the green target mark. Press stop to cancel.")
                fixed_camera_acquired = False
                while True:
                    stop_coordinator.raise_if_cancelled()
                    ok, status = cycle_manager.move_to_target_mark_with_status(
                        debug=True,
                        use_fixed_camera=not fixed_camera_acquired,
                    )
                    fixed_camera_acquired = fixed_camera_acquired or bool(
                        status.get("fixed_camera_success", False)
                    )
                    if ok:
                        return {"ok": True, "stage": step, "details": status}
                    print("[DEBUG] Green target mark not found yet; retrying in 0.5 seconds.")
                    time.sleep(0.5)

            if cycle_manager.screwdriver_client is None:
                return {
                    "ok": False,
                    "stage": step,
                    "reason": "The screw stage requires --enable_screwdriver.",
                }
            print("[DEBUG] Running screwdriver operation only.")
            ok = cycle_manager.run_screwdriver()
            stop_coordinator.raise_if_cancelled()
            return {"ok": bool(ok), "stage": step}
        except ProcedureCancelled as e:
            return {"ok": False, "stage": step, "reason": str(e)}
        finally:
            stop_coordinator.end_operation()

    if args.debug_step:
        print(f"[INFO] Focused debug step selected: {args.debug_step}")
        wait_stop_button_released(hw, voice_client)
        try:
            result = run_debug_step()
            print(f"[INFO] Debug step result: {result}")
        finally:
            stop_coordinator.shutdown()
            try:
                voice_client.stop()
            except Exception as e:
                print(f"[WARN] Failed to stop voice client: {e}")
            try:
                if screwdriver_client is not None:
                    screwdriver_client.stop()
            except Exception as e:
                print(f"[WARN] Failed to stop screwdriver client: {e}")
            hardware_initializer.shutdown()
        return

    voice_client.instructions = (
        "You are Quendabot, a screw-installation robot.\n"
        "Only handle screw-installation requests.\n"
        "The only available tool is `screw_installation_request`.\n"
        "When the user asks to install, pick up, or use a screw, call `screw_installation_request` \n"
        "After the tool runs, reply briefly in plain English.\n"
        "If you hear the word skirt, it means screw. \n"
    )
    voice_client.tools = [
        {
            "type": "function",
            "function": {
                "name": "screw_installation_request",
                "description": "Install or pick up the requested screw type using the robot.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "requirement": {
                            "type": "string",
                            "description": "The screw request.",
                        }
                    },
                    "required": ["requirement"],
                },
            },
        },
    ]
    voice_client.tool_choice = "auto"

    def screw_installation_requested_handler(
        requirement: str = "",
    ):
        screw_request = requirement
        if not screw_request:
            return {
                "status": "failed",
                "reason": "Missing screw request. Expected `screw_type` or `requirement`.",
                "send_to_model": True,
            }
        print(
            f"requirement={requirement!r} resolved={screw_request!r}"
        )
        result = llm_picker.installation_requested_screw_async(
            screw_request,
            voice_client=voice_client,
            debug=args.debug,
            pickup_debug_gui=args.pickup_debug_gui,
            pickup_debug_gui_wait_ms=args.pickup_debug_gui_wait_ms,
        )
        return {**result, "send_to_model": True}

    voice_client.register_tool_handler("screw_installation_request", screw_installation_requested_handler)
    # voice_client.register_tool_handler("stop_action", stop_action_handler)
    voice_client.on_error = lambda e: print("[LLM] error:", e)
    voice_client.on_text_completed = lambda txt: print(f"[LLM] {txt}")
    voice_client.start()
    voice_client.start_background()

    print("[STEP] Checking stop button before starting...")
    wait_stop_button_released(hw, voice_client)

    time.sleep(1.0)
    try:
        # Use direct TTS cue here instead of a model turn to avoid back-to-back
        # realtime responses immediately after startup.
        voice_client.speak_openai("Quendabot online. Screwing mode enabled.")
    except Exception as e:
        print(f"[WARN] Failed to play startup cue: {e}")
    time.sleep(0.3)

    print("[INFO] LLM screw picker running. Speak or type a screw request.")
    print("[INFO] Commands: /wake, /listen, /say <text>, /quit")

    def input_loop():
        try:
            while True:
                try:
                    line = input("> ")
                except EOFError:
                    time.sleep(0.05)
                    continue

                msg = line.strip()
                if not msg:
                    continue

                if msg.lower() in ("/q", "/quit", "quit", "exit"):
                    print("[INFO] Quit requested from keyboard.")
                    break

                if msg.startswith("/listen"):
                    try:
                        voice_client.request_listen(play_cue=True)
                    except Exception as e:
                        print(f"[/listen ERROR] {e}")
                    continue

                if msg.startswith("/say "):
                    msg = msg[5:].strip()
                    if not msg:
                        print("[INFO] Usage: /say <text>")
                        continue

                try:
                    voice_client.respond_once(msg)
                except Exception as e:
                    print(f"[text input ERROR] {e}")
        except KeyboardInterrupt:
            pass

    keyboard_thread = threading.Thread(target=input_loop, name="screw-llm-keyboard", daemon=True)
    keyboard_thread.start()

    try:
        while keyboard_thread.is_alive():
            keyboard_thread.join(timeout=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        print("[INFO] Shutting down screw picker demo...")
        try:
            stop_coordinator.shutdown()
        except Exception as e:
            print(f"[WARN] Failed to stop stop-io monitor: {e}")
        try:
            voice_client.stop()
        except Exception as e:
            print(f"[WARN] Failed to stop voice client: {e}")
        try:
            if screwdriver_client is not None:
                screwdriver_client.stop()
        except Exception as e:
            print(f"[WARN] Failed to stop screwdriver client: {e}")
        try:
            hardware_initializer.shutdown()
        except Exception as e:
            print(f"[WARN] Failed to shutdown hardware cleanly: {e}")


if __name__ == "__main__":
    main()
