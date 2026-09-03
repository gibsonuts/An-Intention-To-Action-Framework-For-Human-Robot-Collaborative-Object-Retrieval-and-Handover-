import argparse
import cv2
import re
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from detectors.sam3_object_detection import (
    Sam3Detector,
    draw_mask_debug,
    mask_centroid,
)
from demo_screwing_llm import (
    LLMScrewPicker,
    ProcedureCancelled,
    StopIOCoordinator,
    check_tool_button_event,
    wait_stop_button_released,
)
from hardware.hardware_init import HardwareInitializer
from hardware.screwdriver_client import ScrewdriverClient
from llm.openai_stt_llm_tts_client import StreamedSpeechPipelineClient
from llm.vlm_watcher import VisionPickUpWatcher
from qbot.qbot_cycles import Config, ManualScrewCycleManager, ScrewCycleManager


ACTIVE_SCREWDRIVER_CLIENT = None
ACTIVE_VOICE_CLIENT = None
ACTIVE_HARDWARE_INITIALIZER = None
ACTIVE_VLM_WATCHER = None
ACTIVE_MANUAL_MODE_CONTROLLER = None


def handle_sigint(signum, frame):
    print("\n[CTRL-C] Cleaning up...")
    global ACTIVE_SCREWDRIVER_CLIENT, ACTIVE_VOICE_CLIENT, ACTIVE_HARDWARE_INITIALIZER
    global ACTIVE_VLM_WATCHER
    global ACTIVE_MANUAL_MODE_CONTROLLER

    if ACTIVE_MANUAL_MODE_CONTROLLER is not None:
        try:
            ACTIVE_MANUAL_MODE_CONTROLLER.stop(wait=True)
        except Exception as e:
            print(f"[WARN] Failed to stop manual mode: {e}")

    if ACTIVE_VLM_WATCHER is not None:
        try:
            ACTIVE_VLM_WATCHER.stop()
        except Exception as e:
            print(f"[WARN] Failed to stop VLM watcher: {e}")

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

    if ACTIVE_HARDWARE_INITIALIZER is not None:
        try:
            ACTIVE_HARDWARE_INITIALIZER.shutdown()
        except Exception as e:
            print(f"[WARN] Failed to shutdown hardware: {e}")

    sys.exit(0)


signal.signal(signal.SIGINT, handle_sigint)


class GripperScrewCycleManager(ScrewCycleManager):
    """Screw cycle manager with a combined gripper pickup/drop-off program."""

    def prepare_gripper_for_load(self) -> None:
        if self.handles.gripper is None:
            print("[WARN] No Robotiq gripper handle is available; continuing with URP-controlled pickup only.")
            return

        try:
            print("[STEP] Preparing Robotiq gripper for object pickup...")
            self.handles.gripper.activate()
            self.handles.gripper.open(block=True)
        except Exception as e:
            print(f"[WARN] Failed to prepare Robotiq gripper before load: {e}")

    def move_to_grasp_position(self) -> None:
        joints = self.config.get("grasp_joint")
        if joints is None:
            raise RuntimeError("Missing `grasp_joint` in qbot/config/cycles.yaml")
        print("[STEP] Moving to gripper grasp position...", joints)
        self.handles.arm.moveJ(joints)

    def _return_to_start_after_load_failure(self, context: str) -> Optional[str]:
        """Attempt recovery without replacing the primary pickup failure."""
        try:
            self.move_to_start_position()
            return None
        except Exception as e:
            error = str(e)
            print(
                f"[WARN] Failed to return to start_joint after {context}: {error}"
            )
            return error

    def _select_mask_closest_to_image_center(
        self,
        masks: List[Dict[str, Any]],
        image_bgr,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Select exactly one detection by mask-centroid distance to image centre."""
        if not masks:
            raise ValueError("Cannot select a centre-most object from an empty mask list")
        height, width = image_bgr.shape[:2]
        center_x = (float(width) - 1.0) / 2.0
        center_y = (float(height) - 1.0) / 2.0
        ranked = []
        for index, mask in enumerate(masks):
            try:
                centroid_x, centroid_y = mask_centroid(mask["segmentation"])
            except Exception:
                x, y, box_width, box_height = mask["bbox"]
                centroid_x = float(x) + float(box_width) / 2.0
                centroid_y = float(y) + float(box_height) / 2.0
            distance_px = (
                (centroid_x - center_x) ** 2 + (centroid_y - center_y) ** 2
            ) ** 0.5
            ranked.append((distance_px, index, centroid_x, centroid_y, mask))

        distance_px, index, centroid_x, centroid_y, selected = min(
            ranked,
            key=lambda item: (item[0], item[1]),
        )
        selection = {
            "detection_index": index,
            "detection_number": index + 1,
            "detection_count": len(masks),
            "centroid_px": [centroid_x, centroid_y],
            "image_center_px": [center_x, center_y],
            "distance_to_center_px": distance_px,
        }
        print(
            f"[PICKUP SELECT] Selected detection {index + 1}/{len(masks)} closest "
            f"to image centre: centroid=({centroid_x:.1f}, {centroid_y:.1f}), "
            f"centre=({center_x:.1f}, {center_y:.1f}), "
            f"distance={distance_px:.1f}px. Only this detection will be planned."
        )
        return selected, selection

    def _configured_pickup_objects(self) -> List[Dict[str, Any]]:
        raw_objects = self.config.get("gripper_pickup_objects", default=[]) or []
        defaults = self.config.get("gripper_pickup_defaults", default={}) or {}
        if not isinstance(defaults, dict):
            raise ValueError("`gripper_pickup_defaults` must be a mapping in cycles.yaml")
        defaults = {
            key: value
            for key, value in defaults.items()
            if key != "allow_unconfigured"
        }
        objects: List[Dict[str, Any]] = []
        for entry in raw_objects:
            if isinstance(entry, str):
                objects.append(
                    {**defaults, "name": entry, "prompt": entry, "aliases": []}
                )
                continue
            if not isinstance(entry, dict):
                print(f"[WARN] Ignoring invalid gripper pickup object: {entry!r}")
                continue
            name = str(entry.get("name", entry.get("prompt", ""))).strip()
            prompt = str(entry.get("prompt", name)).strip()
            if not name or not prompt:
                print(f"[WARN] Ignoring gripper pickup object without name/prompt: {entry!r}")
                continue
            item = {**defaults, **entry}
            item.update({"name": name, "prompt": prompt})
            aliases = item.get("aliases", []) or []
            if isinstance(aliases, str):
                aliases = [aliases]
            item["aliases"] = [str(alias).strip() for alias in aliases]
            objects.append(item)
        return objects

    def _resolve_pickup_object(self, request: str) -> Dict[str, Any]:
        objects = self._configured_pickup_objects()
        requested_text = str(request or "").strip()
        requested = requested_text.lower()
        for item in objects:
            terms = [item["name"], item["prompt"], *item.get("aliases", [])]
            normalized = [str(term).strip().lower() for term in terms if str(term).strip()]
            if any(term == requested or term in requested for term in normalized):
                return item
        if len(objects) == 1 and not requested:
            return objects[0]
        if not requested:
            raise ValueError("The gripper pickup request must include an object name.")

        defaults = self.config.get("gripper_pickup_defaults", default={}) or {}
        if not isinstance(defaults, dict):
            raise ValueError("`gripper_pickup_defaults` must be a mapping in cycles.yaml")
        if not bool(defaults.get("allow_unconfigured", True)):
            available = ", ".join(item["name"] for item in objects)
            raise ValueError(
                f"Object {request!r} is not configured for gripper pickup. "
                f"Available objects: {available}."
            )

        # Tool arguments should contain the object name, but also tolerate a full
        # retrieval phrase so it does not become a poor SAM text prompt.
        object_name = re.sub(
            r"^(?:please\s+)?(?:grab|get|fetch|bring|pick\s+up|pickup|hand\s+me)"
            r"(?:\s+me)?\s+",
            "",
            requested_text,
            flags=re.IGNORECASE,
        ).strip(" .!?\t\r\n")
        object_name = re.sub(
            r"^(?:a|an|the)\s+", "", object_name, flags=re.IGNORECASE
        ).strip()
        object_name = object_name or requested_text
        generic = {
            key: value
            for key, value in defaults.items()
            if key != "allow_unconfigured"
        }
        generic.update({"name": object_name, "prompt": object_name, "aliases": []})
        print(
            f"[INFO] Using generic gripper pickup settings for unconfigured "
            f"object {object_name!r}."
        )
        return generic

    def run_gripper_pickup_program(self) -> None:
        timing = self.config.get("hardware", "timing", default={}) or {}
        program_name = self.config.get(
            "programs",
            "gripper_pickup",
            default="gripper_pickup.urp",
        )

        print(f"[STEP] Running gripper pickup URP program: {program_name}")
        self.handles.arm_dash.connect()
        time.sleep(timing.get("dashboard_connect_delay", 0.1))
        load_response = self.handles.arm_dash.load_urp(program_name)
        print(f"[DASHBOARD] load_urp response: {load_response}")
        load_to_play_delay = max(0.0, float(timing.get("urp_load_delay", 1.0)))
        print(
            f"[STEP] Waiting {load_to_play_delay:.1f} seconds for the gripper "
            "URP program to finish loading before play."
        )
        time.sleep(load_to_play_delay)
        print("[STEP] Calling dashboard play now.")
        play_response = self.handles.arm_dash.play()
        print(f"[DASHBOARD] play response: {play_response}")
        self.handles.arm_dash.wait_for_program(
            start_timeout=float(timing.get("urp_start_timeout", 10.0)),
            finish_timeout=float(timing.get("urp_finish_timeout", 300.0)),
            poll_interval=float(timing.get("urp_poll_interval", 0.25)),
        )
        self.confirm_drill_tcp(context=f"completed {program_name}")
        time.sleep(timing.get("post_pickup_wait", 3.0))
        if self.cancel_check is not None and self.cancel_check():
            raise ProcedureCancelled("Object load cancelled before returning to start_joint.")
        print("[STEP] Combined gripper pickup/drop-off program finished; returning to start_joint.")
        self.move_to_start_position()

    def load_object_with_gripper(
        self,
        request: str,
        *,
        debug: bool = False,
        run_program: bool = True,
    ) -> Dict[str, Any]:
        """
        Detect a requested object, move tcp_gripper above it, and optionally pick it up.
        """
        try:
            object_cfg = self._resolve_pickup_object(request)
        except (RuntimeError, ValueError) as e:
            return {"status": "failed", "stage": "load", "reason": str(e)}

        name = object_cfg["name"]
        prompt = object_cfg["prompt"]
        confidence = float(object_cfg.get("confidence", 0.35))
        clearance = float(
            object_cfg.get(
                "pickup_clearance",
                self.config.get("motion", "offsets", "pickup_clearance", default=0.15),
            )
        )
        orientation_align = str(object_cfg.get("orientation_align", "long"))
        print(
            f"[STEP] Loading object {name!r} using SAM prompt={prompt!r}, "
            f"confidence={confidence:.2f}."
        )
        if self.cancel_check is not None and self.cancel_check():
            raise ProcedureCancelled("Object load cancelled before moving to grasp_joint.")
        self.move_to_grasp_position()
        self.prepare_gripper_for_load()

        if self.cancel_check is not None and self.cancel_check():
            raise ProcedureCancelled("Object load cancelled before detection.")
        self.confirm_drill_tcp(context=f"{name} gripper detection and planning")
        camera = str(object_cfg.get("camera", "arm")).strip().lower()
        depth, color, intr, T_cam = self.camera_helper.get_rgbd_and_intrinsics(camera)
        T_base_active_tcp = self.handles.arm.get_T_base_tcp()
        masks = self.detector.segment(
            color_bgr=color,
            text_prompt=prompt,
            confidence_threshold=confidence,
            category=prompt,
            orientation_align=orientation_align,
        )
        print(f"[STEP] Detected {len(masks)} candidate(s) for {name!r}.")
        selected_mask = None
        center_selection = None
        if masks:
            selected_mask, center_selection = self._select_mask_closest_to_image_center(
                masks,
                color,
            )
        if debug:
            try:
                output_dir = self.config.get("debug", "output_dir", default="data/image_samples")
                output_path = f"{output_dir}/sam3_gripper_object.png"
                vis = draw_mask_debug(color, masks, output_path=output_path)
                if center_selection is not None and selected_mask is not None:
                    image_center = tuple(
                        int(round(value))
                        for value in center_selection["image_center_px"]
                    )
                    selected_centroid = tuple(
                        int(round(value))
                        for value in center_selection["centroid_px"]
                    )
                    box_x, box_y, box_width, box_height = (
                        int(round(value)) for value in selected_mask["bbox"]
                    )
                    box_end = (box_x + box_width, box_y + box_height)
                    cv2.rectangle(
                        vis,
                        (box_x, box_y),
                        box_end,
                        (0, 255, 0),
                        5,
                    )
                    cv2.circle(vis, image_center, 9, (0, 255, 0), 3)
                    cv2.circle(vis, selected_centroid, 10, (0, 255, 0), 3)
                    cv2.line(vis, image_center, selected_centroid, (0, 255, 0), 2)
                    cv2.putText(
                        vis,
                        "SELECTED: closest to centre",
                        (max(5, box_x), max(24, box_y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        3,
                    )
                    cv2.imwrite(output_path, vis)
                cv2.imshow("SAM3 Gripper Object Detection", vis)
                print(f"[DEBUG] Detection output saved to {output_path}. Press a key to continue.")
                cv2.waitKey(0)
                cv2.destroyWindow("SAM3 Gripper Object Detection")
            except Exception as e:
                print(f"[WARN] Object detection debug visualization failed: {e}")
        if not masks:
            print(f"[STEP] No {name!r} found; returning to start_joint.")
            recovery_error = self._return_to_start_after_load_failure(
                f"no {name!r} detections"
            )
            result = {
                "status": "failed",
                "stage": "load",
                "reason": f"No {name!r} objects were detected with prompt {prompt!r}.",
                "object_found": False,
                "detected_objects": 0,
                "returned_to_start": recovery_error is None,
            }
            if recovery_error is not None:
                result["return_to_start_error"] = recovery_error
            return result

        # Selection is intentionally strict: never fall back to an off-centre
        # detection if the centre-most object has invalid depth or is unreachable.
        targets = self.compute_targets(color, prompt, [selected_mask], depth, intr)
        candidates = self.motion_planner.compute_approach_poses(
            T_base_active_tcp,
            targets,
            T_cam,
            camera,
            z_offset=clearance,
            ignore_rotation=True,
            orientation_profile="gripper_object_pickup",
            planning_tcp_name="tcp_gripper",
        )
        best_candidate, best_cost = self.motion_planner.select_best_candidate(
            candidates, ignore_close=False
        )
        if debug:
            debug_candidate = best_candidate or self.motion_planner.select_debug_candidate(candidates)
            self.screw_detector.show_reach_plan_3d_debug(debug_candidate)
        if best_candidate is None:
            rejected_candidates = []
            for index, candidate in enumerate(candidates):
                reachability = candidate.get("reachability", {}) or {}
                rejected_candidates.append(
                    {
                        "candidate": index + 1,
                        "reason": str(reachability.get("reason", "unreachable")),
                        "planned_pose_tcp_drill": [
                            float(value) for value in candidate.get("pose", [])
                        ],
                    }
                )
            rejection_summary = "; ".join(
                f"candidate {item['candidate']}: {item['reason']}"
                for item in rejected_candidates
            ) or "no candidate poses were generated"
            print(
                f"[PLAN FAILURE] {name!r} was detected, but pickup planning "
                f"was rejected before moveL ({rejection_summary}). No target "
                "motion command was sent. Check the camera/TCP frame conversion "
                "and calibration if the planned pose is incorrect."
            )
            print("[STEP] Returning to start_joint after pickup planning failure.")
            recovery_error = self._return_to_start_after_load_failure(
                f"{name!r} pickup planning failure"
            )
            reason = (
                f"Found {len(masks)} {name!r} candidate(s), but could not pick up "
                f"the object because every planned pose failed the pre-motion "
                f"reachability/IK check: {rejection_summary}. No target motion "
                "command was sent."
            )
            print(f"[PICKUP RESULT] {reason}")
            result = {
                "status": "failed",
                "stage": "load",
                "failure_type": "pre_motion_reachability_rejection",
                "planning_rejected": True,
                "motion_command_sent": False,
                "reason": reason,
                "object": name,
                "object_found": True,
                "detected_objects": len(masks),
                "selected_detection": center_selection,
                "rejected_candidates": rejected_candidates,
                "returned_to_start": recovery_error is None,
            }
            if recovery_error is not None:
                result["return_to_start_error"] = recovery_error
            return result

        if self.cancel_check is not None and self.cancel_check():
            raise ProcedureCancelled("Object load cancelled before moving to the pickup pose.")
        print(
            "[STEP] Moving converted tcp_drill command to the selected "
            f"tcp_gripper object pose (distance={best_cost:.4f} m)."
        )
        self.handles.arm.moveL(best_candidate["pose"], speed=self.config.default_speed)

        if run_program:
            self.run_gripper_pickup_program()
        else:
            print("[DEBUG] Skipping gripper_pickup program because run_program=False.")

        return {
            "status": "ok",
            "stage": "load",
            "message": f"Loaded object with gripper: {name}",
            "object": name,
            "prompt": prompt,
            "detected_objects": len(masks),
            "selected_detection": center_selection,
            "selected_target_base": [float(value) for value in best_candidate["target_base"]],
            "command_pose_tcp_drill": [float(value) for value in best_candidate["pose"]],
            "program_ran": bool(run_program),
            "dropoff_performed_by_program": bool(run_program),
            "returned_to_start": bool(run_program),
        }


class GripperLLMScrewPicker(LLMScrewPicker):
    def __init__(
        self,
        cycle_manager: GripperScrewCycleManager,
        manual_cycle_manager: Optional[ManualScrewCycleManager] = None,
        stop_coordinator: Optional[StopIOCoordinator] = None,
    ):
        super().__init__(
            cycle_manager,
            manual_cycle_manager=manual_cycle_manager,
            stop_coordinator=stop_coordinator,
        )

    @property
    def gripper_cycle(self) -> GripperScrewCycleManager:
        return self.cycle

    def load_object_requested(
        self,
        request: str,
        *,
        voice_client: Optional[StreamedSpeechPipelineClient] = None,
        debug: bool = False,
        run_program: bool = True,
    ) -> Dict[str, Any]:
        if not self._busy_lock.acquire(blocking=False):
            return {"status": "busy", "message": "Robot is already executing an operation."}

        def worker() -> None:
            try:
                if self.stop_coordinator is not None:
                    self.stop_coordinator.begin_operation()
                if voice_client is not None:
                    voice_client.speak_openai("Loading the object with the gripper.")
                result = self.gripper_cycle.load_object_with_gripper(
                    request,
                    debug=debug,
                    run_program=run_program,
                )
                if result.get("status") != "ok":
                    reason = str(result.get("reason", "The gripper load failed."))
                    if result.get("object_found"):
                        print(
                            "[DEBUG] Object found but pickup failed: "
                            f"object={result.get('object', request)!r}, "
                            f"detections={result.get('detected_objects', 0)}, "
                            f"reason={reason}"
                        )
                    else:
                        print(f"[ERROR] Gripper load failed: {reason}")
                    recovery_error = result.get("return_to_start_error")
                    if recovery_error:
                        print(
                            "[WARN] Pickup failure recovery also failed: "
                            f"{recovery_error}"
                        )
                    if voice_client is not None:
                        voice_client.speak_openai(reason)
            except ProcedureCancelled as e:
                self._handle_cancelled_operation(
                    request,
                    voice_client=voice_client,
                    reason=str(e),
                )
            except Exception as e:
                if self.stop_coordinator is not None and self.stop_coordinator.is_cancelled():
                    self._handle_cancelled_operation(
                        request,
                        voice_client=voice_client,
                        reason=f"Object load interrupted by reset: {e}",
                    )
                else:
                    print(f"[ERROR] Gripper load worker failed: {e}")
            finally:
                if self.stop_coordinator is not None:
                    self.stop_coordinator.end_operation()
                self._busy_lock.release()
                print("[READY] Gripper load worker is ready for the next command.")

        threading.Thread(
            target=worker,
            name="gripper-load-worker",
            daemon=True,
        ).start()
        return {"status": "ok", "action": "started", "request": request}


class PersistentManualModeController:
    """Runs an LLM-selected manual tool mode until explicitly stopped."""

    DRILL_MODE = "drill"
    GRIPPER_MODE = "gripper"

    def __init__(
        self,
        cycle_manager: GripperScrewCycleManager,
        manual_cycle_manager: ManualScrewCycleManager,
        operation_lock: threading.Lock,
        stop_coordinator: Optional[StopIOCoordinator] = None,
    ) -> None:
        self.cycle = cycle_manager
        self.manual_cycle = manual_cycle_manager
        self.handles = cycle_manager.handles
        self.config = cycle_manager.config
        self.operation_lock = operation_lock
        self.stop_coordinator = stop_coordinator
        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._mode: Optional[str] = None
        self._program_active = False
        self._last_error: Optional[str] = None

    def start(self, mode: str) -> Dict[str, Any]:
        if mode not in (self.DRILL_MODE, self.GRIPPER_MODE):
            return {"status": "failed", "reason": f"Unknown manual mode: {mode!r}."}
        if mode == self.DRILL_MODE and self.cycle.screwdriver_client is None:
            return {
                "status": "failed",
                "reason": "Manual drill mode requires --enable_screwdriver.",
            }
        if mode == self.GRIPPER_MODE and self.handles.gripper is None:
            return {
                "status": "failed",
                "reason": "Manual gripper mode requires an available gripper handle.",
            }

        joint_key = f"manual_{mode}_joint"
        program_key = f"{mode}_admittance"
        if self.config.get(joint_key) is None:
            return {"status": "failed", "reason": f"Missing `{joint_key}` in cycles.yaml."}
        if self.config.get("programs", program_key) is None:
            return {
                "status": "failed",
                "reason": f"Missing `programs.{program_key}` in cycles.yaml.",
            }

        with self._state_lock:
            if self._mode is not None:
                return {
                    "status": "busy",
                    "reason": f"Manual {self._mode} mode is already active. Stop it before changing modes.",
                    "mode": self._mode,
                }
            if not self.operation_lock.acquire(blocking=False):
                return {
                    "status": "busy",
                    "reason": "The robot is already executing another operation.",
                }
            self._mode = mode
            self._last_error = None
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run,
                args=(mode,),
                name=f"manual-{mode}-mode",
                daemon=True,
            )
            self._thread.start()

        return {
            "status": "ok",
            "action": "started",
            "mode": mode,
            "message": f"Manual {mode} mode is starting.",
        }

    def stop(self, *, wait: bool = True) -> Dict[str, Any]:
        with self._state_lock:
            mode = self._mode
            thread = self._thread
            error = self._last_error
        if mode is None or thread is None:
            return {
                "status": "ok" if error is None else "failed",
                "action": "already_stopped",
                "reason": error,
            }

        print(f"[MANUAL] Stop requested for manual {mode} mode.")
        self._stop_event.set()
        try:
            self.handles.arm_dash.stop()
        except Exception as e:
            print(f"[WARN] Failed to interrupt the manual admittance program: {e}")
        if mode == self.DRILL_MODE and self.cycle.screwdriver_client is not None:
            try:
                if self.cycle.screwdriver_client.is_running():
                    self.cycle.screwdriver_client.stop()
            except Exception as e:
                print(f"[WARN] Failed to interrupt the screwdriver: {e}")

        if wait and thread is not threading.current_thread():
            finish_timeout = float(
                self.config.get("hardware", "timing", "manual_stop_timeout", default=30.0)
            )
            thread.join(timeout=max(0.1, finish_timeout))

        with self._state_lock:
            still_active = self._mode is not None
            error = self._last_error
        if still_active:
            return {
                "status": "stopping",
                "action": "stop_requested",
                "mode": mode,
                "reason": "Manual mode is still returning to start_joint.",
            }
        return {
            "status": "ok" if error is None else "failed",
            "action": "stopped",
            "mode": mode,
            "returned_to_start": True,
            "reason": error,
        }

    def _cancelled(self) -> bool:
        return self._stop_event.is_set() or bool(
            self.stop_coordinator is not None and self.stop_coordinator.is_cancelled()
        )

    def _interruptible_sleep(self, seconds: float) -> bool:
        deadline = time.monotonic() + max(0.0, float(seconds))
        while not self._cancelled():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return True
            self._stop_event.wait(min(0.05, remaining))
        return False

    def _move_to_manual_joint(self, mode: str) -> None:
        joint_key = f"manual_{mode}_joint"
        joints = self.config.get(joint_key)
        print(f"[MANUAL] Moving to {joint_key}... {joints}")
        self.handles.arm.moveJ(joints)

    def _start_admittance(self, mode: str) -> bool:
        timing = self.config.get("hardware", "timing", default={}) or {}
        program_key = f"{mode}_admittance"
        program_name = self.config.get("programs", program_key)
        print(f"[MANUAL] Starting programs.{program_key}: {program_name}")
        self.handles.arm_dash.connect()
        if not self._interruptible_sleep(timing.get("dashboard_connect_delay", 0.1)):
            return False
        response = self.handles.arm_dash.load_urp(program_name)
        print(f"[DASHBOARD] load_urp response: {response}")
        if not self._interruptible_sleep(timing.get("urp_load_delay", 0.5)):
            return False
        response = self.handles.arm_dash.play()
        print(f"[DASHBOARD] play response: {response}")
        self._program_active = True
        return self._interruptible_sleep(timing.get("manual_program_start_delay", 0.5))

    def _stop_admittance(self) -> None:
        if not self._program_active:
            return
        try:
            self.manual_cycle._stop_admittance_program()
        finally:
            self._program_active = False

    def _wait_for_fresh_trigger_hold(self, hold_seconds: float) -> bool:
        poll = float(
            self.config.get("hardware", "timing", "trigger_poll_interval", default=0.1)
        )
        while bool(self.handles.arm.get_tool_io()):
            if self._cancelled():
                return False
            time.sleep(poll)

        pressed_at: Optional[float] = None
        print(f"[MANUAL] Hold the trigger for {hold_seconds:.1f} seconds to run the screwdriver.")
        while not self._cancelled():
            pressed = bool(self.handles.arm.get_tool_io())
            now = time.monotonic()
            if pressed:
                if pressed_at is None:
                    pressed_at = now
                elif now - pressed_at >= hold_seconds:
                    print(f"[MANUAL] Confirmed {hold_seconds:.1f}-second trigger hold.")
                    return True
            else:
                pressed_at = None
            time.sleep(poll)
        return False

    def _run_screwdriver(self) -> bool:
        client = self.cycle.screwdriver_client
        if client is None:
            raise RuntimeError("Screwdriver client is not enabled.")
        print("[MANUAL] Running screwdriver operation...")
        client.run_screw_async(debug=False)
        poll = float(
            self.config.get("hardware", "timing", "screwdriver_poll_interval", default=0.5)
        )
        while not self._cancelled():
            status = client.get_status()
            if status.state in ("completed", "error"):
                if status.state == "error":
                    print(f"[WARN] Screwdriver operation failed: {status.error}")
                    return False
                print("[MANUAL] Screwdriver operation completed.")
                return True
            time.sleep(poll)
        client.stop()
        return False

    def _run_drill_mode(self) -> None:
        hold_seconds = float(
            self.config.get("hardware", "timing", "manual_override_hold_s", default=3.0)
        )
        self._move_to_manual_joint(self.DRILL_MODE)
        while not self._cancelled():
            if not self._start_admittance(self.DRILL_MODE):
                break
            if not self._wait_for_fresh_trigger_hold(hold_seconds):
                break
            self._stop_admittance()
            self._run_screwdriver()
            if self._cancelled():
                break
            self._move_to_manual_joint(self.DRILL_MODE)

    def _run_gripper_mode(self) -> None:
        gripper = self.handles.gripper
        gripper.activate()
        gripper.open(block=False)
        self._move_to_manual_joint(self.GRIPPER_MODE)
        if not self._start_admittance(self.GRIPPER_MODE):
            return

        poll = float(
            self.config.get("hardware", "timing", "trigger_poll_interval", default=0.1)
        )
        was_pressed = False
        print("[MANUAL] Hold the trigger to close the gripper; release it to open.")
        while not self._cancelled():
            pressed = bool(self.handles.arm.get_tool_io())
            if pressed != was_pressed:
                if pressed:
                    print("[MANUAL] Trigger held: closing gripper.")
                    gripper.close(block=False)
                else:
                    print("[MANUAL] Trigger released: opening gripper immediately.")
                    gripper.open(block=False)
                was_pressed = pressed
            time.sleep(poll)
        if was_pressed:
            gripper.open(block=False)

    def _run(self, mode: str) -> None:
        error: Optional[str] = None
        try:
            if self.stop_coordinator is not None:
                self.stop_coordinator.begin_operation()
            if mode == self.DRILL_MODE:
                self._run_drill_mode()
            else:
                self._run_gripper_mode()
        except Exception as e:
            error = str(e)
            print(f"[ERROR] Manual {mode} mode failed: {e}")
        finally:
            try:
                self._stop_admittance()
            except Exception as e:
                error = error or f"Failed to stop manual admittance: {e}"
                print(f"[WARN] {error}")
            if mode == self.GRIPPER_MODE and self.handles.gripper is not None:
                try:
                    self.handles.gripper.open(block=False)
                except Exception as e:
                    print(f"[WARN] Failed to open gripper while leaving manual mode: {e}")
            central_reset_owns_recovery = bool(
                self.stop_coordinator is not None
                and self.stop_coordinator.is_cancelled()
                and self.stop_coordinator.central_reset_enabled()
            )
            try:
                if central_reset_owns_recovery:
                    print(
                        "[MANUAL] Return to start_joint deferred to the central "
                        "reset controller."
                    )
                else:
                    if self.stop_coordinator is not None and self.stop_coordinator.is_cancelled():
                        self.stop_coordinator.wait_until_stop_released()
                    print("[MANUAL] Returning to start_joint...")
                    self.cycle.move_to_start_position()
            except Exception as e:
                error = error or f"Failed to return to start_joint: {e}"
                print(f"[ERROR] {error}")
            finally:
                if self.stop_coordinator is not None:
                    self.stop_coordinator.end_operation()
                with self._state_lock:
                    self._last_error = error
                    self._mode = None
                    self._thread = None
                self.operation_lock.release()
                if central_reset_owns_recovery:
                    print(
                        f"[MANUAL] Manual {mode} mode released control; central "
                        "reset is pending."
                    )
                else:
                    print(f"[READY] Manual {mode} mode stopped; ready for the next instruction.")


class SystemResetController:
    """Serialize all stop sources into a safe return to start_joint."""

    def __init__(
        self,
        cycle_manager: GripperScrewCycleManager,
        manual_mode_controller: PersistentManualModeController,
        stop_coordinator: StopIOCoordinator,
        operation_lock: threading.Lock,
    ) -> None:
        self.cycle = cycle_manager
        self.manual_mode = manual_mode_controller
        self.stop_coordinator = stop_coordinator
        self.operation_lock = operation_lock
        self._reset_lock = threading.Lock()

    def is_resetting(self) -> bool:
        return self._reset_lock.locked()

    def controls_locked(self) -> bool:
        return (
            self.stop_coordinator.is_stop_held()
            or self.stop_coordinator.is_cancelled()
            or self.is_resetting()
        )

    def _move_to_start_when_ready(self) -> None:
        timeout = float(
            self.cycle.config.get(
                "hardware", "timing", "reset_motion_ready_timeout_s", default=15.0
            )
        )
        retry_interval = float(
            self.cycle.config.get(
                "hardware", "timing", "reset_motion_retry_interval_s", default=0.5
            )
        )
        deadline = time.monotonic() + max(0.1, timeout)
        attempt = 0
        last_error: Optional[Exception] = None
        while time.monotonic() < deadline:
            self.stop_coordinator.wait_until_stop_released(clear_cancel=False)
            attempt += 1
            try:
                self.cycle.move_to_start_position()
                return
            except Exception as e:
                last_error = e
                remaining = max(0.0, deadline - time.monotonic())
                print(
                    f"[RESET] start_joint move attempt {attempt} was rejected; "
                    f"waiting for motion-ready ({remaining:.1f}s remaining)."
                )
                time.sleep(max(0.05, retry_interval))
        raise RuntimeError(
            "Robot did not become motion-ready for the start_joint reset within "
            f"{timeout:.1f}s. Last error: {last_error}"
        )

    def reset(self, source: str, *, wait: bool = True) -> Dict[str, Any]:
        if not self._reset_lock.acquire(blocking=False):
            return {
                "status": "resetting",
                "action": "reset_in_progress",
                "reason": "A system reset is already in progress.",
            }

        acquired_operation_lock = False
        try:
            reason = f"System reset requested by {source}."
            self.stop_coordinator.request_cancel(reason)
            self.manual_mode.stop(wait=False)

            # Motion is prohibited until the physical stop input is released.
            self.stop_coordinator.wait_until_stop_released(clear_cancel=False)
            timeout = float(
                self.cycle.config.get(
                    "hardware", "timing", "system_reset_timeout_s", default=60.0
                )
            )
            if not self.stop_coordinator.wait_for_hardware_interrupt(timeout):
                return {
                    "status": "failed",
                    "action": "reset_failed",
                    "source": source,
                    "reason": "Timed out waiting for hardware stop commands to finish.",
                    "returned_to_start": False,
                }
            acquired_operation_lock = self.operation_lock.acquire(
                timeout=max(0.1, timeout) if wait else 0.1
            )
            if not acquired_operation_lock:
                return {
                    "status": "resetting",
                    "action": "reset_requested",
                    "reason": "Waiting for the active operation to finish cancellation recovery.",
                }

            # The input may have been pressed again while waiting for a worker.
            self.stop_coordinator.wait_until_stop_released(clear_cancel=False)
            try:
                self.cycle.handles.arm_dash.stop()
            except Exception as e:
                print(f"[WARN] Dashboard stop during system reset failed: {e}")
            print(f"[RESET] Returning to start_joint after {source}.")
            self._move_to_start_when_ready()
            # Keep controls locked if stop was pressed again during reset motion.
            self.stop_coordinator.wait_until_stop_released(clear_cancel=False)
            self.stop_coordinator.complete_reset()
            print("[READY] System reset complete; waiting for the next instruction.")
            return {
                "status": "ok",
                "action": "reset_complete",
                "source": source,
                "returned_to_start": True,
            }
        except Exception as e:
            return {
                "status": "failed",
                "action": "reset_failed",
                "source": source,
                "reason": str(e),
                "returned_to_start": False,
            }
        finally:
            if acquired_operation_lock:
                self.operation_lock.release()
            self._reset_lock.release()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--debug-step",
        choices=(
            "load",
            "pickup",
            "pickup_target",
            "pickup_target_screwdriver",
            "target",
            "screw",
            "watcher",
        ),
        help="Run only one debug stage.",
    )
    parser.add_argument(
        "--debug-request",
        default="a screw",
        help="Configured object name for load, or screw description for pickup.",
    )
    parser.add_argument("--pickup-debug-gui", action="store_true")
    parser.add_argument("--pickup-debug-gui-wait-ms", type=int, default=1)
    parser.add_argument("--enable_screwdriver", action="store_true")
    parser.add_argument(
        "--load-debug-no-program",
        action="store_true",
        help="For --debug-step load, stop above the object without running gripper_pickup.",
    )
    return parser.parse_args()


def main():
    global ACTIVE_SCREWDRIVER_CLIENT, ACTIVE_VOICE_CLIENT, ACTIVE_HARDWARE_INITIALIZER
    global ACTIVE_VLM_WATCHER
    global ACTIVE_MANUAL_MODE_CONTROLLER

    args = parse_args()
    cycles_config = Config(
        str(Path(__file__).resolve().parent / "qbot" / "config" / "cycles.yaml")
    )
    stop_io_config = cycles_config.get("hardware", "stop_io", default={}) or {}
    stop_active_high = bool(stop_io_config.get("active_high", True))
    detector = Sam3Detector.from_config(cycles_config.get("sam3", default={}) or {})

    hardware_initializer = HardwareInitializer(
        camera_arm_name="camera_drill",
        camera_fixed_name="camera_fixed",
        ignore_gripper=False,
        tool_name="tcp_drill",
        camera_tool_name="tcp_drill",
        extra_arm_cameras={"gripper": "camera_gripper"},
        stop_dashboard_program_on_init=False,
        debug=args.debug,
    )
    ACTIVE_HARDWARE_INITIALIZER = hardware_initializer
    hw = hardware_initializer.initialize()

    if not args.debug_step:
        wait_stop_button_released(hw, None, active_high=stop_active_high)
        start_joint = cycles_config.get("start_joint")
        if start_joint is None:
            raise RuntimeError("Missing `start_joint` in qbot/config/cycles.yaml")
        print("[STEP] Initial startup move to start_joint...", start_joint)
        hw.arm.moveJ(start_joint)

    screwdriver_client = ScrewdriverClient() if args.enable_screwdriver else None
    ACTIVE_SCREWDRIVER_CLIENT = screwdriver_client

    voice_client = StreamedSpeechPipelineClient.load()
    ACTIVE_VOICE_CLIENT = voice_client
    stop_coordinator = StopIOCoordinator(
        hw,
        voice_client=voice_client,
        screwdriver_client=screwdriver_client,
        stop_active_high=stop_active_high,
    )
    stop_coordinator.start()

    cycle_manager = GripperScrewCycleManager(
        hw,
        detector,
        screwdriver_client=screwdriver_client,
        voice_client=voice_client,
        move_to_start=False,
        cancel_check=stop_coordinator.is_cancelled,
    )
    manual_cycle_manager = ManualScrewCycleManager(
        hw,
        screwdriver_client,
        voice_client,
        cancel_check=stop_coordinator.is_cancelled,
    )
    llm_picker = GripperLLMScrewPicker(
        cycle_manager,
        manual_cycle_manager=manual_cycle_manager,
        stop_coordinator=stop_coordinator,
    )
    manual_mode_controller = PersistentManualModeController(
        cycle_manager,
        manual_cycle_manager,
        llm_picker._busy_lock,
        stop_coordinator=stop_coordinator,
    )
    reset_controller = SystemResetController(
        cycle_manager,
        manual_mode_controller,
        stop_coordinator,
        llm_picker._busy_lock,
    )
    stop_coordinator.enable_central_reset()
    stop_coordinator.set_stop_released_handler(
        lambda: reset_controller.reset("physical stop release", wait=True)
    )
    ACTIVE_MANUAL_MODE_CONTROLLER = manual_mode_controller
    vlm_watcher: Optional[VisionPickUpWatcher] = None

    def vlm_watcher_settings():
        watcher_cfg = cycle_manager.config.get("vlm_watcher", default={}) or {}
        if hw.cam_fixed is None:
            raise RuntimeError("The VLM watcher requires the fixed camera.")
        pickup_objects = cycle_manager._configured_pickup_objects()
        known_objects = [item["name"] for item in pickup_objects]
        watcher_actions = watcher_cfg.get("actions", {}) or {}
        if not isinstance(watcher_actions, dict):
            raise ValueError("vlm_watcher.actions must be a mapping")
        allowed_action_tools = {
            "screw_installation_request",
            "load_object_request",
        }
        for action_name, action_cfg in watcher_actions.items():
            if not isinstance(action_cfg, dict):
                raise ValueError(
                    f"vlm_watcher.actions.{action_name} must be a mapping"
                )
            tool_name = str(action_cfg.get("tool", "")).strip()
            if tool_name not in allowed_action_tools:
                raise ValueError(
                    f"vlm_watcher action {action_name!r} uses unknown tool "
                    f"{tool_name!r}; expected one of {sorted(allowed_action_tools)}."
                )
            if bool(action_cfg.get("auto_execute", False)) and tool_name != "screw_installation_request":
                raise ValueError(
                    f"vlm_watcher action {action_name!r} enables auto_execute, "
                    "which currently supports only screw_installation_request."
                )
            if not str(action_cfg.get("requirement", "")).strip():
                raise ValueError(
                    f"vlm_watcher action {action_name!r} requires a non-empty requirement."
                )
        allowed_suggestions = known_objects + [str(name) for name in watcher_actions]
        allowed_suggestions_lower = {name.lower() for name in allowed_suggestions}
        watcher_rules = watcher_cfg.get("rules", []) or []
        for rule in watcher_rules:
            if isinstance(rule, dict):
                suggested = str(rule.get("suggest", "")).strip()
                if suggested and suggested.lower() not in allowed_suggestions_lower:
                    raise ValueError(
                        f"VLM watcher rule suggests {suggested!r}, but it is not an "
                        f"allowed object/action: {allowed_suggestions}."
                    )
        return watcher_cfg, allowed_suggestions, watcher_rules, watcher_actions

    def create_vlm_watcher(on_suggestion, *, debug_window: Optional[bool] = None):
        watcher_cfg, known_suggestions, watcher_rules, _ = vlm_watcher_settings()
        show_debug_window = (
            bool(watcher_cfg.get("show_debug_window", False))
            if debug_window is None
            else bool(debug_window)
        )
        watcher = VisionPickUpWatcher(
            camera=hw.cam_fixed,
            api_url=str(
                watcher_cfg.get(
                    "api_url", "http://127.0.0.1:1234/v1/chat/completions"
                )
            ),
            interval_sec=float(watcher_cfg.get("interval_sec", 2.0)),
            show_debug_window=show_debug_window,
            on_suggestion=on_suggestion,
            temperature=float(watcher_cfg.get("temperature", 0.2)),
            max_tokens=int(watcher_cfg.get("max_tokens", -1)),
            known_objects=known_suggestions,
            rules=watcher_rules,
            request_timeout_sec=float(watcher_cfg.get("request_timeout_sec", 30.0)),
        )
        return watcher, watcher_cfg, known_suggestions

    def run_debug_step() -> Dict[str, Any]:
        nonlocal vlm_watcher
        global ACTIVE_VLM_WATCHER
        requested_step = str(args.debug_step)
        step = requested_step
        pickup_sequence_info: Optional[Dict[str, Any]] = None
        target_sequence_info: Optional[Dict[str, Any]] = None
        stop_coordinator.begin_operation()
        try:
            if (
                requested_step == "pickup_target_screwdriver"
                and cycle_manager.screwdriver_client is None
            ):
                return {
                    "ok": False,
                    "stage": requested_step,
                    "reason": (
                        "The pickup-target-screwdriver sequence requires "
                        "--enable_screwdriver."
                    ),
                }

            if step == "watcher":
                result: Dict[str, Any] = {}
                suggestion_received = threading.Event()

                def on_debug_suggestion(object_name: str, reason: str) -> None:
                    action_cfg = watcher_actions.get(object_name, {}) or {}
                    result.update(
                        {
                            "ok": True,
                            "stage": "watcher",
                            "suggestion": object_name,
                            "type": "action" if action_cfg else "object",
                            "tool": action_cfg.get("tool", "load_object_request"),
                            "requirement": action_cfg.get("requirement", object_name),
                            "auto_execute": bool(action_cfg.get("auto_execute", False)),
                            "reason": reason,
                        }
                    )
                    print(
                        f"[DEBUG VLM WATCHER] suggestion={object_name!r}, "
                        f"reason={reason!r}"
                    )
                    suggestion_received.set()

                watcher_cfg, known_suggestions, watcher_rules, watcher_actions = vlm_watcher_settings()
                print(f"[DEBUG VLM WATCHER] known_suggestions={known_suggestions}")
                print(f"[DEBUG VLM WATCHER] actions={watcher_actions}")
                print(f"[DEBUG VLM WATCHER] rules={watcher_rules}")
                show_window = args.debug and bool(
                    watcher_cfg.get("debug_show_window", True)
                )
                timeout = max(1.0, float(watcher_cfg.get("debug_timeout_sec", 60.0)))
                vlm_watcher, _, _ = create_vlm_watcher(
                    on_debug_suggestion,
                    debug_window=show_window,
                )
                ACTIVE_VLM_WATCHER = vlm_watcher
                vlm_watcher.start()
                print(
                    f"[DEBUG VLM WATCHER] Watching fixed camera for up to "
                    f"{timeout:.1f} seconds. Press stop to cancel."
                )
                started = time.monotonic()
                try:
                    while not suggestion_received.wait(timeout=0.1):
                        stop_coordinator.raise_if_cancelled()
                        if time.monotonic() - started >= timeout:
                            return {
                                "ok": False,
                                "stage": "watcher",
                                "reason": "No valid VLM suggestion before timeout.",
                                "known_suggestions": known_suggestions,
                                "rules": watcher_rules,
                            }
                    return result
                finally:
                    vlm_watcher.stop()
                    vlm_watcher = None
                    ACTIVE_VLM_WATCHER = None

            if step == "load":
                print(f"[DEBUG] Running generic object load for request: {args.debug_request!r}")
                return cycle_manager.load_object_with_gripper(
                    args.debug_request,
                    debug=args.debug,
                    run_program=not args.load_debug_no_program,
                )

            if step in ("pickup", "pickup_target", "pickup_target_screwdriver"):
                print(
                    f"[DEBUG] Running screw pickup stage with the drill camera "
                    f"for request: {args.debug_request!r}"
                )
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
                    debug=args.debug,
                )
                if not ok:
                    return {"ok": False, "stage": step, "reason": info.get("reason", "Screw target was not reachable."), "details": info}

                selected_target_base = info.get("selected_target_base")
                verify_cfg = cycle_manager.config.get(
                    "pickup_verification", default={}
                ) or {}
                if bool(verify_cfg.get("enable_local_refine", True)) and selected_target_base:
                    print(
                        "[DEBUG] Refining screw pickup with local drill-camera "
                        "screw-head detection."
                    )
                    refine_ok, refine_info = cycle_manager.refine_pickup_target_locally(
                        target_base=selected_target_base,
                        camera="arm",
                        z_offset=pickup_clearance,
                        debug=args.debug,
                        debug_gui=args.debug,
                        debug_gui_wait_ms=args.pickup_debug_gui_wait_ms,
                    )
                    stop_coordinator.raise_if_cancelled()
                    info["pickup_refine"] = refine_info
                    if not refine_ok:
                        return {
                            "ok": False,
                            "stage": step,
                            "reason": refine_info.get(
                                "reason", "Local screw pickup refinement failed."
                            ),
                            "details": info,
                        }
                else:
                    info["pickup_refine"] = {
                        "skipped": True,
                        "reason": (
                            "Local refinement is disabled."
                            if selected_target_base
                            else "Coarse pickup did not return selected_target_base."
                        ),
                    }

                cycle_manager.run_pickup_program()
                stop_coordinator.raise_if_cancelled()
                if step == "pickup":
                    return {"ok": True, "stage": step, "details": info}
                pickup_sequence_info = info
                step = "target"
                print(
                    "[DEBUG] Screw pickup stage completed; continuing directly "
                    "to target acquisition."
                )

            if step == "target":
                print("[DEBUG] Repeatedly checking for the green target mark. Press stop to cancel.")
                fixed_camera_acquired = False
                manual_override_hold_s = float(
                    cycle_manager.config.get(
                        "hardware",
                        "timing",
                        "manual_override_hold_s",
                        default=3.0,
                    )
                )
                cycle_manager.move_to_screw_target_position()
                while True:
                    stop_coordinator.raise_if_cancelled()
                    trigger_event = check_tool_button_event(
                        cycle_manager.handles,
                        listen_window_s=0.6,
                        long_press_sec=manual_override_hold_s,
                        poll=0.05,
                    )
                    if trigger_event == "long":
                        if cycle_manager.screwdriver_client is None:
                            return {
                                "ok": False,
                                "stage": requested_step,
                                "reason": (
                                    "The manual target override requires "
                                    "--enable_screwdriver."
                                ),
                            }
                        print(
                            "[MANUAL] Target search bypass requested; "
                            "starting the configured admittance program."
                        )
                        if not manual_cycle_manager.manual_position(
                            debug=args.debug,
                            trigger_hold_sec=manual_override_hold_s,
                        ):
                            stop_coordinator.raise_if_cancelled()
                            return {
                                "ok": False,
                                "stage": requested_step,
                                "reason": "Manual admittance positioning failed.",
                            }
                        target_sequence_info = {
                            "manual_override": True,
                            "green_target_skipped": True,
                            "admittance_completed": True,
                        }
                        step = "screw"
                        print(
                            "[MANUAL] Green target search skipped; "
                            "continuing to the screwdriver operation."
                        )
                        break
                    if fixed_camera_acquired:
                        print(
                            "[DEBUG] Refining target mark from current pose; "
                            "skipping screw_target_joint move."
                        )
                    ok, status = cycle_manager.move_to_target_mark_with_status(
                        debug=args.debug,
                        use_fixed_camera=not fixed_camera_acquired,
                    )
                    fixed_camera_acquired = fixed_camera_acquired or bool(
                        status.get("fixed_camera_success", False)
                    )
                    if ok:
                        if requested_step == "pickup_target_screwdriver":
                            target_sequence_info = status
                            step = "screw"
                            print(
                                "[DEBUG] Target stage completed; continuing "
                                "directly to the screwdriver operation."
                            )
                            break
                        if requested_step == "pickup_target":
                            return {
                                "ok": True,
                                "stage": requested_step,
                                "details": {
                                    "pickup": pickup_sequence_info,
                                    "target": status,
                                },
                            }
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
            if requested_step == "pickup_target_screwdriver":
                return {
                    "ok": bool(ok),
                    "stage": requested_step,
                    "details": {
                        "pickup": pickup_sequence_info,
                        "target": target_sequence_info,
                        "screwdriver": {"ok": bool(ok)},
                    },
                }
            return {"ok": bool(ok), "stage": step}
        except ProcedureCancelled as e:
            return {"ok": False, "stage": requested_step, "reason": str(e)}
        finally:
            stop_coordinator.end_operation()

    if args.debug_step:
        print(f"[INFO] Focused debug step selected: {args.debug_step}")
        # Focused hardware tests run before the voice worker is started.
        wait_stop_button_released(hw, None, active_high=stop_active_high)
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

    base_voice_instructions = (
        "You are Quendabot, a screw-installation robot with a general-purpose gripper.\n"
        "The available tools are `screw_installation_request`, `load_object_request`, "
        "`start_manual_drill_mode`, `start_manual_gripper_mode`, and `stop_and_reset`.\n"
        "Call `load_object_request` to retrieve any object for the user with the "
        "gripper. This includes a screw when the user says grab, get, fetch, bring, hand me, "
        "pick up, or pickup without explicitly asking for installation.\n"
        "For example, `grab a screw`, `get a screw`, and `pick up a screw` must call "
        "`load_object_request` with requirement `screw`.\n"
        "Call `screw_installation_request` only for the full screw pickup-and-installation "
        "workflow, and only when the user explicitly asks to install a screw or asks for "
        "screw pickup and installation.\n"
        "For example, `install a screw` and `do screw pickup and installation` call "
        "`screw_installation_request`. A pickup request by itself is not installation intent.\n"
        "Call `start_manual_drill_mode` when the user asks to enter, start, or use manual drill mode.\n"
        "Call `start_manual_gripper_mode` when the user asks to enter, start, or use manual gripper mode.\n"
        "Always call `stop_and_reset` whenever the user says stop, cancel, reset, abort, "
        "exit manual mode, or leave manual mode, regardless of the current operation.\n"
        "Do not call an installation or load tool when the user is only requesting a manual mode.\n"
        "When a CURRENT VLM WATCHER SUGGESTION is present and the user accepts it, "
        "call the exact Tool with the exact Requirement shown in that suggestion.\n"
        "After the tool runs, reply briefly in plain English and report its exact failure reason.\n"
        "When a tool reports planning_rejected=true and motion_command_sent=false, say that "
        "the object was detected but the pose failed the pre-motion IK/reachability check. "
        "Do not claim the pendant, Remote Control mode, dashboard, controller, or motion "
        "command caused the failure unless the tool result explicitly says so.\n"
        "If you hear the word skirt, it means screw.\n"
    )
    voice_client.instructions = base_voice_instructions
    pending_suggestion: Dict[str, str] = {}
    suggestion_lock = threading.Lock()
    last_suggestion_times: Dict[str, float] = {}
    watcher_action_configs = (
        cycle_manager.config.get("vlm_watcher", "actions", default={}) or {}
    )

    def set_pending_suggestion(suggestion_name: str, reason: str) -> Dict[str, str]:
        action_cfg = watcher_action_configs.get(suggestion_name, {}) or {}
        if action_cfg:
            suggestion_type = "action"
            tool_name = str(action_cfg.get("tool", "screw_installation_request"))
            requirement = str(action_cfg.get("requirement", "screw"))
        else:
            suggestion_type = "object"
            tool_name = "load_object_request"
            requirement = suggestion_name
        details = {
            "suggestion": suggestion_name,
            "type": suggestion_type,
            "tool": tool_name,
            "requirement": requirement,
            "reason": reason,
        }
        with suggestion_lock:
            pending_suggestion.clear()
            pending_suggestion.update(details)
            voice_client.instructions = (
                base_voice_instructions
                + "\nCURRENT VLM WATCHER SUGGESTION:\n"
                + f"Suggestion: {suggestion_name}\nType: {suggestion_type}\n"
                + f"Tool: {tool_name}\nRequirement: {requirement}\n"
                + f"Reason: {reason}\n"
            )
        return details

    def clear_pending_suggestion() -> None:
        with suggestion_lock:
            pending_suggestion.clear()
            voice_client.instructions = base_voice_instructions
    voice_client.tools = [
        {
            "type": "function",
            "function": {
                "name": "screw_installation_request",
                "description": (
                    "Run the full drill-camera screw pickup and screw installation workflow. "
                    "Use only when the user explicitly requests installation or both screw "
                    "pickup and installation. Never use for grab, get, fetch, bring, hand-me, "
                    "or pickup-only requests; those use load_object_request."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "requirement": {
                            "type": "string",
                            "description": "The screw request.",
                        },
                    },
                    "required": ["requirement"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "load_object_request",
                "description": (
                    "Retrieve any requested object for the user using the gripper camera and "
                    "Robotiq gripper. Use for grab, get, fetch, bring, hand-me, and pickup-only "
                    "requests, including requests such as 'grab a screw'."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "requirement": {
                            "type": "string",
                            "description": "The object name to find and retrieve.",
                        }
                    },
                    "required": ["requirement"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "start_manual_drill_mode",
                "description": "Enter persistent manual drill mode for hand-guided screwdriver use.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "start_manual_gripper_mode",
                "description": "Enter persistent manual gripper mode with trigger-controlled closing and opening.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "stop_and_reset",
                "description": "Immediately cancel any operation, wait for physical stop release, and return to start_joint.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        },
    ]
    voice_client.tool_choice = "auto"

    def controls_locked_result() -> Optional[Dict[str, Any]]:
        if not reset_controller.controls_locked():
            return None
        return {
            "status": "blocked",
            "reason": (
                "The system is stopped or resetting. Release the physical stop "
                "input and wait for the robot to return to start_joint."
            ),
            "send_to_model": True,
        }

    def screw_installation_requested_handler(requirement: str = ""):
        locked = controls_locked_result()
        if locked is not None:
            return locked
        screw_request = requirement
        if not screw_request:
            return {
                "status": "failed",
                "reason": "Missing screw request. Expected `requirement`.",
                "send_to_model": True,
            }
        clear_pending_suggestion()
        print(f"requirement={requirement!r}")
        result = llm_picker.installation_requested_screw_async(
            screw_request,
            voice_client=voice_client,
            debug=args.debug,
            pickup_debug_gui=args.debug and args.pickup_debug_gui,
            pickup_debug_gui_wait_ms=args.pickup_debug_gui_wait_ms,
        )
        return {**result, "send_to_model": True}

    def load_object_requested_handler(requirement: str = ""):
        locked = controls_locked_result()
        if locked is not None:
            return locked
        if not requirement:
            return {
                "status": "failed",
                "reason": "Missing object request. Expected `requirement`.",
                "send_to_model": True,
            }
        clear_pending_suggestion()
        result = llm_picker.load_object_requested(
            requirement,
            voice_client=voice_client,
            debug=args.debug,
            run_program=True,
        )
        return {
            **result,
            "send_to_model": True,
            "suppress_followup_response": result.get("status") == "ok",
        }

    def start_manual_drill_mode_handler():
        locked = controls_locked_result()
        if locked is not None:
            return locked
        clear_pending_suggestion()
        return {
            **manual_mode_controller.start(PersistentManualModeController.DRILL_MODE),
            "send_to_model": True,
        }

    def start_manual_gripper_mode_handler():
        locked = controls_locked_result()
        if locked is not None:
            return locked
        clear_pending_suggestion()
        return {
            **manual_mode_controller.start(PersistentManualModeController.GRIPPER_MODE),
            "send_to_model": True,
        }

    def stop_and_reset_handler():
        clear_pending_suggestion()
        return {
            **reset_controller.reset("voice or text command", wait=True),
            "send_to_model": True,
        }

    voice_client.register_tool_handler("screw_installation_request", screw_installation_requested_handler)
    voice_client.register_tool_handler("load_object_request", load_object_requested_handler)
    voice_client.register_tool_handler("start_manual_drill_mode", start_manual_drill_mode_handler)
    voice_client.register_tool_handler("start_manual_gripper_mode", start_manual_gripper_mode_handler)
    voice_client.register_tool_handler("stop_and_reset", stop_and_reset_handler)
    voice_client.on_error = lambda e: print("[LLM] error:", e)
    voice_client.on_text_completed = lambda txt: print(f"[LLM] {txt}")
    voice_client.start()
    voice_client.start_background()

    watcher_cfg = cycle_manager.config.get("vlm_watcher", default={}) or {}
    if bool(watcher_cfg.get("enabled", True)):
        suggestion_cooldown = max(
            0.0, float(watcher_cfg.get("suggestion_cooldown_sec", 30.0))
        )

        def on_vlm_suggestion(suggestion_name: str, reason: str) -> None:
            if llm_picker._busy_lock.locked():
                print(f"[VLM WATCHER] Ignoring {suggestion_name!r}; robot is busy.")
                return
            now = time.monotonic()
            key = suggestion_name.lower()
            previous = last_suggestion_times.get(key, float("-inf"))
            if now - previous < suggestion_cooldown:
                return
            last_suggestion_times[key] = now
            action_cfg = watcher_action_configs.get(suggestion_name, {}) or {}
            if bool(action_cfg.get("auto_execute", False)):
                requirement = str(action_cfg.get("requirement", "screw"))
                if cycle_manager.screwdriver_client is None:
                    message = (
                        "Cannot auto-execute screw_installation because the "
                        "screwdriver client is disabled. Start this demo with "
                        "--enable_screwdriver."
                    )
                    print(f"[VLM WATCHER] {message}")
                    voice_client.speak_openai(message)
                    return
                print(
                    f"[VLM WATCHER] Auto-executing {suggestion_name!r}: "
                    f"screw_installation_request({requirement!r}); reason={reason!r}"
                )
                execution_result = screw_installation_requested_handler(requirement)
                print(f"[VLM WATCHER] Action result: {execution_result}")
                if execution_result.get("status") == "ok" and vlm_watcher is not None:
                    vlm_watcher.pause_watcher()

                    def resume_watcher_after_action() -> None:
                        while llm_picker._busy_lock.locked():
                            time.sleep(0.25)
                        if suggestion_cooldown:
                            time.sleep(suggestion_cooldown)
                        last_suggestion_times[key] = time.monotonic()
                        if vlm_watcher is not None:
                            vlm_watcher.resume_watcher()

                    threading.Thread(
                        target=resume_watcher_after_action,
                        name="vlm-watcher-action-cooldown",
                        daemon=True,
                    ).start()
                return

            details = set_pending_suggestion(suggestion_name, reason)
            reason_text = f" because {reason}" if reason else ""
            spoken_name = str(
                action_cfg.get("spoken_name", suggestion_name.replace("_", " "))
            )
            question = (
                f"It looks like you may need {spoken_name}{reason_text}. "
                "Would you like me to do that for you?"
            )
            print(
                f"[VLM WATCHER] Suggesting {suggestion_name!r}: {reason}; "
                f"on acceptance call {details['tool']}({details['requirement']!r})"
            )
            voice_client.speak_openai(question)

        vlm_watcher, _, known_objects = create_vlm_watcher(
            on_vlm_suggestion,
        )
        ACTIVE_VLM_WATCHER = vlm_watcher
        vlm_watcher.start()
        print(
            "[INFO] Fixed-camera VLM watcher started with suggestions: "
            + ", ".join(known_objects)
        )

    print("[STEP] Checking stop button before starting...")
    wait_stop_button_released(
        hw,
        voice_client,
        active_high=stop_active_high,
    )

    try:
        voice_client.speak_openai("Quendabot online. General gripper loading mode enabled.")
    except Exception as e:
        print(f"[WARN] Failed to play startup cue: {e}")

    print("[INFO] Gripper LLM demo running. Speak or type an object or screw request.")
    print("[INFO] Commands: /wake, /listen, /say <text>, /stop, /quit")

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

                if msg.lower() in ("stop", "/stop", "cancel", "reset", "abort"):
                    result = reset_controller.reset("keyboard command", wait=True)
                    print(f"[RESET] {result}")
                    continue

                try:
                    voice_client.respond_once(msg)
                except Exception as e:
                    print(f"[text input ERROR] {e}")
        except KeyboardInterrupt:
            pass

    keyboard_thread = threading.Thread(target=input_loop, name="gripper-screw-llm-keyboard", daemon=True)
    keyboard_thread.start()

    try:
        while keyboard_thread.is_alive():
            keyboard_thread.join(timeout=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        print("[INFO] Shutting down gripper screw demo...")
        try:
            manual_mode_controller.stop(wait=True)
            ACTIVE_MANUAL_MODE_CONTROLLER = None
        except Exception as e:
            print(f"[WARN] Failed to stop manual mode cleanly: {e}")
        try:
            if vlm_watcher is not None:
                vlm_watcher.stop()
                ACTIVE_VLM_WATCHER = None
        except Exception as e:
            print(f"[WARN] Failed to stop VLM watcher: {e}")
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
