from __future__ import annotations

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import threading, yaml
from typing import Optional
try:
    import mediapipe as mp
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "MediaPipe is required for Qbot hand/person tracking. Install it in the "
        "active environment, for example: "
        "conda run -n quendabot_anygrasp python -m pip install mediapipe"
    ) from exc

from hardware.hardware_init import HardwareInitializer, HardwareHandles
from detectors.mediapipe_body_tracker import RealSenseMediapipeTracker,DisplayHub
from commons.grasp_utils import check_path_exists
from qbot.behaviours.base_action import ACTIONS_REGISTRY
import argparse, time
from hardware.hardware_init import HardwareInitializer

CFG_PATH = 'config/behaviour.yaml'

class BehaviorManager:
    def __init__(self, hw: HardwareHandles, debug: bool = False):
        self.hw = hw
        self.arm = hw.arm
        if self.arm is None:
            raise ValueError("HardwareHandles.arm is None (ignore_arm=True?).")
        if hw.T_base_fixed_camera is None:
            raise ValueError("HardwareHandles.T_base_fixed_camera is required for person tracking.")

        # config
        cfg = {}
        cfg_file = check_path_exists(CFG_PATH,__file__)
        if cfg_file:
            with cfg_file.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        else:
            print('ERROR no cfg file', cfg_file)
            raise SystemExit(1)
        self.cfg = cfg
        self.debug = debug

        # tracker on fixed camera
        self.fix_cam_tracker = RealSenseMediapipeTracker(
            camera_obj=hw.cam_fixed,
            cam_cfg=hw.cam_fixed_cfg,
            debug=debug,
            preview_window_name = "Fix Camera MediaPipe",
        )
        # tracker on arm camera
        self.arm_cam_tracker = RealSenseMediapipeTracker(
            camera_obj=hw.cam_arm,
            cam_cfg=hw.cam_arm_cfg,
            debug=debug,
            preview_window_name = "Arm Camera MediaPipe",
        )


        self.T_base_fixed_camera = hw.T_base_fixed_camera
        self.T_tcp_gripper = hw.T_tcp_cam 
   
        self.pose_module = mp.solutions.pose
        self.landmark_map = {
            "nose": self.pose_module.PoseLandmark.NOSE.value,
            "left_shoulder": self.pose_module.PoseLandmark.LEFT_SHOULDER.value,
            "right_shoulder": self.pose_module.PoseLandmark.RIGHT_SHOULDER.value,
            "left_hip": self.pose_module.PoseLandmark.LEFT_HIP.value,
            "right_hip": self.pose_module.PoseLandmark.RIGHT_HIP.value,
            "left_wrist": self.pose_module.PoseLandmark.LEFT_WRIST.value,
            "right_wrist": self.pose_module.PoseLandmark.RIGHT_WRIST.value,
        }

        # single-action state
        self._lock = threading.RLock()
        self._active_name: Optional[str] = None
        self._active_stop: Optional[threading.Event] = None
        self._active_thread: Optional[threading.Thread] = None
        self._active_obj = None  # the BaseAction instance

    # ------------- tracker lifecycle -------------
    def start_tracker(self):
        try:
            self.fix_cam_tracker.start_thread()
        except Exception:
            pass

    def stop_tracker(self):
        try:
            self.fix_cam_tracker.stop()
        except Exception:
            pass

    # ------------- single-action orchestrator -------------
    def start_action(self, name: str, **kwargs) -> None:
        name = name.strip()
        with self._lock:
            alias = {
                "look": "idle_lookaround",
                "point": "point_at_person",
                "handover": "handover_to_hand",
            }
            canonical = alias.get(name, name)
            cls = ACTIONS_REGISTRY.get(canonical) or ACTIONS_REGISTRY.get(name)
            if cls is None:
                raise KeyError(f"Unknown action '{name}'. Known: {sorted(ACTIONS_REGISTRY.keys())}")

            # stop current if different
            if self._active_name is not None:
                if self._active_name == canonical:
                    # already running
                    return
                self._stop_current_locked()

            # instantiate
            action = cls(hw=self.hw, cfg=self.cfg, tracker=self.fix_cam_tracker, alt_tracker=self.arm_cam_tracker, manager=self, debug=self.debug)
            on_complete = kwargs.pop("on_complete", None)
            action.set_callback("on_complete", on_complete)
            on_notify = kwargs.pop("on_notify", None)
            action.set_callback("on_notify", on_notify)

            stop_ev = threading.Event()
            th = threading.Thread(target=action.run, args=(stop_ev,), kwargs=kwargs, daemon=True)

            # set active
            self._active_name = canonical
            self._active_stop = stop_ev
            self._active_thread = th
            self._active_obj = action

            action.on_start()
            th.start()

    def stop_action(self, name: str) -> None:
        alias = {
            "look": "idle_lookaround",
            "point": "point_at_person",
            "handover": "handover_to_hand",
        }
        canonical = alias.get(name, name)
        with self._lock:
            if self._active_name == canonical:
                self._stop_current_locked()

    def stop_current_action(self) -> None:
        with self._lock:
            self._stop_current_locked()

    def stop_all_actions(self) -> None:
        # Only one is ever running, but this satisfies the API.
        self.stop_current_action()

    def _stop_current_locked(self):
        if self._active_thread is None:
            return
        stop_ev = self._active_stop
        th = self._active_thread
        obj = self._active_obj
        # clear active first to avoid races
        self._active_name = None
        self._active_stop = None
        self._active_thread = None
        self._active_obj = None
        try:
            if stop_ev is not None:
                stop_ev.set()
            if th is not None and th.is_alive() and threading.current_thread() is not th:
                th.join(timeout=1.0)
        finally:
            try:
                if obj is not None:
                    obj.on_stop()
            except Exception:
                pass

    def get_active_action(self) -> Optional[str]:
        with self._lock:
            return self._active_name

    # ------------- convenience wrappers -------------
    def start_point_at_person(self, on_complete=None):
        self.start_action("point_at_person", on_complete=on_complete)

    def start_idle_lookaround(self, on_complete=None):
        self.start_action("idle_lookaround", on_complete=on_complete)

    def start_handover_to_hand(self, hand: str = "either", on_complete=None):
        self.start_action("handover_to_hand", hand=hand, on_complete=on_complete)


    # ------------- shutdown -------------
    def shutdown(self, include_tracker: bool = True, park: bool = False):
        self.stop_all_actions()
        if include_tracker:
            self.stop_tracker()
        if park:
            try:
                if hasattr(self.arm, "locations"):
                    self.arm.request_stop()
            except Exception as e:
                print(f"[shutdown] park error: {e}")



def main():
    parser = argparse.ArgumentParser(description="Run Qbot behaviors (modular)")
    parser.add_argument("--behavior", choices=["point", "look", "handover"], default="point")
    parser.add_argument("--hand", choices=["left", "right","either"], default="either", help="For --behavior handover")
    parser.add_argument("--ignore_gripper", action="store_true", help="Run with fake arm/tracker")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()

    initializer = HardwareInitializer(ignore_gripper=args.ignore_gripper)
    hw = initializer.initialize()
    print(hw.arm.locations)

    mgr = BehaviorManager(hw, debug=args.debug)

    try:
        if args.behavior == "point":
            mgr.start_point_at_person()
        elif args.behavior == "look":
            mgr.start_idle_lookaround()
        elif args.behavior == "handover":
            mgr.start_handover_to_hand(hand=args.hand)
        else:
            raise SystemExit(f"Unknown behavior {args.behavior}")

        if args.debug:        
            hub = DisplayHub([mgr.fix_cam_tracker,mgr.arm_cam_tracker])
        # mediapipe landmarks

        while True:
            if args.debug:        
              hub.tick(poll_ms=1)
            time.sleep(0.1)
    finally:
        mgr.shutdown(include_tracker=True, park=False)

if __name__ == "__main__":
    main()
