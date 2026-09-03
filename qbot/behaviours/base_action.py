# behaviours/base_action.py
from __future__ import annotations
import threading
from typing import Dict, Any, Optional, Callable
import numpy  as np
ACTIONS_REGISTRY: Dict[str, type["BaseAction"]] = {}


def register_action(name: str):
    """Decorator to register an action class under a short name."""
    def _decorator(cls: type["BaseAction"]):
        ACTIONS_REGISTRY[name] = cls
        return cls
    return _decorator

class BaseAction:
    """Base interface for all actions. Subclasses implement run()."""

    def __init__(self, *, hw, cfg: Dict[str, Any], tracker,alt_tracker, manager, debug: bool = False):
        self.hw = hw
        self.arm = hw.arm
        self.gripper = hw.gripper
        self.tracker = tracker
        self.alt_tracker = alt_tracker
        self.cfg = cfg or {}
        self.manager = manager
        self.debug = debug
        self._callbacks: Dict[str, Callable[[Dict[str, Any]], None]] = {}  # NEW

    # ---- callbacks (NEW) ----
    def set_callback(self, name: str, fn: Optional[Callable[[Dict[str, Any]], None]]):
        if callable(fn):
            self._callbacks[name] = fn

    def _emit(self, name: str, payload: Dict[str, Any]):
        cb = self._callbacks.get(name)
        if not callable(cb):
            return
        def _call():
            try:
                cb(payload)
            except Exception as e:
                print(f"[{self.__class__.__name__}] callback '{name}' error:", e)
        threading.Thread(target=_call, daemon=True).start()

    def complete(self, status: str, **extra):
        """Notify completion with a unified payload for all actions."""
        payload = {"status": status, "action": self.__class__.__name__}
        payload.update(extra or {})
        self._emit("on_complete", payload)

    def notify(self, status: str, **extra):
        """Notify completion with a unified payload for all actions."""
        payload = {"status": status, "action": self.__class__.__name__}
        payload.update(extra or {})
        self._emit("on_notify", payload)

    
    # Optional lifecycle hooks
    def on_start(self) -> None:
        pass

    def on_stop(self) -> None:
        pass

    # Required
    def run(self, stop_event: threading.Event, **kwargs):  # pragma: no cover
        raise NotImplementedError

    # -------- shared helpers (available to all actions) --------
    def dbg(self, *a):
        if self.debug:
            print(*a)

    def select_target_point(self, body, target, landmark_map):
        """
        body: dict[int -> np.ndarray(3,)] OR array-like
        target: str (e.g., 'nose') or numeric-like (e.g., '0' or 0)
        landmark_map: dict[str -> int] mapping names to indices
        returns: np.ndarray(3,) or None
        """
        # Resolve target index
        idx = None
        if isinstance(target, str):
            if landmark_map and target in landmark_map:
                idx = int(landmark_map[target])
            else:
                # string might actually be a number (e.g. "0")
                try:
                    idx = int(target)
                except Exception:
                    idx = None
        else:
            # numeric target
            try:
                idx = int(target)
            except Exception:
                idx = None

        # Handle dict[int -> point]
        if isinstance(body, dict):
            if idx is not None and idx in body:
                pt = np.asarray(body[idx], dtype=float)
                return pt if pt.size == 3 and not np.isnan(pt).any() else None

            # Fallback: centroid of all valid points
            pts = [np.asarray(v, dtype=float) for v in body.values()
                if v is not None and np.asarray(v).size == 3 and not np.isnan(v).any()]
            if len(pts) == 0:
                return None
            return np.mean(np.vstack(pts), axis=0)

        # Handle array-like (Nx3 or 3,)
        arr = np.asarray(body, dtype=float)
        if arr.ndim == 1 and arr.size == 3:
            return arr if not np.isnan(arr).any() else None
        if arr.ndim == 2 and arr.shape[1] == 3 and arr.shape[0] > 0:
            if idx is not None and 0 <= idx < arr.shape[0]:
                pt = arr[idx]
                return pt if not np.isnan(pt).any() else None
            return np.nanmean(arr, axis=0)
        return None
