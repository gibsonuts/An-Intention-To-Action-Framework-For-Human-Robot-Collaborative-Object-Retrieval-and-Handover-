from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import time
import threading
from typing import Optional, Any, Tuple, Callable, List

import base64
import io
import json
from datetime import datetime

import cv2
import requests
import numpy as np
from PIL import Image
import yaml

from hardware.hardware_init import HardwareInitializer   # your existing initializer
from llm.openai_realtime_client import OpenAIRealtimeClient

CFG_PATH = "config/vlm_watcher.yaml"


def check_path_exists(input_path, file=__file__):
    if not input_path:
        print("input path is none")
        return None

    base_dir = Path(file).resolve().parent
    p = Path(input_path)
    if not p.is_absolute():
        p = base_dir / p

    try:
        abs_path = p.resolve(strict=True)
        print(abs_path)
        return abs_path
    except FileNotFoundError:
        print(p, "does not exist")
        return None


class VisionPickUpWatcher:
    """
    Uses an existing camera object (from HardwareInitializer) instead of cv2.VideoCapture.

    `camera` is expected to be something like `hw.cam_arm` or `hw.cam_fixed`
    that implements `get_rgb_jpeg()`.

    The VLM is purely perceptual:
    - It gets the current frame + KNOWN_OBJECTS.
    - If appropriate, it calls `suggest_object(object_name, reason, urgency_percent)`.
    - This watcher then calls `on_suggestion(object_name, urgency_percent, reason)`.

    Higher-level logic (e.g., asking the user and calling get_object) is handled
    outside this class.
    """

    def __init__(
        self,
        camera: Any,
        api_url: str = "http://127.0.0.1:1234/v1/chat/completions",
        interval_sec: float = 2.0,
        show_debug_window: bool = False,
        on_suggestion: Optional[Callable[[str, str], None]] = None,
        temperature: float = 0.2,
        max_tokens: int = -1,
        known_objects: Optional[List[str]] = None,
        rules: Optional[List[Any]] = None,
        request_timeout_sec: float = 30.0,
    ):
        self.camera = camera
        self.api_url = api_url
        self.pause = False
  
        self.interval_sec = interval_sec
        self.show_debug_window = show_debug_window
        self.on_suggestion = on_suggestion
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.request_timeout_sec = max(1.0, float(request_timeout_sec))

        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()

        self.config_path = check_path_exists(CFG_PATH, __file__)

        # ---------- Load prompt + tools + known objects from YAML ----------
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except FileNotFoundError:
            raise RuntimeError(f"Config file not found: {self.config_path}")

        self._system_prompt: str = cfg.get("system_prompt", "").strip()
        if not self._system_prompt:
            raise RuntimeError("system_prompt missing or empty in YAML config")
        self._extra_information: str = cfg.get("extra_infomation", "").strip()
        
        self.model = cfg.get("model", "gemma-3-12B-it-QAT-Q4_0.gguf")
        self._tools = cfg.get("tools", [])
        if not isinstance(self._tools, list) or not self._tools:
            raise RuntimeError("tools missing or empty in YAML config")

        configured_objects = cfg.get("known_objects", []) or []
        self.known_objects = [
            str(obj).strip() for obj in (
                known_objects if known_objects is not None else configured_objects
            )
            if str(obj).strip()
        ]
        if not self.known_objects:
            raise RuntimeError("known_objects is empty")

        if rules is not None:
            formatted_rules = []
            for index, rule in enumerate(rules, start=1):
                if isinstance(rule, str):
                    text = rule.strip()
                elif isinstance(rule, dict):
                    condition = str(rule.get("if", rule.get("when", ""))).strip()
                    suggestion = str(rule.get("suggest", "")).strip()
                    text = f"If {condition}, suggest '{suggestion}'." if condition and suggestion else ""
                else:
                    text = ""
                if text:
                    formatted_rules.append(f"{index}. {text}")
            if not formatted_rules:
                raise RuntimeError("VLM watcher rules are empty or invalid")
            self._extra_information = "Important suggestion rules:\n" + "\n".join(formatted_rules)
        if not self._extra_information:
            raise RuntimeError("VLM watcher rules/extra information are empty")

        # Normalize for comparison
        self._known_objects_lower = {obj.lower(): obj for obj in self.known_objects}


        print(f"[VisionPickUpWatcher] known_objects={self.known_objects}")
    # ---------- Public control methods ----------

    def start(self):
        """Start the watcher in its own thread."""
        if self._thread and self._thread.is_alive():
            return
        self._running.set()
        self._thread = threading.Thread(target=self._run_loop, name="VisionPickUpWatcher", daemon=True)
        self._thread.start()
        print("[VisionPickUpWatcher] started.")

    def stop(self):
        """Stop the watcher thread and clean up resources."""
        self._running.clear()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        print("[VisionPickUpWatcher] stopped.")

    # ---------- Internal helpers ----------

    def _frame_to_data_url(self, frame, max_w=1280, max_h=720) -> str:
        """Convert OpenCV BGR frame to base64 PNG data URL."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        buf.close()
        return f"data:image/png;base64,{b64}"

    def _call_llm(self, image_data_url: str, extra_context: str = "") -> Tuple[str, list]:
        """
        Send frame to VLM, return (content, tool_calls).

        content is normally "" when idle. tool_calls is a list if any tools were called.

        NOTE: This call is stateless; we send only:
          - system prompt from YAML
          - a single user message containing:
              - current instructions
              - KNOWN_OBJECTS text
              - current image
        """
        known_suggestions_text = "\n".join(f"- {name}" for name in self.known_objects) or "- (none configured)"

        user_text = (
            "You are viewing a live camera frame from the robot's camera.\n"
            "Follow the configured rules and choose at most one KNOWN_SUGGESTION.\n"
            "A suggestion may name a fetchable object or a configured robot action.\n"
            "If yes, call the `object_suggester` tool.\n"
            "If not, stay silent (empty content, no tools).\n\n"
            f"KNOWN_SUGGESTIONS:\n{known_suggestions_text}\n\n"
            f"Suggestion rules:\n{extra_context}"
        )
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": self._system_prompt}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ],
                },
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
            "tools": self._tools,
            "tool_choice": "auto",
        }

        resp = requests.post(
            self.api_url,
            json=payload,
            timeout=self.request_timeout_sec,
        )
        resp.raise_for_status()
        data = resp.json()
        message = data["choices"][0]["message"]
        content = (message.get("content") or "").strip()
        tool_calls = message.get("tool_calls") or []
        return content, tool_calls

    def pause_watcher(self):
        self.pause = True

    def resume_watcher(self):
        self.pause = False

    def _dispatch_tool_call(self, tool_call: dict):
        """
        Handle a single tool call; expects only `suggest_object`.

        The tool should have arguments:
          - object_name (string)
          - reason (string)
          - urgency_percent (number 0–100)
        """
        
        function_name = tool_call["function"]["name"]
        raw_arguments = tool_call["function"].get("arguments", "{}")

        try:
            args = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
        except json.JSONDecodeError:
            print(f"[VisionPickUpWatcher] Error decoding tool arguments for {function_name}: {raw_arguments}")
            return

        if function_name != "object_suggester":
            print(f"[VisionPickUpWatcher] searching found : {function_name} with args: {raw_arguments}")
            return
        
        print(args)
        # people_detected = args.get("people_detected",0)
        suggestions = args.get("object_suggestion") or []
        if isinstance(suggestions, str):
            suggestions = [suggestions]
        if not isinstance(suggestions, list):
            print(f"[VisionPickUpWatcher] Ignoring invalid object_suggestion: {suggestions!r}")
            return
        reason = (args.get("reason") or "").strip()
        # if people_detected == 0:
        #     return
        for obj_raw in suggestions:
            obj_key = str(obj_raw).strip().lower()
            if not obj_key or obj_key == "nothing":
                continue
            object_name = self._known_objects_lower.get(obj_key)
            if object_name is None:
                print(
                    f"[VisionPickUpWatcher] Ignoring suggestion {obj_raw!r}; "
                    "it is not in known_objects."
                )
                continue
            if self.on_suggestion:
                try:
                    self.on_suggestion(object_name, reason)
                except Exception as e:
                    print("[VisionPickUpWatcher] Error in on_suggestion callback:", e)
            return

        # urg_raw = args.get("urgency_percent", 0)

        # if not obj_raw or not reason:
        #     print("[VisionPickUpWatcher] Ignoring suggest_object: missing object_name or reason.")
        #     return

        # # Map back to canonical name from known_objects; enforce that it's valid.
        # obj_key = obj_raw.lower()
        # if obj_key not in self._known_objects_lower:
        #     print(f"[VisionPickUpWatcher] Ignoring suggest_object: object_name {obj_raw!r} not in KNOWN_OBJECTS.")
        #     return

        # object_name = self._known_objects_lower[obj_key]

        # try:
        #     urgency_percent = float(urg_raw)
        # except (TypeError, ValueError):
        #     print(f"[VisionPickUpWatcher] Ignoring suggest_object: invalid urgency_percent={urg_raw!r}.")
        #     return


        # # Clamp urgency to 0–100 just in case.
        # urgency_percent = max(0.0, min(100.0, urgency_percent))

        # print(
        #     f"[VisionPickUpWatcher] suggest_object: "
        #     f"people_detected='{people_detected}, object='{object_name}', urgency={urgency_percent:.1f}%, reason='{reason}'"
        # )

        # if self.on_suggestion:
        #     try:
        #         self.on_suggestion(object_name, urgency_percent, reason)
        #     except Exception as e:
        #         print("[VisionPickUpWatcher] Error in on_suggestion callback:", e)

    def _run_loop(self):
        """
        Main loop:
        - Grabs JPEG frames from the provided camera (RealSense via HardwareInitializer).
        - Decodes to BGR for OpenCV.
        - Periodically sends frames to the VLM.
        """
        print("[VisionPickUpWatcher] Using HardwareInitializer camera (no cv2.VideoCapture).")
        last_time = 0.0

        while self._running.is_set():
            # --- Get frame from RealSense / network camera ---
            if self.pause:
                time.sleep(1)
                continue
            try:
                frame_jpeg = self.camera.get_rgb_jpeg()
                if frame_jpeg is None:
                    print("[VisionPickUpWatcher] WARNING: camera returned None bytes.")
                    time.sleep(0.1)
                    continue

                np_bytes = np.frombuffer(frame_jpeg, np.uint8)
                frame = cv2.imdecode(np_bytes, cv2.IMREAD_COLOR)
                if frame is None:
                    print("[VisionPickUpWatcher] WARNING: failed to decode JPEG.")
                    time.sleep(0.1)
                    continue
            except Exception as e:
                print("[VisionPickUpWatcher] ERROR: failed to get frame from camera:", e)
                time.sleep(0.1)
                continue

            # --- Throttle calls to VLM ---
            now = time.time()
            if now - last_time < self.interval_sec:
                time.sleep(min(0.05, self.interval_sec - (now - last_time)))
                continue
            last_time = now

            try:
                img_url = self._frame_to_data_url(frame)
                ctx = f"Timestamp: {datetime.now().isoformat()}"
                content, tool_calls = self._call_llm(img_url, extra_context=self._extra_information)
            except Exception as e:
                print("[VisionPickUpWatcher] Error calling VLM:", e)
                continue

            # --- Optional debug window ---
            if self.show_debug_window:
                cv2.imshow("VisionPickUpWatcher (debug)", frame)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC closes debug window
                    self.show_debug_window = False
                    cv2.destroyWindow("VisionPickUpWatcher (debug)")


            if tool_calls:
                for tc in tool_calls:
                    self._dispatch_tool_call(tc)
            else:
                # Idle; ignore stray content (should be empty if prompt is respected).
                if content:
                    # print("[VisionPickUpWatcher] Ignoring idle VLM text:", content)
                    pass

        if self.show_debug_window:
            cv2.destroyAllWindows()
        print("[VisionPickUpWatcher] Loop exited.")


def make_on_suggestion():
    """
    Returns a callback for VisionPickUpWatcher:
      on_suggestion(object_name, urgency_percent, reason)
    """
    def _callback(object_name: str, reason: str):
        print(f"[SuggestionCallback] {object_name=} {reason=}")


        # Ask the user via your voice agent
  
        # The LLM can be prompted so that a "yes" will trigger get_object(object_name)
        # text = (
        #     f"It looks like you might need your {object_name} "
        #     f"because {reason}. Would you like me to get it for you?"
        # )
        # client.speak_openai(text)

    return _callback

def main():
    initializer = HardwareInitializer(ignore_arm=False, ignore_gripper=False, debug=False)
    hw = initializer.initialize()

    client = OpenAIRealtimeClient.load()
    client.on_error = lambda e: print("[App] error:", e)

    # Set up Qbot, get_object tool, etc. (your existing code)
    # ...

    # Read threshold from watcher instance after construction
    camera = hw.cam_fixed
    watcher = VisionPickUpWatcher(
        camera=camera,
        api_url="http://127.0.0.1:1234/v1/chat/completions",
        interval_sec=2.0,
        show_debug_window=True,
        on_suggestion=make_on_suggestion,  # we'll attach after constructing
        temperature=0.2,
        max_tokens=-1,
    )

    watcher.start()
    print("[Main] VisionPickUpWatcher running. Press Ctrl+C to exit.")

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[Main] KeyboardInterrupt received. Stopping watcher...")
    finally:
        watcher.stop()
        initializer.shutdown()
        print("[Main] Exiting.")


if __name__ == "__main__":
    main()
