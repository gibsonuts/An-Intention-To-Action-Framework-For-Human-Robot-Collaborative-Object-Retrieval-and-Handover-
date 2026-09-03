#!/usr/bin/env python3
from __future__ import annotations

import base64
import io
import json
import sys
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox, ttk
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hardware.camera_rs_client import NetworkRealSenseCamera, _check_server_reachable


RECORDS_DIR = ROOT / "data" / "records"
HARDWARE_CONFIG = ROOT / "hardware" / "config" / "config.yaml"
WATCHER_CONFIG = ROOT / "llm" / "config" / "vlm_watcher.yaml"


def ensure_records_dir() -> None:
    RECORDS_DIR.mkdir(parents=True, exist_ok=True)


def load_camera_defaults() -> Dict[str, Any]:
    defaults = {
        "server_ip": "192.168.10.100",
        "port": 5552,
        "topic_prefix": "realsense",
        "camera_id": "camera_fixed",
    }
    if not HARDWARE_CONFIG.exists():
        return defaults
    try:
        with HARDWARE_CONFIG.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        cam_cfg = cfg.get("camera_fixed", {}) or {}
        defaults["server_ip"] = cam_cfg.get("network_camera_ip", defaults["server_ip"])
        defaults["port"] = int(cam_cfg.get("network_camera_port", defaults["port"]))
    except Exception:
        pass
    return defaults


def resolve_camera_topic(camera_name_or_id: str) -> str:
    camera_name_or_id = str(camera_name_or_id or "").strip()
    if not camera_name_or_id:
        return ""
    if not HARDWARE_CONFIG.exists():
        return camera_name_or_id
    try:
        with HARDWARE_CONFIG.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        cam_cfg = cfg.get(camera_name_or_id)
        if isinstance(cam_cfg, dict):
            serial = str(cam_cfg.get("serial", "")).strip()
            if serial:
                return serial
    except Exception:
        pass
    return camera_name_or_id


def visualize_depth(depth_mm_u16: np.ndarray) -> np.ndarray:
    depth_m = depth_mm_u16.astype(np.float32) / 1000.0
    vis = np.clip((depth_m - 0.2) / (3.0 - 0.2), 0, 1)
    vis_u8 = (vis * 255).astype(np.uint8)
    return cv2.applyColorMap(vis_u8, cv2.COLORMAP_JET)


def fit_bgr_to_tk(image_bgr: np.ndarray, width: int, height: int) -> ImageTk.PhotoImage:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    img.thumbnail((width, height), Image.Resampling.LANCZOS)
    return ImageTk.PhotoImage(img)


@dataclass
class FramePrediction:
    frame_index: int
    timestamp_sec: float
    predicted_object: Optional[str]
    predicted_objects: List[str]
    reason: str
    raw_tool_calls: List[dict]
    response_content: str
    request_debug: Dict[str, Any]
    response_debug: Dict[str, Any]


@dataclass
class InferenceQueueItem:
    label: str
    expected_objects: List[str]
    model: str
    api_url: str
    record_dir: str
    record_name: str
    created_at: str


@dataclass
class InferenceResultSummary:
    created_at: str
    label: str
    expected_objects: List[str]
    analyzed_frames: int
    correct_frames: int
    true_positive: int
    false_positive: int
    false_negative: int
    ratio_text: str
    percentage: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    f1_score: Optional[float]
    result_path: Optional[str]
    metrics_path: Optional[str]


class RecordSession:
    def __init__(self, record_dir: Path, fps: float) -> None:
        self.record_dir = record_dir
        self.rgb_dir = record_dir / "rgb"
        self.depth_dir = record_dir / "depth"
        self.fps = float(fps)
        self.frame_count = 0
        self.timestamps_sec: List[float] = []
        self.started_at = datetime.now().isoformat(timespec="seconds")
        self.rgb_dir.mkdir(parents=True, exist_ok=True)
        self.depth_dir.mkdir(parents=True, exist_ok=True)

    def save_frame(self, color_rgb: np.ndarray, depth_mm: np.ndarray, timestamp_sec: float) -> None:
        rgb_bgr = color_rgb[:, :, ::-1]
        rgb_path = self.rgb_dir / f"{self.frame_count:06d}.jpg"
        depth_path = self.depth_dir / f"{self.frame_count:06d}.npy"
        ok = cv2.imwrite(str(rgb_path), rgb_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ok:
            raise RuntimeError(f"Failed to write RGB frame to {rgb_path}")
        np.save(depth_path, depth_mm.astype(np.uint16, copy=False))
        self.timestamps_sec.append(float(timestamp_sec))
        self.frame_count += 1

    def close(self, extra_meta: Optional[Dict[str, Any]] = None) -> Path:
        metadata = {
            "started_at": self.started_at,
            "ended_at": datetime.now().isoformat(timespec="seconds"),
            "fps": self.fps,
            "frame_count": self.frame_count,
            "timestamps_sec": self.timestamps_sec,
            "color_space": "rgb",
        }
        if extra_meta:
            metadata.update(extra_meta)
        meta_path = self.record_dir / "metadata.json"
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        return meta_path


class SavedRecord:
    def __init__(self, record_dir: Path) -> None:
        self.record_dir = record_dir
        self.metadata_path = record_dir / "metadata.json"
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata.json in {record_dir}")
        with self.metadata_path.open("r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        self.frame_count = int(self.metadata.get("frame_count", 0))
        self.fps = float(self.metadata.get("fps", 10.0))
        self.timestamps_sec = list(self.metadata.get("timestamps_sec", []))
        self.color_space = str(self.metadata.get("color_space", "")).strip().lower()
        self.frame_ids = self._discover_frame_ids()
        if self.frame_ids:
            self.frame_count = len(self.frame_ids)

    def _discover_frame_ids(self) -> List[int]:
        rgb_dir = self.record_dir / "rgb"
        depth_dir = self.record_dir / "depth"
        if not rgb_dir.is_dir() or not depth_dir.is_dir():
            return []

        def _collect_ids(folder: Path, ext: str) -> set[int]:
            ids: set[int] = set()
            for p in folder.glob(f"*.{ext}"):
                stem = p.stem.strip()
                if stem.isdigit():
                    ids.add(int(stem))
            return ids

        rgb_ids = _collect_ids(rgb_dir, "jpg")
        depth_ids = _collect_ids(depth_dir, "npy")
        common_ids = sorted(rgb_ids & depth_ids)
        return common_ids

    @property
    def name(self) -> str:
        return self.record_dir.name

    def relative_timestamp_sec(self, index: int) -> float:
        if index < len(self.timestamps_sec) and self.timestamps_sec:
            base = float(self.timestamps_sec[0])
            return max(0.0, float(self.timestamps_sec[index]) - base)
        if self.fps <= 0:
            return float(index)
        return float(index) / self.fps

    def rgb_path(self, index: int) -> Path:
        frame_id = self.frame_ids[index] if self.frame_ids else index
        return self.record_dir / "rgb" / f"{frame_id:06d}.jpg"

    def depth_path(self, index: int) -> Path:
        frame_id = self.frame_ids[index] if self.frame_ids else index
        return self.record_dir / "depth" / f"{frame_id:06d}.npy"

    def load_frame(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        color_bgr = cv2.imread(str(self.rgb_path(index)), cv2.IMREAD_COLOR)
        if color_bgr is None:
            raise RuntimeError(f"Failed to load RGB frame {index} from {self.record_dir}")
        depth_mm = np.load(self.depth_path(index))
        if depth_mm.dtype != np.uint16:
            depth_mm = depth_mm.astype(np.uint16, copy=False)
        if self.color_space == "rgb":
            color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        else:
            # Legacy recordings were saved before channel order was fixed.
            # Treat the loaded bytes as already-RGB to keep playback/inference correct.
            color_rgb = color_bgr.copy()
        return depth_mm, color_rgb


class VLMRecordAnalyzer:
    def __init__(
        self,
        api_url: str,
        config_override: Optional[Dict[str, Any]] = None,
        request_timeout_sec: float = 10.0,
    ) -> None:
        self.api_url = api_url
        self.request_timeout_sec = float(request_timeout_sec)
        self.model = "Qwen3-VL-30B-A3B-Instruct-Q4_K_M.gguf"
        self.system_prompt = ""
        self.extra_information = ""
        self.tools: List[dict] = []
        self.known_objects: List[str] = []
        self._known_objects_lower: Dict[str, str] = {}
        self._load_config(config_override=config_override)

    @staticmethod
    def load_watcher_config() -> Dict[str, Any]:
        if not WATCHER_CONFIG.exists():
            raise RuntimeError(f"Watcher config not found: {WATCHER_CONFIG}")
        with WATCHER_CONFIG.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _load_config(self, config_override: Optional[Dict[str, Any]] = None) -> None:
        cfg = dict(config_override or self.load_watcher_config())
        self.model = str(cfg.get("model", self.model))
        self.system_prompt = str(cfg.get("system_prompt", "")).strip()
        self.extra_information = str(cfg.get("extra_infomation", "")).strip()
        self.tools = list(cfg.get("tools", []) or [])
        self.known_objects = [str(x) for x in (cfg.get("known_objects", []) or [])]
        self._known_objects_lower = {name.lower(): name for name in self.known_objects}

    def _frame_to_data_url(self, color_rgb: np.ndarray) -> str:
        img = Image.fromarray(color_rgb)
        img.thumbnail((1280, 720), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        data = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{data}"

    def _call_llm(self, color_rgb: np.ndarray) -> Tuple[str, List[dict], Dict[str, Any], Dict[str, Any]]:
        known_objects_text = "\n".join(f"- {name}" for name in self.known_objects) or "- (none configured)"
        user_text = (
            "You are viewing a saved camera frame from the robot's fixed camera.\n"
            "Decide whether the image suggests one known object.\n"
            "Use the configured tools exactly as in the live watcher.\n\n"
            f"KNOWN_OBJECTS:\n{known_objects_text}\n\n"
            f"extra information about object suggestion criteria:\n{self.extra_information}"
        )
        image_data_url = self._frame_to_data_url(color_rgb)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ],
                },
            ],
            "temperature": 0.2,
            "max_tokens": -1,
            "stream": False,
            "tools": self.tools,
            "tool_choice": "auto",
        }
        request_debug = {
            "api_url": self.api_url,
            "model": self.model,
            "color_space": "rgb",
            "image_shape": list(color_rgb.shape),
            "system_prompt": self.system_prompt,
            "user_text": user_text,
            "known_objects": self.known_objects,
            "tools": self.tools,
            "payload_preview": {
                "model": payload["model"],
                "temperature": payload["temperature"],
                "max_tokens": payload["max_tokens"],
                "tool_choice": payload["tool_choice"],
                "messages": [
                    payload["messages"][0],
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_text},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"<omitted base64 png, {len(image_data_url)} chars>"},
                            },
                        ],
                    },
                ],
                "tools": self.tools,
            },
            "request_timeout_sec": self.request_timeout_sec,
        }
        response = requests.post(self.api_url, json=payload, timeout=(3.0, self.request_timeout_sec))
        response.raise_for_status()
        data = response.json()
        message = data["choices"][0]["message"]
        content = (message.get("content") or "").strip()
        tool_calls = message.get("tool_calls") or []
        response_debug = {
            "status_code": response.status_code,
            "message": message,
            "content": content,
            "tool_calls": tool_calls,
            "raw_json": data,
        }
        return content, tool_calls, request_debug, response_debug

    def _normalize_suggested_objects(self, raw_value: Any) -> List[str]:
        raw_items: List[str] = []
        if isinstance(raw_value, list):
            raw_items = [str(item).strip() for item in raw_value]
        elif isinstance(raw_value, str):
            raw_text = raw_value.strip()
            if raw_text:
                if "," in raw_text:
                    raw_items = [part.strip() for part in raw_text.split(",")]
                else:
                    raw_items = [raw_text]

        normalized: List[str] = []
        for item in raw_items:
            if not item:
                continue
            if item.lower() == "nothing":
                continue
            canonical = self._known_objects_lower.get(item.lower())
            if canonical and canonical not in normalized:
                normalized.append(canonical)
        return normalized

    def _extract_prediction(self, tool_calls: List[dict]) -> Tuple[List[str], str]:
        for tool_call in tool_calls:
            fn = ((tool_call or {}).get("function") or {}).get("name")
            if fn != "object_suggester":
                continue
            raw_args = ((tool_call or {}).get("function") or {}).get("arguments", "{}")
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args or {})
            except Exception:
                return [], f"Failed to parse tool args: {raw_args}"
            raw_object = args.get("object_suggestion", "")
            reason = str(args.get("reason", "")).strip()
            object_names = self._normalize_suggested_objects(raw_object)
            return object_names, reason
        return [], ""

    def predict(self, frame_index: int, timestamp_sec: float, color_rgb: np.ndarray) -> FramePrediction:
        content, tool_calls, request_debug, response_debug = self._call_llm(color_rgb)
        predicted_objects, reason = self._extract_prediction(tool_calls)
        return FramePrediction(
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            predicted_object=(predicted_objects[0] if predicted_objects else None),
            predicted_objects=predicted_objects,
            reason=reason,
            raw_tool_calls=tool_calls,
            response_content=content,
            request_debug=request_debug,
            response_debug=response_debug,
        )


class RealSenseRecorderGUI:
    def __init__(self, root: tk.Tk) -> None:
        ensure_records_dir()
        self.root = root
        self.root.title("RealSense Fixed Camera Recorder")
        self.root.geometry("1380x860")

        defaults = load_camera_defaults()
        self.server_ip_var = tk.StringVar(value=str(defaults["server_ip"]))
        self.port_var = tk.StringVar(value=str(defaults["port"]))
        self.topic_prefix_var = tk.StringVar(value=str(defaults["topic_prefix"]))
        self.camera_id_var = tk.StringVar(value=str(defaults["camera_id"]))
        self.record_name_var = tk.StringVar(value="")
        self.record_fps_var = tk.StringVar(value="10")
        self.status_var = tk.StringVar(value="Idle")
        self.api_url_var = tk.StringVar(value="http://192.168.10.103:1234/v1/chat/completions")
        self.model_var = tk.StringVar(value="Qwen3-VL-30B-A3B-Instruct-Q4_K_M.gguf")
        self.display_mode = "live"
        self.playing = False
        self.preview_job: Optional[str] = None
        self.playback_job: Optional[str] = None
        self._suppress_seek_callback = False
        self.camera: Optional[NetworkRealSenseCamera] = None
        self.current_session: Optional[RecordSession] = None
        self.current_record: Optional[SavedRecord] = None
        self.playback_index = 0
        self.last_record_time = 0.0
        self.latest_color_rgb: Optional[np.ndarray] = None
        self.latest_depth_mm: Optional[np.ndarray] = None
        self.latest_tk_rgb: Optional[ImageTk.PhotoImage] = None
        self.latest_tk_depth: Optional[ImageTk.PhotoImage] = None
        self.inference_thread: Optional[threading.Thread] = None
        self.inference_stop = threading.Event()
        self.live_inference_thread: Optional[threading.Thread] = None
        self.live_inference_stop = threading.Event()
        self.live_inference_interval_var = tk.StringVar(value="2.0")
        self.vlm_timeout_var = tk.StringVar(value="10")
        self._inference_run_id = 0
        self._live_inference_run_id = 0
        self.inference_queue: List[InferenceQueueItem] = []
        self.queue_thread: Optional[threading.Thread] = None
        self.result_history: List[InferenceResultSummary] = []
        self.object_counter: Counter[str] = Counter()
        self.predictions: List[FramePrediction] = []
        self.prompt_text_widget: Optional[tk.Text] = None
        self.extra_info_text_widget: Optional[tk.Text] = None
        self.tools_text_widget: Optional[tk.Text] = None
        self.known_objects_text_widget: Optional[tk.Text] = None
        self.model_combo: Optional[ttk.Combobox] = None
        self.live_prediction_label: Optional[ttk.Label] = None
        self.text_context_menu: Optional[tk.Menu] = None
        self.request_debug_widget: Optional[tk.Text] = None
        self.response_debug_widget: Optional[tk.Text] = None
        self.expected_objects_frame: Optional[ttk.Frame] = None
        self.expected_object_vars: Dict[str, tk.BooleanVar] = {}
        self.queue_listbox: Optional[tk.Listbox] = None
        self.results_listbox: Optional[tk.Listbox] = None
        self.result_details_widget: Optional[tk.Text] = None

        self._build_ui()
        self.load_watcher_config_into_editor()
        self.refresh_records()
        self.connect_camera()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill="both", expand=True)
        self._setup_text_editing_support()

        controls = ttk.LabelFrame(main, text="Camera / Recording", padding=10)
        controls.pack(fill="x")

        row1 = ttk.Frame(controls)
        row1.pack(fill="x", pady=4)
        ttk.Label(row1, text="Server IP").pack(side="left")
        ttk.Entry(row1, textvariable=self.server_ip_var, width=16).pack(side="left", padx=4)
        ttk.Label(row1, text="Port").pack(side="left")
        ttk.Entry(row1, textvariable=self.port_var, width=8).pack(side="left", padx=4)
        ttk.Label(row1, text="Topic").pack(side="left")
        ttk.Entry(row1, textvariable=self.topic_prefix_var, width=12).pack(side="left", padx=4)
        ttk.Label(row1, text="Camera").pack(side="left")
        ttk.Entry(row1, textvariable=self.camera_id_var, width=14).pack(side="left", padx=4)
        ttk.Button(row1, text="Reconnect", command=self.connect_camera).pack(side="left", padx=8)
        ttk.Label(row1, textvariable=self.status_var).pack(side="right")

        row2 = ttk.Frame(controls)
        row2.pack(fill="x", pady=4)
        ttk.Label(row2, text="Record name").pack(side="left")
        ttk.Entry(row2, textvariable=self.record_name_var, width=28).pack(side="left", padx=4)
        ttk.Label(row2, text="FPS").pack(side="left")
        ttk.Entry(row2, textvariable=self.record_fps_var, width=6).pack(side="left", padx=4)
        ttk.Button(row2, text="Start Recording", command=self.start_recording).pack(side="left", padx=8)
        ttk.Button(row2, text="Stop Recording", command=self.stop_recording).pack(side="left", padx=4)
        ttk.Button(row2, text="Open Records Folder", command=self.open_records_folder).pack(side="left", padx=8)

        content = ttk.Panedwindow(main, orient="horizontal")
        content.pack(fill="both", expand=True, pady=10)

        left = ttk.Frame(content)

        preview = ttk.LabelFrame(left, text="Preview / Playback", padding=10)
        preview.pack(fill="both", expand=True)

        image_row = ttk.Frame(preview)
        image_row.pack(fill="both", expand=True)
        self.rgb_label = ttk.Label(image_row, text="RGB preview unavailable", anchor="center")
        self.rgb_label.pack(side="left", fill="both", expand=True, padx=(0, 5))
        self.depth_label = ttk.Label(image_row, text="Depth preview unavailable", anchor="center")
        self.depth_label.pack(side="left", fill="both", expand=True, padx=(5, 0))

        playback = ttk.LabelFrame(left, text="Playback", padding=10)
        playback.pack(fill="x", pady=(10, 0))
        row3 = ttk.Frame(playback)
        row3.pack(fill="x")
        ttk.Button(row3, text="Load Selected", command=self.load_selected_record).pack(side="left")
        ttk.Button(row3, text="Show Live", command=self.show_live_view).pack(side="left", padx=4)
        ttk.Button(row3, text="Play", command=self.play_loaded_record).pack(side="left", padx=4)
        ttk.Button(row3, text="Pause", command=self.pause_playback).pack(side="left", padx=4)
        ttk.Button(row3, text="Prev", command=lambda: self.step_playback(-1)).pack(side="left", padx=4)
        ttk.Button(row3, text="Next", command=lambda: self.step_playback(1)).pack(side="left", padx=4)
        self.frame_scale = ttk.Scale(playback, from_=0, to=0, orient="horizontal", command=self.on_seek)
        self.frame_scale.pack(fill="x", pady=(8, 0))
        self.playback_info = ttk.Label(playback, text="No record loaded")
        self.playback_info.pack(fill="x", pady=(6, 0))

        right = ttk.Frame(content, width=460)
        right.pack_propagate(False)

        records_box = ttk.LabelFrame(right, text="Saved Records", padding=10)
        records_box.pack(fill="x")
        records_list_frame = ttk.Frame(records_box)
        records_list_frame.pack(fill="x", expand=False)
        self.records_list = tk.Listbox(records_list_frame, height=6)
        records_scroll = ttk.Scrollbar(records_list_frame, orient="vertical", command=self.records_list.yview)
        self.records_list.configure(yscrollcommand=records_scroll.set)
        self.records_list.grid(row=0, column=0, sticky="nsew")
        records_scroll.grid(row=0, column=1, sticky="ns")
        records_list_frame.rowconfigure(0, weight=1)
        records_list_frame.columnconfigure(0, weight=1)
        ttk.Button(records_box, text="Refresh", command=self.refresh_records).pack(fill="x", pady=(8, 0))

        right_panes = ttk.Panedwindow(right, orient="vertical")
        right_panes.pack(fill="both", expand=True, pady=(10, 0))

        tabs = ttk.Notebook(right_panes)

        inference_tab = ttk.Frame(tabs, padding=10)
        results_tab = ttk.Frame(tabs, padding=10)
        vlm_tab = ttk.Frame(tabs, padding=10)
        debug_tab = ttk.Frame(tabs, padding=10)
        tabs.add(inference_tab, text="Inference")
        tabs.add(results_tab, text="Results")
        tabs.add(vlm_tab, text="VLM Config")
        tabs.add(debug_tab, text="Debug")
        right_panes.add(tabs, weight=3)

        inference_box = ttk.LabelFrame(inference_tab, text="Inference", padding=10)
        inference_box.pack(fill="both", expand=True)
        ttk.Label(inference_box, text="Watcher API URL").pack(anchor="w")
        ttk.Entry(inference_box, textvariable=self.api_url_var).pack(fill="x", pady=(0, 6))
        timeout_row = ttk.Frame(inference_box)
        timeout_row.pack(fill="x", pady=(0, 6))
        ttk.Label(timeout_row, text="VLM timeout (s)").pack(side="left")
        ttk.Entry(timeout_row, textvariable=self.vlm_timeout_var, width=8).pack(side="left", padx=6)
        ttk.Label(inference_box, text="Expected objects for scoring").pack(anchor="w")
        self.expected_objects_frame = ttk.Frame(inference_box)
        self.expected_objects_frame.pack(fill="x", pady=(0, 8))
        queue_actions = ttk.Frame(inference_box)
        queue_actions.pack(fill="x", pady=(0, 8))
        ttk.Button(queue_actions, text="Run Inference On Loaded Record", command=self.run_inference).pack(side="left")
        ttk.Button(queue_actions, text="Add Test To Queue", command=self.add_current_test_to_queue).pack(side="left", padx=6)
        ttk.Button(queue_actions, text="Run Queue", command=self.run_inference_queue).pack(side="left")
        ttk.Button(queue_actions, text="Clear Queue", command=self.clear_inference_queue).pack(side="left", padx=6)
        ttk.Button(inference_box, text="Stop Inference", command=self.stop_inference).pack(fill="x")
        ttk.Label(inference_box, text="Queued tests").pack(anchor="w", pady=(8, 0))
        self.queue_listbox = tk.Listbox(inference_box, height=5)
        self.queue_listbox.pack(fill="x", pady=(0, 6))
        results_box = ttk.LabelFrame(results_tab, text="Results History", padding=10)
        results_box.pack(fill="both", expand=True)
        self.results_listbox = tk.Listbox(results_box, height=10)
        self.results_listbox.pack(fill="x", pady=(0, 6))
        self.results_listbox.bind("<<ListboxSelect>>", self.on_result_selected)
        ttk.Button(results_box, text="Save Results History", command=self.save_results_history).pack(fill="x")
        ttk.Button(results_box, text="Clear Results History", command=self.clear_results_history).pack(fill="x")

        ttk.Label(results_box, text="Result details (select an item above)").pack(anchor="w", pady=(8, 0))
        details_frame = ttk.Frame(results_box)
        details_frame.pack(fill="both", expand=True, pady=(0, 6))
        self.result_details_widget = tk.Text(details_frame, height=10, wrap="none")
        details_y = ttk.Scrollbar(details_frame, orient="vertical", command=self.result_details_widget.yview)
        details_x = ttk.Scrollbar(details_frame, orient="horizontal", command=self.result_details_widget.xview)
        self.result_details_widget.configure(yscrollcommand=details_y.set, xscrollcommand=details_x.set)
        self.result_details_widget.grid(row=0, column=0, sticky="nsew")
        details_y.grid(row=0, column=1, sticky="ns")
        details_x.grid(row=1, column=0, sticky="ew")
        details_frame.rowconfigure(0, weight=1)
        details_frame.columnconfigure(0, weight=1)
        live_row = ttk.Frame(inference_box)
        live_row.pack(fill="x", pady=(8, 0))
        ttk.Label(live_row, text="Live interval (s)").pack(side="left")
        ttk.Entry(live_row, textvariable=self.live_inference_interval_var, width=8).pack(side="left", padx=6)
        ttk.Button(live_row, text="Start Live Inference", command=self.start_live_inference).pack(side="left", padx=4)
        ttk.Button(live_row, text="Stop Live Inference", command=self.stop_live_inference).pack(side="left", padx=4)
        self.live_prediction_label = ttk.Label(inference_box, text="Live prediction: idle", justify="left")
        self.live_prediction_label.pack(fill="x", pady=(6, 0))
        self.score_label = ttk.Label(inference_box, text="Score: not run")
        self.score_label.pack(fill="x", pady=(8, 0))
        self.counter_label = ttk.Label(inference_box, text="Object counts: none", justify="left")
        self.counter_label.pack(fill="x", pady=(6, 0))
        self.metrics_label = ttk.Label(inference_box, text="Precision/Recall/F1: not run", justify="left")
        self.metrics_label.pack(fill="x", pady=(6, 0))

        editor_box = ttk.LabelFrame(vlm_tab, text="VLM Prompt / Tools", padding=10)
        editor_box.pack(fill="both", expand=True)
        editor_actions = ttk.Frame(editor_box)
        editor_actions.pack(fill="x", pady=(0, 6))
        ttk.Button(editor_actions, text="Reload YAML", command=self.load_watcher_config_into_editor).pack(side="left")
        ttk.Button(editor_actions, text="Save YAML", command=self.save_watcher_config_from_editor).pack(side="left", padx=6)

        model_row = ttk.Frame(editor_box)
        model_row.pack(fill="x", pady=(0, 6))
        ttk.Label(model_row, text="Model").pack(side="left")
        ttk.Button(model_row, text="Refresh Models", command=self.refresh_studio_models).pack(side="right")
        self.model_combo = ttk.Combobox(editor_box, textvariable=self.model_var, values=[], state="normal")
        self.model_combo.pack(fill="x", pady=(0, 6))

        ttk.Label(editor_box, text="System prompt").pack(anchor="w")
        self.prompt_text_widget = tk.Text(editor_box, height=6, wrap="word")
        self.prompt_text_widget.pack(fill="x", pady=(0, 6))

        ttk.Label(editor_box, text="Extra information").pack(anchor="w")
        self.extra_info_text_widget = tk.Text(editor_box, height=5, wrap="word")
        self.extra_info_text_widget.pack(fill="x", pady=(0, 6))

        ttk.Label(editor_box, text="Known objects (one per line)").pack(anchor="w")
        self.known_objects_text_widget = tk.Text(editor_box, height=5, wrap="word")
        self.known_objects_text_widget.pack(fill="x", pady=(0, 6))

        ttk.Label(editor_box, text="Tools YAML").pack(anchor="w")
        self.tools_text_widget = tk.Text(editor_box, height=12, wrap="none")
        self.tools_text_widget.pack(fill="both", expand=True)

        debug_box = ttk.LabelFrame(debug_tab, text="VLM Request / Response", padding=10)
        debug_box.pack(fill="both", expand=True)
        ttk.Label(debug_box, text="Last request").pack(anchor="w")
        self.request_debug_widget = tk.Text(debug_box, height=16, wrap="none")
        self.request_debug_widget.pack(fill="both", expand=True, pady=(0, 8))
        ttk.Label(debug_box, text="Last response").pack(anchor="w")
        self.response_debug_widget = tk.Text(debug_box, height=16, wrap="none")
        self.response_debug_widget.pack(fill="both", expand=True)

        log_box = ttk.LabelFrame(right_panes, text="Log", padding=10)
        self.log_text = tk.Text(log_box, height=16, wrap="word")
        self.log_text.pack(fill="both", expand=True)
        right_panes.add(log_box, weight=2)

        content.add(left, weight=3)
        content.add(right, weight=2)

        for widget in [
            self.prompt_text_widget,
            self.extra_info_text_widget,
            self.known_objects_text_widget,
            self.tools_text_widget,
            self.request_debug_widget,
            self.response_debug_widget,
            self.result_details_widget,
        ]:
            self._bind_text_shortcuts(widget)

    def log(self, message: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert("end", f"[{stamp}] {message}\n")
        self.log_text.see("end")

    def connect_camera(self) -> None:
        self.stop_preview_loop()
        if self.camera is not None:
            try:
                self.camera.stop()
            except Exception:
                pass
            self.camera = None
        try:
            server_ip = self.server_ip_var.get().strip()
            port = int(self.port_var.get().strip())
            topic_prefix = self.topic_prefix_var.get().strip()
            raw_camera_id = self.camera_id_var.get().strip()
            resolved_camera_id = resolve_camera_topic(raw_camera_id)
            if not _check_server_reachable(server_ip, port, topic_prefix):
                raise RuntimeError(f"No stream received from tcp://{server_ip}:{port} with topic '{topic_prefix}/'.")
            self.camera = NetworkRealSenseCamera(
                intrinsics=None,
                server_ip=server_ip,
                port=port,
                topic_prefix=topic_prefix,
                camera_id=resolved_camera_id or None,
            )
            self.camera.start()
            self.display_mode = "live"
            self.status_var.set("Camera connected")
            if raw_camera_id and raw_camera_id != resolved_camera_id:
                self.log(f"Resolved camera topic '{raw_camera_id}' -> '{resolved_camera_id}'.")
            self.log("Connected to fixed RealSense client stream.")
            self.start_preview_loop()
        except Exception as e:
            self.status_var.set("Camera connection failed")
            self.log(f"Camera connection failed: {e}")
            messagebox.showerror("Camera connection failed", str(e))

    def start_preview_loop(self) -> None:
        self.preview_job = self.root.after(30, self.update_preview)

    def stop_preview_loop(self) -> None:
        if self.preview_job is not None:
            self.root.after_cancel(self.preview_job)
            self.preview_job = None

    def update_preview(self) -> None:
        try:
            if self.camera is not None:
                depth_mm, color_rgb = self.camera.get_rgbd()
                self.latest_depth_mm = depth_mm
                self.latest_color_rgb = color_rgb
                if self.current_session is not None:
                    self._maybe_save_live_frame(depth_mm, color_rgb)
                if self.display_mode == "live":
                    self.render_frame(depth_mm, color_rgb)
        except Exception:
            pass
        finally:
            self.preview_job = self.root.after(30, self.update_preview)

    def _maybe_save_live_frame(self, depth_mm: np.ndarray, color_rgb: np.ndarray) -> None:
        if self.current_session is None:
            return
        interval = 1.0 / max(self.current_session.fps, 0.1)
        now = time.time()
        if now - self.last_record_time < interval:
            return
        timestamp_sec = now
        self.current_session.save_frame(color_rgb, depth_mm, timestamp_sec)
        self.last_record_time = now
        self.status_var.set(f"Recording {self.current_session.frame_count} frames")

    def render_frame(self, depth_mm: np.ndarray, color_rgb: np.ndarray) -> None:
        color_bgr = color_rgb[:, :, ::-1].copy()
        depth_bgr = visualize_depth(depth_mm)
        self.latest_tk_rgb = fit_bgr_to_tk(color_bgr, 620, 420)
        self.latest_tk_depth = fit_bgr_to_tk(depth_bgr, 620, 420)
        self.rgb_label.configure(image=self.latest_tk_rgb, text="")
        self.depth_label.configure(image=self.latest_tk_depth, text="")

    def start_recording(self) -> None:
        if self.camera is None:
            messagebox.showerror("No camera", "Connect to the camera first.")
            return
        if self.current_session is not None:
            messagebox.showwarning("Already recording", "Stop the current recording first.")
            return
        try:
            fps = float(self.record_fps_var.get().strip())
            if fps <= 0:
                raise ValueError("FPS must be positive.")
        except Exception as e:
            messagebox.showerror("Invalid FPS", str(e))
            return

        name = self.record_name_var.get().strip() or datetime.now().strftime("record_%Y%m%d_%H%M%S")
        safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)
        record_dir = RECORDS_DIR / safe_name
        if record_dir.exists():
            messagebox.showerror("Record exists", f"{record_dir.name} already exists.")
            return

        self.current_session = RecordSession(record_dir=record_dir, fps=fps)
        self.display_mode = "live"
        self.last_record_time = 0.0
        self.status_var.set("Recording")
        self.log(f"Recording started: {record_dir}")

    def stop_recording(self) -> None:
        if self.current_session is None:
            return
        try:
            meta_path = self.current_session.close(
                {
                    "camera_id": self.camera_id_var.get().strip(),
                    "server_ip": self.server_ip_var.get().strip(),
                    "port": int(self.port_var.get().strip()),
                    "topic_prefix": self.topic_prefix_var.get().strip(),
                }
            )
            self.log(f"Recording saved: {meta_path.parent}")
        finally:
            self.current_session = None
            self.status_var.set("Idle")
            self.refresh_records()

    def open_records_folder(self) -> None:
        chosen = filedialog.askopenfilename(
            initialdir=str(RECORDS_DIR),
            title="Records folder",
            filetypes=[("Metadata", "metadata.json"), ("All files", "*.*")],
        )
        if chosen:
            self.log(f"Selected file in records: {chosen}")

    def refresh_records(self) -> None:
        ensure_records_dir()
        self.records_list.delete(0, "end")
        record_dirs = sorted([p for p in RECORDS_DIR.iterdir() if p.is_dir()], reverse=True)
        for path in record_dirs:
            self.records_list.insert("end", path.name)

    def load_selected_record(self) -> None:
        selection = self.records_list.curselection()
        if not selection:
            messagebox.showwarning("No record selected", "Select a record first.")
            return
        name = self.records_list.get(selection[0])
        record_dir = RECORDS_DIR / name
        try:
            self.current_record = SavedRecord(record_dir)
            if self.current_record.frame_count <= 0:
                raise RuntimeError(f"{record_dir.name} has no recorded frames.")
            self.display_mode = "playback"
            self.playback_index = 0
            self.frame_scale.configure(to=max(self.current_record.frame_count - 1, 0))
            self.update_expected_objects()
            self.show_playback_frame(0)
            self.log(f"Loaded record: {record_dir}")
        except Exception as e:
            messagebox.showerror("Load failed", str(e))

    def update_expected_objects(self) -> None:
        previous = {name for name, var in self.expected_object_vars.items() if var.get()}
        values: List[str] = ["none"]
        try:
            analyzer = VLMRecordAnalyzer(self.api_url_var.get().strip(), config_override=self.get_vlm_config_from_editor())
            values.extend([obj for obj in analyzer.known_objects if obj.lower() != "nothing"])
        except Exception:
            pass
        self.expected_object_vars = {}
        if self.expected_objects_frame is None:
            return
        for child in self.expected_objects_frame.winfo_children():
            child.destroy()
        for idx, value in enumerate(values):
            var = tk.BooleanVar(value=value in previous)
            self.expected_object_vars[value] = var
            ttk.Checkbutton(
                self.expected_objects_frame,
                text=value,
                variable=var,
            ).grid(row=idx // 2, column=idx % 2, sticky="w", padx=(0, 12), pady=2)

    def get_selected_expected_objects(self) -> List[str]:
        return [name for name, var in self.expected_object_vars.items() if var.get()]

    def _make_queue_item_from_current_state(self) -> InferenceQueueItem:
        expected_objects = self.get_selected_expected_objects()
        if not expected_objects:
            raise RuntimeError("Select at least one scoring object before queuing a test.")
        if self.current_record is None:
            raise RuntimeError("Load a saved record before queuing a test.")
        label = ", ".join(expected_objects)
        return InferenceQueueItem(
            label=label,
            expected_objects=list(expected_objects),
            model=self.model_var.get().strip(),
            api_url=self.api_url_var.get().strip(),
            record_dir=str(self.current_record.record_dir),
            record_name=self.current_record.name,
            created_at=datetime.now().isoformat(timespec="seconds"),
        )

    def _refresh_queue_listbox(self) -> None:
        if self.queue_listbox is None:
            return
        self.queue_listbox.delete(0, "end")
        for idx, item in enumerate(self.inference_queue, start=1):
            self.queue_listbox.insert("end", f"{idx}. {item.record_name} | {item.label} | {item.model}")

    def add_current_test_to_queue(self) -> None:
        if self.current_record is None:
            messagebox.showwarning("No record loaded", "Load a record first.")
            return
        try:
            item = self._make_queue_item_from_current_state()
        except Exception as e:
            messagebox.showwarning("Queue item invalid", str(e))
            return
        self.inference_queue.append(item)
        self._refresh_queue_listbox()
        self.log(f"Queued test: {item.label}")

    def clear_inference_queue(self) -> None:
        self.inference_queue = []
        self._refresh_queue_listbox()
        self.log("Inference queue cleared.")

    def _refresh_results_listbox(self) -> None:
        if self.results_listbox is None:
            return
        self.results_listbox.delete(0, "end")
        for idx, item in enumerate(self.result_history, start=1):
            self.results_listbox.insert("end", f"{idx}. {item.created_at} | {item.label} | {item.ratio_text}")

    def clear_results_history(self) -> None:
        self.result_history = []
        self._refresh_results_listbox()
        self._set_text_widget(self.result_details_widget, "")
        self.log("Results history cleared.")

    def save_results_history(self) -> None:
        if not self.result_history:
            messagebox.showwarning("No results", "No results history to save.")
            return
        out_path = RECORDS_DIR / f"results_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        payload = {
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "count": len(self.result_history),
            "results": [asdict(item) for item in self.result_history],
        }
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        self.log(f"Results history saved to {out_path}")

    def on_result_selected(self, event: tk.Event) -> None:
        if self.results_listbox is None:
            return
        selection = self.results_listbox.curselection()
        if not selection:
            return
        idx = selection[0]
        if idx < 0 or idx >= len(self.result_history):
            return
        item = self.result_history[idx]
        percent_txt = f"{item.percentage:.2f}%" if item.percentage is not None else "n/a"
        self.score_label.configure(text=f"Score ({item.label}): {item.ratio_text} ({percent_txt})")
        precision_txt = f"{item.precision:.4f}" if item.precision is not None else "n/a"
        recall_txt = f"{item.recall:.4f}" if item.recall is not None else "n/a"
        f1_txt = f"{item.f1_score:.4f}" if item.f1_score is not None else "n/a"
        self.metrics_label.configure(
            text=(
                f"Target: {item.label} | TP={item.true_positive} FP={item.false_positive} "
                f"FN={item.false_negative} | Precision={precision_txt} Recall={recall_txt} F1={f1_txt}"
            )
        )
        self._set_text_widget(self.result_details_widget, json.dumps(asdict(item), indent=2))
        if item.result_path:
            self.log(f"Selected result: {item.result_path}")
        if item.metrics_path:
            self.log(f"Selected metrics: {item.metrics_path}")

    def show_playback_frame(self, index: int) -> None:
        if self.current_record is None:
            return
        self.display_mode = "playback"
        index = max(0, min(index, self.current_record.frame_count - 1))
        self.playback_index = index
        depth_mm, color_rgb = self.current_record.load_frame(index)
        self.render_frame(depth_mm, color_rgb)
        self._suppress_seek_callback = True
        try:
            self.frame_scale.set(index)
        finally:
            self._suppress_seek_callback = False
        ts = None
        if self.current_record is not None:
            ts = self.current_record.relative_timestamp_sec(index)
        ts_txt = f"{ts:.3f}" if isinstance(ts, (int, float)) else "n/a"
        self.playback_info.configure(
            text=f"{self.current_record.name}: frame {index + 1}/{self.current_record.frame_count} | fps {self.current_record.fps:.2f} | t={ts_txt}"
        )

    def on_seek(self, value: str) -> None:
        if self.current_record is None:
            return
        if self._suppress_seek_callback:
            return
        self.pause_playback()
        self.show_playback_frame(int(float(value)))

    def play_loaded_record(self) -> None:
        if self.current_record is None:
            return
        self.display_mode = "playback"
        self.playing = True
        self._schedule_next_playback_frame()

    def pause_playback(self) -> None:
        self.playing = False
        if self.playback_job is not None:
            self.root.after_cancel(self.playback_job)
            self.playback_job = None

    def _schedule_next_playback_frame(self) -> None:
        if not self.playing or self.current_record is None:
            return
        self.show_playback_frame(self.playback_index)
        self.playback_index += 1
        if self.playback_index >= self.current_record.frame_count:
            self.playing = False
            return
        delay_ms = int(1000.0 / max(self.current_record.fps, 0.1))
        self.playback_job = self.root.after(delay_ms, self._schedule_next_playback_frame)

    def step_playback(self, delta: int) -> None:
        if self.current_record is None:
            return
        self.pause_playback()
        self.show_playback_frame(self.playback_index + delta)

    def run_inference(self) -> None:
        if self.current_record is None:
            messagebox.showwarning("No record loaded", "Load a record first.")
            return
        if self.inference_thread is not None and self.inference_thread.is_alive() and not self.inference_stop.is_set():
            messagebox.showwarning("Inference running", "Stop the current inference first.")
            return
        try:
            queue_item = self._make_queue_item_from_current_state()
        except Exception as e:
            messagebox.showwarning("No scoring target", str(e))
            return
        self.predictions = []
        self.object_counter = Counter()
        self.display_mode = "inference"
        self.score_label.configure(text="Score: running")
        self.counter_label.configure(text="Object counts: running")
        self.metrics_label.configure(text="Precision/Recall/F1: running")
        self.inference_stop.clear()
        self._inference_run_id += 1
        self.inference_thread = threading.Thread(
            target=self._run_inference_worker,
            kwargs={"queue_item": queue_item, "save_suffix": None},
            daemon=True,
        )
        self.inference_thread.start()

    def run_inference_queue(self) -> None:
        if self.current_record is None:
            messagebox.showwarning("No record loaded", "Load a record first.")
            return
        if not self.inference_queue:
            messagebox.showwarning("Queue empty", "Add at least one test to the queue first.")
            return
        if self.inference_thread is not None and self.inference_thread.is_alive() and not self.inference_stop.is_set():
            messagebox.showwarning("Inference running", "Stop the current inference first.")
            return
        self.predictions = []
        self.object_counter = Counter()
        self.display_mode = "inference"
        self.score_label.configure(text="Score: queue running")
        self.counter_label.configure(text=f"Object counts: queued {len(self.inference_queue)} test(s)")
        self.metrics_label.configure(text="Precision/Recall/F1: queue running")
        self.inference_stop.clear()
        self._inference_run_id += 1
        queued_items = list(self.inference_queue)
        self.inference_thread = threading.Thread(
            target=self._run_inference_queue_worker,
            kwargs={"queue_items": queued_items},
            daemon=True,
        )
        self.inference_thread.start()

    def stop_inference(self) -> None:
        self.inference_stop.set()
        self._inference_run_id += 1
        self.score_label.configure(text="Score: stopped")
        self.metrics_label.configure(text="Precision/Recall/F1: stopped")
        self.log("Inference stop requested.")

    def start_live_inference(self) -> None:
        if self.camera is None:
            messagebox.showwarning("No camera", "Connect to the live camera first.")
            return
        if self.live_inference_thread is not None and self.live_inference_thread.is_alive() and not self.live_inference_stop.is_set():
            messagebox.showwarning("Live inference running", "Stop the current live inference first.")
            return
        try:
            interval = float(self.live_inference_interval_var.get().strip())
            if interval <= 0:
                raise ValueError("Interval must be positive.")
        except Exception as e:
            messagebox.showerror("Invalid interval", str(e))
            return
        self.live_inference_stop.clear()
        self._live_inference_run_id += 1
        self.live_inference_thread = threading.Thread(target=self._run_live_inference_worker, daemon=True)
        self.live_inference_thread.start()
        if self.live_prediction_label is not None:
            self.live_prediction_label.configure(text="Live prediction: running")
        self.log(f"Started live inference at {interval:.2f}s interval.")

    def stop_live_inference(self) -> None:
        self.live_inference_stop.set()
        self._live_inference_run_id += 1
        if self.live_prediction_label is not None:
            self.live_prediction_label.configure(text="Live prediction: stopped")
        self.log("Live inference stop requested.")

    def _run_inference_worker(
        self,
        queue_item: InferenceQueueItem,
        save_suffix: Optional[str],
        *,
        record_override: Optional[SavedRecord] = None,
        clear_thread_on_exit: bool = True,
    ) -> None:
        record = record_override or self.current_record
        if record is None:
            return
        run_id = self._inference_run_id
        try:
            timeout_sec = float(self.vlm_timeout_var.get().strip())
            config_override = self.get_vlm_config_from_editor()
            config_override["model"] = queue_item.model
            analyzer = VLMRecordAnalyzer(
                queue_item.api_url,
                config_override=config_override,
                request_timeout_sec=timeout_sec,
            )
        except Exception as e:
            self.root.after(0, lambda msg=str(e): messagebox.showerror("Inference setup failed", msg))
            if clear_thread_on_exit:
                self.inference_thread = None
            return

        expected_objects = list(queue_item.expected_objects)
        self.root.after(0, self.update_expected_objects)

        correct_frames = 0
        analyzed_frames = 0
        true_positive = 0
        false_positive = 0
        false_negative = 0
        local_counter: Counter[str] = Counter()
        local_predictions: List[FramePrediction] = []

        for frame_index in range(record.frame_count):
            if self.inference_stop.is_set() or run_id != self._inference_run_id:
                break
            depth_mm, color_rgb = record.load_frame(frame_index)
            timestamp_sec = record.relative_timestamp_sec(frame_index)
            try:
                prediction = analyzer.predict(frame_index, timestamp_sec, color_rgb)
            except Exception as e:
                if self.inference_stop.is_set() or run_id != self._inference_run_id:
                    break
                self.root.after(0, lambda msg=f"Frame {frame_index} inference failed: {e}": self.log(msg))
                continue
            if self.inference_stop.is_set() or run_id != self._inference_run_id:
                break

            local_predictions.append(prediction)
            analyzed_frames += 1

            for predicted_object in prediction.predicted_objects:
                local_counter[predicted_object] += 1

            if self._is_target_prediction(prediction.predicted_objects, expected_objects):
                true_positive += 1
                correct_frames += 1
            else:
                false_negative += 1
                if prediction.predicted_objects:
                    false_positive += 1

            if run_id == self._inference_run_id and not self.inference_stop.is_set():
                self.root.after(0, lambda idx=frame_index, d=depth_mm, c=color_rgb, p=prediction: self._update_inference_preview(idx, d, c, p))
                self.root.after(
                    0,
                    lambda idx=frame_index, preds=prediction.predicted_objects, reason=prediction.reason: self.log(
                        f"Inference frame {idx}: objects={preds or ['none']} reason={reason or '-'}"
                    ),
                )

        self.predictions = local_predictions
        self.object_counter = local_counter
        if run_id == self._inference_run_id and not self.inference_stop.is_set():
            self.root.after(
                0,
                lambda: self._finish_inference_ui(
                    analyzed_frames,
                    correct_frames,
                    true_positive,
                    false_positive,
                    false_negative,
                    expected_objects,
                    save_suffix,
                    record,
                ),
            )
        if clear_thread_on_exit:
            self.inference_thread = None

    def _run_inference_queue_worker(self, queue_items: List[InferenceQueueItem]) -> None:
        total = len(queue_items)
        for idx, item in enumerate(queue_items, start=1):
            if self.inference_stop.is_set():
                break
            try:
                queue_record = SavedRecord(Path(item.record_dir))
            except Exception as e:
                self.root.after(0, lambda msg=str(e), rn=item.record_name: self.log(f"Queue record load failed ({rn}): {msg}"))
                continue
            save_suffix = f"queue_{idx:02d}"
            self.root.after(
                0,
                lambda i=idx, t=total, label=item.label, rn=item.record_name, rec=queue_record: self._set_queue_active_record(
                    i, t, rn, label, rec
                ),
            )
            self._run_inference_worker(
                item,
                save_suffix,
                record_override=queue_record,
                clear_thread_on_exit=False,
            )
            if self.inference_stop.is_set():
                break
        self.root.after(0, lambda: self.log("Inference queue finished."))
        self.inference_thread = None

    def _set_queue_active_record(
        self,
        idx: int,
        total: int,
        record_name: str,
        label: str,
        record: SavedRecord,
    ) -> None:
        self.current_record = record
        self.playback_index = 0
        self.frame_scale.configure(to=max(self.current_record.frame_count - 1, 0))
        self.show_playback_frame(0)
        self.score_label.configure(text=f"Queue {idx}/{total}: {record_name} | {label}")

    def _run_live_inference_worker(self) -> None:
        try:
            run_id = self._live_inference_run_id
            timeout_sec = float(self.vlm_timeout_var.get().strip())
            analyzer = VLMRecordAnalyzer(
                self.api_url_var.get().strip(),
                config_override=self.get_vlm_config_from_editor(),
                request_timeout_sec=timeout_sec,
            )
            interval = float(self.live_inference_interval_var.get().strip())
        except Exception as e:
            self.root.after(0, lambda msg=str(e): messagebox.showerror("Live inference setup failed", msg))
            self.live_inference_thread = None
            return

        while not self.live_inference_stop.is_set() and run_id == self._live_inference_run_id:
            try:
                if self.camera is None:
                    raise RuntimeError("Camera not connected.")
                depth_mm, color_rgb = self.camera.get_rgbd()
                prediction = analyzer.predict(frame_index=-1, timestamp_sec=time.time(), color_rgb=color_rgb)
                if self.live_inference_stop.is_set() or run_id != self._live_inference_run_id:
                    break
                self.root.after(
                    0,
                    lambda d=depth_mm, c=color_rgb, p=prediction: self._update_live_inference_ui(d, c, p),
                )
            except Exception as e:
                if self.live_inference_stop.is_set() or run_id != self._live_inference_run_id:
                    break
                self.root.after(0, lambda msg=str(e): self.log(f"Live inference failed: {msg}"))
            if self.live_inference_stop.wait(interval):
                break
        self.live_inference_thread = None

    def _update_live_inference_ui(self, depth_mm: np.ndarray, color_rgb: np.ndarray, prediction: FramePrediction) -> None:
        self.display_mode = "inference"
        self.render_frame(depth_mm, color_rgb)
        pred = ", ".join(prediction.predicted_objects) if prediction.predicted_objects else "none"
        reason = prediction.reason or prediction.response_content or "-"
        if self.live_prediction_label is not None:
            self.live_prediction_label.configure(text=f"Live prediction: {pred} | {reason}")
        self._set_debug_widgets(prediction)
        self.log(f"Live inference: objects={pred} reason={reason}")

    def _update_inference_preview(
        self,
        frame_index: int,
        depth_mm: np.ndarray,
        color_rgb: np.ndarray,
        prediction: Optional[FramePrediction] = None,
    ) -> None:
        self.render_frame(depth_mm, color_rgb)
        if self.current_record is not None:
            self.playback_index = frame_index
            self._suppress_seek_callback = True
            try:
                self.frame_scale.set(frame_index)
            finally:
                self._suppress_seek_callback = False
        if prediction is not None:
            self._set_debug_widgets(prediction)

    def _set_debug_widgets(self, prediction: FramePrediction) -> None:
        request_text = json.dumps(prediction.request_debug, indent=2)
        response_text = json.dumps(prediction.response_debug, indent=2)
        self._set_text_widget(self.request_debug_widget, request_text)
        self._set_text_widget(self.response_debug_widget, response_text)

    def _finish_inference_ui(
        self,
        analyzed_frames: int,
        correct_frames: int,
        true_positive: int,
        false_positive: int,
        false_negative: int,
        expected_objects: List[str],
        save_suffix: Optional[str],
        record: Optional[SavedRecord],
    ) -> None:
        self.display_mode = "playback" if self.current_record is not None else "live"
        target = ", ".join(expected_objects)
        ratio_text = "0/0"
        percentage_value: Optional[float] = None
        if analyzed_frames == 0:
            self.score_label.configure(text="Score: no frames analyzed")
        else:
            ratio_text = f"{correct_frames}/{analyzed_frames}"
            percentage_value = 100.0 * correct_frames / analyzed_frames
            self.score_label.configure(text=f"Score ({target}): {ratio_text} ({percentage_value:.2f}%)")

        if self.object_counter:
            parts = [f"{name}: {count}" for name, count in self.object_counter.most_common()]
            self.counter_label.configure(text="Object counts: " + ", ".join(parts))
        else:
            self.counter_label.configure(text="Object counts: none")

        precision = (true_positive / (true_positive + false_positive)) if (true_positive + false_positive) else None
        recall = (true_positive / (true_positive + false_negative)) if (true_positive + false_negative) else None
        f1_score = None
        if precision is not None and recall is not None and (precision + recall) > 0:
            f1_score = 2.0 * precision * recall / (precision + recall)
        metric_target = ", ".join(expected_objects)
        precision_txt = f"{precision:.4f}" if precision is not None else "n/a"
        recall_txt = f"{recall:.4f}" if recall is not None else "n/a"
        f1_txt = f"{f1_score:.4f}" if f1_score is not None else "n/a"
        self.metrics_label.configure(
            text=(
                f"Target: {metric_target} | TP={true_positive} FP={false_positive} "
                f"FN={false_negative} | Precision={precision_txt} Recall={recall_txt} F1={f1_txt}"
            )
        )

        result_path: Optional[str] = None
        metrics_path: Optional[str] = None
        if record is not None and self.predictions:
            metric_target_slug = "__".join(obj.replace(" ", "_") for obj in expected_objects)
            suffix = f"_{save_suffix}" if save_suffix else ""
            results = {
                "record": record.name,
                "analyzed_at": datetime.now().isoformat(timespec="seconds"),
                "expected_objects": expected_objects,
                "correct_frames": correct_frames,
                "analyzed_frames": analyzed_frames,
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "ratio": (ratio_text if analyzed_frames else None),
                "percentage": percentage_value,
                "object_counter": dict(self.object_counter),
                "predictions": [
                    {
                        "frame_index": p.frame_index,
                        "timestamp_sec": p.timestamp_sec,
                        "predicted_object": p.predicted_object,
                        "predicted_objects": p.predicted_objects,
                        "reason": p.reason,
                        "response_content": p.response_content,
                        "raw_tool_calls": p.raw_tool_calls,
                        "request_debug": p.request_debug,
                        "response_debug": p.response_debug,
                    }
                    for p in self.predictions
                ],
            }
            out_path = record.record_dir / f"inference_results{suffix}.json"
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            self.log(f"Inference results written to {out_path}")
            result_path = str(out_path)
            metrics_path = record.record_dir / f"inference_metrics_{metric_target_slug}{suffix}.json"
            with metrics_path.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "record": record.name,
                        "target_objects": expected_objects,
                        "analyzed_at": results["analyzed_at"],
                        "true_positive": true_positive,
                        "false_positive": false_positive,
                        "false_negative": false_negative,
                        "precision": precision,
                        "recall": recall,
                        "f1_score": f1_score,
                        "accuracy_ratio": results["ratio"],
                        "accuracy_percentage": results["percentage"],
                    },
                    f,
                    indent=2,
                )
            self.log(f"Inference metrics written to {metrics_path}")
            metrics_path = str(metrics_path)

        summary = InferenceResultSummary(
            created_at=datetime.now().isoformat(timespec="seconds"),
            label=target,
            expected_objects=list(expected_objects),
            analyzed_frames=analyzed_frames,
            correct_frames=correct_frames,
            true_positive=true_positive,
            false_positive=false_positive,
            false_negative=false_negative,
            ratio_text=ratio_text,
            percentage=percentage_value,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            result_path=result_path,
            metrics_path=metrics_path,
        )
        self.result_history.append(summary)
        self._refresh_results_listbox()

    def show_live_view(self) -> None:
        self.pause_playback()
        self.display_mode = "live"
        self.playback_info.configure(text="Live preview")

    def load_watcher_config_into_editor(self) -> None:
        try:
            cfg = VLMRecordAnalyzer.load_watcher_config()
        except Exception as e:
            messagebox.showerror("Load watcher config failed", str(e))
            return
        self.model_var.set(str(cfg.get("model", self.model_var.get())).strip())
        self._set_text_widget(self.prompt_text_widget, str(cfg.get("system_prompt", "")).strip())
        self._set_text_widget(self.extra_info_text_widget, str(cfg.get("extra_infomation", "")).strip())
        self._set_text_widget(
            self.known_objects_text_widget,
            "\n".join(str(x) for x in (cfg.get("known_objects", []) or [])),
        )
        tools_yaml = yaml.safe_dump(cfg.get("tools", []) or [], sort_keys=False)
        self._set_text_widget(self.tools_text_widget, tools_yaml)
        self.log(f"Loaded watcher config from {WATCHER_CONFIG}")
        self.update_expected_objects()
        self.refresh_studio_models()

    def save_watcher_config_from_editor(self) -> None:
        try:
            cfg = self.get_vlm_config_from_editor()
            with WATCHER_CONFIG.open("w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)
            self.log(f"Saved watcher config to {WATCHER_CONFIG}")
            self.update_expected_objects()
        except Exception as e:
            messagebox.showerror("Save watcher config failed", str(e))

    def refresh_studio_models(self) -> None:
        try:
            models = self._fetch_studio_models()
        except Exception as e:
            self.log(f"Model refresh failed: {e}")
            return
        if self.model_combo is not None:
            self.model_combo.configure(values=models)
        if models:
            current = self.model_var.get().strip()
            if current not in models:
                self.model_var.set(models[0])
        self.log(f"Loaded {len(models)} model(s) from Studio.")

    def get_vlm_config_from_editor(self) -> Dict[str, Any]:
        system_prompt = self._get_text_widget(self.prompt_text_widget).strip()
        extra_information = self._get_text_widget(self.extra_info_text_widget).strip()
        tools_raw = self._get_text_widget(self.tools_text_widget).strip()
        known_objects_raw = self._get_text_widget(self.known_objects_text_widget).strip()

        try:
            tools = yaml.safe_load(tools_raw) if tools_raw else []
        except Exception as e:
            raise RuntimeError(f"Tools YAML is invalid: {e}") from e
        if tools is None:
            tools = []
        if not isinstance(tools, list):
            raise RuntimeError("Tools YAML must parse to a list.")

        known_objects = [line.strip() for line in known_objects_raw.splitlines() if line.strip()]
        return {
            "model": self.model_var.get().strip() or "Qwen3-VL-30B-A3B-Instruct-Q4_K_M.gguf",
            "system_prompt": system_prompt,
            "extra_infomation": extra_information,
            "tools": tools,
            "known_objects": known_objects,
        }

    def _fetch_studio_models(self) -> List[str]:
        api_url = self.api_url_var.get().strip()
        if not api_url:
            raise RuntimeError("API URL is empty.")
        models_url = api_url
        if "/v1/" in api_url:
            prefix = api_url.split("/v1/", 1)[0]
            models_url = prefix + "/v1/models"
        else:
            models_url = api_url.rstrip("/") + "/v1/models"

        response = requests.get(models_url, timeout=10)
        response.raise_for_status()
        payload = response.json()
        items = payload.get("data", [])
        if not isinstance(items, list):
            raise RuntimeError(f"Unexpected models response from {models_url}")
        model_ids = []
        for item in items:
            if isinstance(item, dict):
                model_id = str(item.get("id", "")).strip()
                if model_id:
                    model_ids.append(model_id)
        return model_ids

    @staticmethod
    def _set_text_widget(widget: Optional[tk.Text], value: str) -> None:
        if widget is None:
            return
        widget.delete("1.0", "end")
        widget.insert("1.0", value)

    @staticmethod
    def _get_text_widget(widget: Optional[tk.Text]) -> str:
        if widget is None:
            return ""
        return widget.get("1.0", "end-1c")

    def _setup_text_editing_support(self) -> None:
        self.text_context_menu = tk.Menu(self.root, tearoff=0)
        self.text_context_menu.add_command(label="Cut", command=lambda: self._text_event_generate("<<Cut>>"))
        self.text_context_menu.add_command(label="Copy", command=lambda: self._text_event_generate("<<Copy>>"))
        self.text_context_menu.add_command(label="Paste", command=lambda: self._text_event_generate("<<Paste>>"))
        self.text_context_menu.add_separator()
        self.text_context_menu.add_command(label="Select All", command=self._select_all_text)

    def _bind_text_shortcuts(self, widget: Optional[tk.Text]) -> None:
        if widget is None:
            return
        widget.bind("<Control-a>", self._on_select_all_text)
        widget.bind("<Control-A>", self._on_select_all_text)
        widget.bind("<Button-3>", self._show_text_context_menu)

    def _show_text_context_menu(self, event: tk.Event) -> str:
        widget = event.widget
        try:
            widget.focus_set()
        except Exception:
            pass
        if self.text_context_menu is not None:
            self.text_context_menu.tk_popup(event.x_root, event.y_root)
        return "break"

    def _text_event_generate(self, sequence: str) -> None:
        widget = self.root.focus_get()
        if widget is not None:
            try:
                widget.event_generate(sequence)
            except Exception:
                pass

    def _on_select_all_text(self, event: tk.Event) -> str:
        widget = event.widget
        try:
            widget.tag_add("sel", "1.0", "end-1c")
            widget.mark_set("insert", "1.0")
            widget.see("insert")
        except Exception:
            pass
        return "break"

    def _select_all_text(self) -> None:
        widget = self.root.focus_get()
        if widget is None:
            return
        try:
            widget.tag_add("sel", "1.0", "end-1c")
            widget.mark_set("insert", "1.0")
            widget.see("insert")
        except Exception:
            pass

    @staticmethod
    def _is_target_prediction(predicted_objects: List[str], expected_objects: List[str]) -> bool:
        if not expected_objects:
            return False
        expected_set = set(expected_objects)
        if not predicted_objects and "none" in expected_set:
            return True
        return any(obj in expected_set for obj in predicted_objects)

    def on_close(self) -> None:
        self.stop_inference()
        self.stop_live_inference()
        self.pause_playback()
        self.stop_recording()
        self.stop_preview_loop()
        if self.camera is not None:
            try:
                self.camera.stop()
            except Exception:
                pass
        self.root.destroy()


def main() -> None:
    ensure_records_dir()
    root = tk.Tk()
    ttk.Style().theme_use("clam")
    RealSenseRecorderGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
