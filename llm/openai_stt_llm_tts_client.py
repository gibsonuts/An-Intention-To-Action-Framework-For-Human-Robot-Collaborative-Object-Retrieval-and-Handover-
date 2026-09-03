#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import os
import queue
import re
import tempfile
import threading
import time
from collections import deque
from typing import Any, Callable, Dict, Optional

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf
import yaml

try:
    import azure.cognitiveservices.speech as speechsdk
except ImportError:
    speechsdk = None

from commons.grasp_utils import check_path_exists

CFG_PATH = "config/config.yaml"


def pcm16_from_float32(x: np.ndarray) -> bytes:
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767).astype(np.int16).tobytes()


def float32_from_pcm16(b: bytes) -> np.ndarray:
    if not b:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(b, dtype=np.int16).astype(np.float32) / 32767.0


def rms_peak(x: np.ndarray) -> tuple[float, float]:
    if x.size == 0:
        return 0.0, 0.0
    rms = float(np.sqrt(np.mean(np.square(x))))
    peak = float(np.max(np.abs(x)))
    return rms, peak


def apply_fade(audio_f32: np.ndarray, sr: int, ms: int) -> np.ndarray:
    if audio_f32.size == 0 or ms <= 0:
        return audio_f32
    n = min(audio_f32.size, int(sr * (ms / 1000.0)))
    if n <= 1:
        return audio_f32
    y = audio_f32.copy()
    y[:n] *= np.linspace(0.0, 1.0, n, dtype=np.float32)
    y[-n:] *= np.linspace(1.0, 0.0, n, dtype=np.float32)
    return y


def resample_audio(audio_f32: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if audio_f32.size == 0 or src_sr == dst_sr:
        return audio_f32.astype(np.float32, copy=False)
    duration_s = audio_f32.size / float(src_sr)
    dst_len = max(1, int(round(duration_s * dst_sr)))
    src_x = np.linspace(0.0, 1.0, num=audio_f32.size, endpoint=False, dtype=np.float32)
    dst_x = np.linspace(0.0, 1.0, num=dst_len, endpoint=False, dtype=np.float32)
    return np.interp(dst_x, src_x, audio_f32).astype(np.float32, copy=False)


def format_float(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def load_cfg_dict() -> tuple[Path, Dict[str, Any]]:
    cfg_file = check_path_exists(CFG_PATH, __file__)
    if not cfg_file:
        raise FileNotFoundError(f"Config file not found: {CFG_PATH}")
    with cfg_file.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg_file, cfg


def update_top_level_yaml_block(path: Path, section: str, values: Dict[str, Any]) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    block_start = None
    block_end = len(lines)
    section_header = f"{section}:"

    for idx, line in enumerate(lines):
        if line.strip() == section_header and not line.startswith(" "):
            block_start = idx
            break
    if block_start is None:
        raise ValueError(f"Top-level section '{section}' not found in {path}")

    for idx in range(block_start + 1, len(lines)):
        line = lines[idx]
        if line and not line.startswith(" ") and not line.startswith("#"):
            block_end = idx
            break

    key_to_line: Dict[str, int] = {}
    for idx in range(block_start + 1, block_end):
        line = lines[idx]
        if not line.startswith("  "):
            continue
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key = stripped.split(":", 1)[0].strip()
        key_to_line[key] = idx

    for key, value in values.items():
        rendered = yaml.safe_dump({key: value}, default_flow_style=False, sort_keys=False).strip()
        rendered_value = rendered.split(":", 1)[1].strip()
        new_line = f"  {key}: {rendered_value}"
        if key in key_to_line:
            lines[key_to_line[key]] = new_line
        else:
            lines.insert(block_end, new_line)
            block_end += 1

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def measure_rms_window(
    *,
    duration_s: float,
    sample_rate: int,
    block_ms: int,
    latency: str | float = "low",
) -> Dict[str, float]:
    frames_per_block = max(1, int(sample_rate * (block_ms / 1000.0)))
    total_blocks = max(1, int(np.ceil(duration_s * 1000.0 / block_ms)))
    rms_values: list[float] = []
    peak_values: list[float] = []
    with sd.InputStream(
        channels=1,
        samplerate=sample_rate,
        dtype="float32",
        blocksize=frames_per_block,
        latency=latency,
    ) as stream:
        for _ in range(total_blocks):
            data, _ = stream.read(frames_per_block)
            mono = data.reshape(-1).astype(np.float32)
            rms, peak = rms_peak(mono)
            rms_values.append(rms)
            peak_values.append(peak)
    rms_arr = np.asarray(rms_values, dtype=np.float32)
    peak_arr = np.asarray(peak_values, dtype=np.float32)
    return {
        "rms_mean": float(rms_arr.mean()) if rms_arr.size else 0.0,
        "rms_p50": float(np.percentile(rms_arr, 50)) if rms_arr.size else 0.0,
        "rms_p95": float(np.percentile(rms_arr, 95)) if rms_arr.size else 0.0,
        "peak_max": float(peak_arr.max()) if peak_arr.size else 0.0,
    }


def build_calibrated_params(
    *,
    silence_stats: Dict[str, float],
    speech_stats: Dict[str, float],
    profile: str,
) -> Dict[str, Any]:
    profile_name = (profile or "balanced").lower()
    profile_defaults = {
        "fast": {
            "input_block_ms": 20,
            "input_preroll_ms": 140,
            "start_trigger_blocks": 2,
            "wake_capture_delay_ms": 90,
            "min_speech_seconds": 0.06,
            "end_silence_seconds": 0.20,
            "sentence_min_chars": 32,
        },
        "balanced": {
            "input_block_ms": 30,
            "input_preroll_ms": 180,
            "start_trigger_blocks": 2,
            "wake_capture_delay_ms": 120,
            "min_speech_seconds": 0.08,
            "end_silence_seconds": 0.28,
            "sentence_min_chars": 48,
        },
        "safe": {
            "input_block_ms": 40,
            "input_preroll_ms": 240,
            "start_trigger_blocks": 2,
            "wake_capture_delay_ms": 180,
            "min_speech_seconds": 0.12,
            "end_silence_seconds": 0.40,
            "sentence_min_chars": 72,
        },
    }
    defaults = dict(profile_defaults.get(profile_name, profile_defaults["balanced"]))
    noise_floor = float(silence_stats.get("rms_p95", 0.0))
    speech_floor = max(float(speech_stats.get("rms_p50", 0.0)), float(speech_stats.get("rms_p95", 0.0)) * 0.55)
    if speech_floor <= 0.0:
        calibrated_rms = 0.008
    else:
        lower_bound = max(0.003, noise_floor * 2.2)
        upper_bound = max(lower_bound, speech_floor * 0.55)
        calibrated_rms = min(max((noise_floor + speech_floor) * 0.5, lower_bound), upper_bound)
    defaults["min_speech_rms"] = round(calibrated_rms, 4)
    return defaults


def print_calibration_summary(
    *,
    silence_stats: Dict[str, float],
    speech_stats: Dict[str, float],
    params: Dict[str, Any],
    profile: str,
):
    print("\nCalibration summary")
    print(f"  profile: {profile}")
    print(
        "  silence:"
        f" rms_mean={format_float(silence_stats['rms_mean'])}"
        f" rms_p95={format_float(silence_stats['rms_p95'])}"
        f" peak_max={format_float(silence_stats['peak_max'])}"
    )
    print(
        "  speech:"
        f" rms_mean={format_float(speech_stats['rms_mean'])}"
        f" rms_p50={format_float(speech_stats['rms_p50'])}"
        f" rms_p95={format_float(speech_stats['rms_p95'])}"
        f" peak_max={format_float(speech_stats['peak_max'])}"
    )
    print("  recommended speech_pipeline values:")
    for key in [
        "min_speech_rms",
        "input_block_ms",
        "input_preroll_ms",
        "start_trigger_blocks",
        "wake_capture_delay_ms",
        "min_speech_seconds",
        "end_silence_seconds",
        "sentence_min_chars",
    ]:
        print(f"    {key}: {params[key]}")


def run_calibration_mode(*, write_config: bool = False, profile: str = "balanced") -> int:
    cfg_path, cfg = load_cfg_dict()
    speech_cfg = dict(cfg.get("speech_pipeline", {}) or {})
    sample_rate = int(speech_cfg.get("input_sample_rate", 16000))
    block_ms = int(speech_cfg.get("input_block_ms", 30))

    print("Speech pipeline calibration")
    print(f"  config: {cfg_path}")
    try:
        in_dev, out_dev = sd.default.device
        print(f"  default devices: input={in_dev} output={out_dev}")
    except Exception:
        pass
    print(f"  sample_rate={sample_rate} input_block_ms={block_ms}")
    print("  profile options: fast | balanced | safe")

    input("Press Enter for 2 seconds of room-noise measurement...")
    print("Measuring silence...")
    silence_stats = measure_rms_window(duration_s=2.0, sample_rate=sample_rate, block_ms=block_ms)

    input("Press Enter, then speak normally for 3 seconds...")
    print("Measuring speech...")
    speech_stats = measure_rms_window(duration_s=3.0, sample_rate=sample_rate, block_ms=block_ms)

    params = build_calibrated_params(
        silence_stats=silence_stats,
        speech_stats=speech_stats,
        profile=profile,
    )
    print_calibration_summary(
        silence_stats=silence_stats,
        speech_stats=speech_stats,
        params=params,
        profile=profile,
    )

    should_write = write_config
    if not should_write:
        answer = input("\nWrite these values into llm/config/config.yaml? [y/N]: ").strip().lower()
        should_write = answer in {"y", "yes"}

    if should_write:
        update_top_level_yaml_block(cfg_path, "speech_pipeline", params)
        print(f"Updated {cfg_path}")
    else:
        print("Config file not modified.")
    return 0


class RollingWavWriter:
    def __init__(self, path: Optional[str], samplerate: int):
        self.path = path
        self.sr = samplerate
        self.wf = None
        self.lock = threading.Lock()
        if path:
            import wave

            self.wf = wave.open(path, "wb")
            self.wf.setnchannels(1)
            self.wf.setsampwidth(2)
            self.wf.setframerate(samplerate)

    def write_float32(self, x: np.ndarray):
        if not self.wf:
            return
        with self.lock:
            self.wf.writeframes(pcm16_from_float32(x))

    def close(self):
        if not self.wf:
            return
        with self.lock:
            self.wf.close()
        self.wf = None


class OutputPlayer:
    def __init__(
        self,
        samplerate: int,
        block_ms: int = 40,
        *,
        prebuffer_ms: int = 120,
        device: Optional[Any] = None,
        latency: str | float = "high",
        debug: bool = False,
        dump_path: Optional[str] = None,
    ):
        self.source_sr = int(samplerate)
        self.sr = self._select_output_samplerate(device, self.source_sr, debug=debug)
        self.block = int(self.sr * (block_ms / 1000.0))
        self.queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=512)
        self.leftover = np.zeros(0, dtype=np.float32)
        self.underruns = 0
        self.prebuffer_blocks = max(1, int(np.ceil(prebuffer_ms / max(1, block_ms))))
        self._playback_armed = False
        self._pending_frames = 0
        self._playback_lock = threading.Lock()
        self._idle_event = threading.Event()
        self._idle_event.set()
        self.debug = debug
        self._last_stat = time.time()
        self._frames_out = 0
        self.writer = RollingWavWriter(dump_path, self.sr) if dump_path else None

        def callback(outdata, frames, time_info, status):
            if status and self.debug:
                print("[TTS] output status:", status)
            out = np.zeros(frames, dtype=np.float32)
            filled = 0
            if not self._playback_armed:
                buffered_blocks = self.queue.qsize() + (1 if self.leftover.size > 0 else 0)
                if buffered_blocks < self.prebuffer_blocks:
                    outdata[:, 0] = out
                    self.underruns += 1
                    return
                self._playback_armed = True
            while filled < frames:
                if self.leftover.size == 0:
                    try:
                        self.leftover = self.queue.get_nowait()
                    except queue.Empty:
                        self.underruns += 1
                        self._playback_armed = False
                        break
                take = min(frames - filled, self.leftover.size)
                out[filled : filled + take] = self.leftover[:take]
                self.leftover = self.leftover[take:]
                filled += take
            if filled > 0:
                with self._playback_lock:
                    self._pending_frames = max(0, self._pending_frames - filled)
                    if self._pending_frames == 0 and self.queue.qsize() == 0 and self.leftover.size == 0:
                        self._idle_event.set()
            outdata[:, 0] = out
            self._frames_out += frames
            if self.writer and filled > 0:
                self.writer.write_float32(out)
            now = time.time()
            if self.debug and (now - self._last_stat) >= 1.0:
                print(
                    f"[TTS] q={self.queue.qsize()} underruns={self.underruns} "
                    f"prebuffer_blocks={self.prebuffer_blocks} played={self._frames_out / self.sr:.2f}s"
                )
                self._last_stat = now
                self._frames_out = 0

        self.stream = sd.OutputStream(
            channels=1,
            samplerate=self.sr,
            dtype="float32",
            blocksize=0,
            device=device,
            latency=latency,
            callback=callback,
        )
        self.stream.start()

    @staticmethod
    def _select_output_samplerate(device: Optional[Any], requested_sr: int, *, debug: bool = False) -> int:
        def is_supported(sr: int) -> bool:
            try:
                sd.check_output_settings(device=device, samplerate=sr, channels=1, dtype="float32")
                return True
            except Exception:
                return False

        if is_supported(requested_sr):
            return requested_sr

        candidates: list[int] = []
        try:
            query_device = device
            if query_device is None:
                _, query_device = sd.default.device
            info = sd.query_devices(query_device, "output")
            default_sr = int(round(float(info.get("default_samplerate", 0))))
            if default_sr > 0:
                candidates.append(default_sr)
        except Exception:
            pass
        candidates.extend([48000, 44100, 32000, 16000])

        for sr in dict.fromkeys(candidates):
            if sr != requested_sr and is_supported(sr):
                if debug:
                    print(f"[TTS] output samplerate {requested_sr} Hz unsupported; using {sr} Hz")
                return sr

        return requested_sr

    def enqueue(self, audio_f32: np.ndarray):
        if audio_f32.size == 0:
            return
        audio_f32 = resample_audio(audio_f32, self.source_sr, self.sr)
        with self._playback_lock:
            self._pending_frames += int(audio_f32.size)
            self._idle_event.clear()
        idx = 0
        while idx < audio_f32.size:
            end = min(audio_f32.size, idx + self.block)
            self.queue.put(audio_f32[idx:end], block=True)
            idx = end

    def wait_until_idle(self, timeout: Optional[float] = None):
        if self._idle_event.wait(timeout=timeout):
            self._playback_armed = False
            return True
        return False

    def pending_duration_s(self) -> float:
        with self._playback_lock:
            return self._pending_frames / float(self.sr)

    def close(self):
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass
        if self.writer:
            self.writer.close()


class StreamedSpeechPipelineClient:
    on_transcript: Optional[Callable[[str], None]]
    on_text_delta: Optional[Callable[[str], None]]
    on_text_completed: Optional[Callable[[str], None]]
    on_error: Optional[Callable[[Dict[str, Any]], None]]

    def __init__(
        self,
        *,
        openai_api_key: Optional[str],
        instructions: str = "",
        reinforced_instructions: Optional[str] = None,
        llm_provider: str = "openai",
        llm_base_url: str = "https://api.openai.com/v1",
        llm_api_key: Optional[str] = None,
        llm_model: str = "gpt-4o-mini",
        llm_temperature: float = 0.6,
        tools: Optional[list] = None,
        tool_choice: str | Dict[str, Any] = "auto",
        stt_base_url: str = "https://api.openai.com/v1",
        stt_model: str = "whisper-1",
        stt_language: Optional[str] = "en",
        tts_base_url: str = "https://api.openai.com/v1",
        tts_model: str = "tts-1",
        tts_voice: str = "alloy",
        tts_instructions: Optional[str] = None,
        tts_enabled: bool = True,
        tts_response_format: str = "pcm",
        wake_sound_path: Optional[str] = None,
        sleep_sound_path: Optional[str] = None,
        fail_sound_path: Optional[str] = None,
        use_wake_word: bool = False,
        speech_key: Optional[str] = None,
        speech_region: Optional[str] = None,
        keyword_table_path: Optional[str] = None,
        input_sample_rate: int = 16000,
        tts_sample_rate: int = 24000,
        input_block_ms: int = 30,
        input_preroll_ms: int = 180,
        start_trigger_blocks: int = 2,
        wake_capture_delay_ms: int = 120,
        wake_rearm_delay_ms: int = 350,
        output_block_ms: int = 40,
        output_prebuffer_ms: int = 120,
        output_device: Optional[Any] = None,
        output_latency: str | float = "high",
        fade_ms: int = 5,
        min_speech_rms: float = 0.008,
        min_speech_seconds: float = 0.08,
        end_silence_seconds: float = 0.28,
        max_record_seconds: float = 12.0,
        sentence_min_chars: int = 48,
        print_transcript: bool = True,
        transcript_file: Optional[str] = None,
        debug_pipeline: bool = False,
        debug_audio_in: bool = False,
        debug_audio_out: bool = False,
        dump_mic_wav: Optional[str] = None,
        dump_out_wav: Optional[str] = None,
        request_timeout_s: float = 120.0,
    ):
        self.openai_api_key = openai_api_key
        self.instructions = instructions or ""
        self.reinforced_instructions = reinforced_instructions or ""
        self.llm_provider = (llm_provider or "openai").lower()
        self.llm_base_url = llm_base_url.rstrip("/")
        self.llm_api_key = llm_api_key
        self.llm_model = llm_model
        self._resolved_llm_model: Optional[str] = None
        self.llm_temperature = float(llm_temperature)
        self.tools = tools or []
        self.tool_choice = tool_choice
        self.stt_base_url = stt_base_url.rstrip("/")
        self.stt_model = stt_model
        self.stt_language = stt_language
        self.tts_base_url = tts_base_url.rstrip("/")
        self.tts_model = tts_model
        self.tts_voice = tts_voice
        self.tts_instructions = str(tts_instructions or "").strip()
        self.tts_enabled = bool(tts_enabled)
        self.tts_response_format = tts_response_format
        self.wake_sound_path = wake_sound_path
        self.sleep_sound_path = sleep_sound_path
        self.fail_sound_path = fail_sound_path
        self.USE_WAKE_WORD = bool(use_wake_word)
        self.SPEECH_KEY = speech_key
        self.SPEECH_REGION = speech_region
        self.KEYWORD_TABLE_PATH = str(keyword_table_path) if keyword_table_path else None
        self.INPUT_SAMPLE_RATE = int(input_sample_rate)
        self.TTS_SAMPLE_RATE = int(tts_sample_rate)
        self.INPUT_BLOCK_MS = int(input_block_ms)
        self.INPUT_PREROLL_MS = int(input_preroll_ms)
        self.START_TRIGGER_BLOCKS = int(start_trigger_blocks)
        self.WAKE_CAPTURE_DELAY_MS = int(wake_capture_delay_ms)
        self.WAKE_REARM_DELAY_MS = int(wake_rearm_delay_ms)
        self.OUTPUT_BLOCK_MS = int(output_block_ms)
        self.OUTPUT_PREBUFFER_MS = int(output_prebuffer_ms)
        self.OUTPUT_DEVICE = output_device
        self.OUTPUT_LATENCY = output_latency
        self.FADE_MS = int(fade_ms)
        self.MIN_SPEECH_RMS = float(min_speech_rms)
        self.MIN_SPEECH_SECONDS = float(min_speech_seconds)
        self.END_SILENCE_SECONDS = float(end_silence_seconds)
        self.MAX_RECORD_SECONDS = float(max_record_seconds)
        self.SENTENCE_MIN_CHARS = int(sentence_min_chars)
        self.PRINT_TRANSCRIPT = bool(print_transcript)
        self.TRANSCRIPT_FILE = transcript_file
        self.DEBUG_PIPELINE = bool(debug_pipeline)
        self.DEBUG_AUDIO_IN = bool(debug_audio_in)
        self.DEBUG_AUDIO_OUT = bool(debug_audio_out)
        self.DUMP_MIC_WAV = dump_mic_wav
        self.REQUEST_TIMEOUT_S = float(request_timeout_s)
        self._running = threading.Event()
        self._wake_event = threading.Event()
        self._keyword_thread: Optional[threading.Thread] = None
        self._tts_thread: Optional[threading.Thread] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._tts_queue: "queue.Queue[Optional[str]]" = queue.Queue()
        self._tts_work_pending = 0
        self._tts_pending_lock = threading.Lock()
        self._tts_pending_empty = threading.Event()
        self._tts_pending_empty.set()
        self.player = OutputPlayer(
            self.TTS_SAMPLE_RATE,
            block_ms=self.OUTPUT_BLOCK_MS,
            prebuffer_ms=self.OUTPUT_PREBUFFER_MS,
            device=self.OUTPUT_DEVICE,
            latency=self.OUTPUT_LATENCY,
            debug=self.DEBUG_AUDIO_OUT,
            dump_path=dump_out_wav,
        )
        self.mic_writer = RollingWavWriter(dump_mic_wav, self.INPUT_SAMPLE_RATE) if dump_mic_wav else None
        self.session = requests.Session()
        self._tool_handlers: Dict[str, Callable[..., Dict[str, Any]]] = {}
        self._conversation_lock = threading.Lock()
        self._stream_raw_text = ""
        self._stream_visible_chars = 0
        self.on_transcript = None
        self.on_text_delta = None
        self.on_text_completed = None
        self.on_error = None
        self._next_listen_should_respond = True
        self._playing_error_cue = False

    @classmethod
    def load(cls) -> "StreamedSpeechPipelineClient":
        cfg_file = check_path_exists(CFG_PATH, __file__)
        if not cfg_file:
            raise FileNotFoundError(f"Config file not found: {CFG_PATH}")
        with cfg_file.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        def expand_env(x):
            if isinstance(x, str):
                return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", lambda m: os.environ.get(m.group(1), ""), x)
            if isinstance(x, list):
                return [expand_env(v) for v in x]
            if isinstance(x, dict):
                return {k: expand_env(v) for k, v in x.items()}
            return x

        cfg = expand_env(cfg)
        speech_cfg = cfg.get("speechsdk", {})
        preset = cfg.get("presets", {})
        base_client = cfg.get("realtime_client") or cfg.get("client", {})
        pipeline_cfg = cfg.get("speech_pipeline", {})
        llm_provider = str(pipeline_cfg.get("llm_provider", "openai") or "openai").lower()
        llm_base_url = str(pipeline_cfg.get("llm_base_url", "") or "").strip()
        llm_model = str(pipeline_cfg.get("llm_model", "") or "").strip()
        llm_api_key = pipeline_cfg.get("llm_api_key") or pipeline_cfg.get("openai_api_key") or base_client.get("api_key")

        if llm_provider == "openai":
            if llm_base_url in ("", "http://localhost:1234/v1", "http://127.0.0.1:1234/v1"):
                llm_base_url = "https://api.openai.com/v1"
            if llm_model.lower() in ("", "auto", "current", "currently_loaded"):
                llm_model = "gpt-4o-mini"
            if not llm_api_key:
                llm_api_key = pipeline_cfg.get("openai_api_key") or base_client.get("api_key")
        else:
            if not llm_base_url:
                llm_base_url = "http://localhost:1234/v1"
            if not llm_model:
                llm_model = "currently_loaded"

        instructions = pipeline_cfg.get("instructions") or cfg.get("instructions", "")
        tools = cfg.get("tools", [])

        return cls(
            openai_api_key=pipeline_cfg.get("openai_api_key") or base_client.get("api_key"),
            instructions=instructions,
            reinforced_instructions=cfg.get("reinforced_instructions"),
            llm_provider=llm_provider,
            llm_base_url=llm_base_url or "https://api.openai.com/v1",
            llm_api_key=llm_api_key,
            llm_model=llm_model or "gpt-4o-mini",
            llm_temperature=pipeline_cfg.get("llm_temperature", 0.6),
            tools=tools,
            tool_choice=base_client.get("tool_choice", "auto"),
            stt_base_url=pipeline_cfg.get("stt_base_url", "https://api.openai.com/v1"),
            stt_model=pipeline_cfg.get("stt_model", "whisper-1"),
            stt_language=pipeline_cfg.get("stt_language", "en"),
            tts_base_url=pipeline_cfg.get("tts_base_url", "https://api.openai.com/v1"),
            tts_model=pipeline_cfg.get("tts_model", "tts-1"),
            tts_voice=pipeline_cfg.get("tts_voice") or base_client.get("voice", "alloy"),
            tts_instructions=pipeline_cfg.get("tts_instructions"),
            tts_enabled=pipeline_cfg.get("tts_enabled", True),
            tts_response_format=pipeline_cfg.get("tts_response_format", "pcm"),
            wake_sound_path=check_path_exists(preset.get("wake_sound"), __file__),
            sleep_sound_path=check_path_exists(preset.get("sleep_sound"), __file__),
            fail_sound_path=check_path_exists(preset.get("fail_sound"), __file__),
            use_wake_word=bool(speech_cfg.get("enable_wake_word", False)),
            speech_key=speech_cfg.get("subscription_key"),
            speech_region=speech_cfg.get("region"),
            keyword_table_path=check_path_exists(speech_cfg.get("keyword_table"), __file__),
            input_sample_rate=pipeline_cfg.get("input_sample_rate", 16000),
            tts_sample_rate=pipeline_cfg.get("tts_sample_rate", 24000),
            input_block_ms=pipeline_cfg.get("input_block_ms", 30),
            input_preroll_ms=pipeline_cfg.get("input_preroll_ms", 180),
            start_trigger_blocks=pipeline_cfg.get("start_trigger_blocks", 2),
            wake_capture_delay_ms=pipeline_cfg.get("wake_capture_delay_ms", 120),
            wake_rearm_delay_ms=pipeline_cfg.get("wake_rearm_delay_ms", 350),
            output_block_ms=pipeline_cfg.get("output_block_ms", 40),
            output_prebuffer_ms=pipeline_cfg.get("output_prebuffer_ms", 120),
            output_device=pipeline_cfg.get("output_device"),
            output_latency=pipeline_cfg.get("output_latency", "high"),
            fade_ms=pipeline_cfg.get("fade_ms", 5),
            min_speech_rms=pipeline_cfg.get("min_speech_rms", 0.008),
            min_speech_seconds=pipeline_cfg.get("min_speech_seconds", 0.08),
            end_silence_seconds=pipeline_cfg.get("end_silence_seconds", 0.28),
            max_record_seconds=pipeline_cfg.get("max_record_seconds", 12.0),
            sentence_min_chars=pipeline_cfg.get("sentence_min_chars", 48),
            print_transcript=pipeline_cfg.get("print_transcript", True),
            transcript_file=pipeline_cfg.get("transcript_file"),
            debug_pipeline=pipeline_cfg.get("debug_pipeline", False),
            debug_audio_in=pipeline_cfg.get("debug_audio_in", False),
            debug_audio_out=pipeline_cfg.get("debug_audio_out", False),
            dump_mic_wav=pipeline_cfg.get("dump_mic_wav"),
            dump_out_wav=pipeline_cfg.get("dump_out_wav"),
            request_timeout_s=pipeline_cfg.get("request_timeout_s", 120.0),
        )

    def _debug(self, stage: str, message: str):
        if self.DEBUG_PIPELINE:
            print(f"[PIPELINE:{stage}] {message}", flush=True)

    def _emit_error(self, payload: Dict[str, Any]):
        if (
            not self.tts_enabled
            and self.fail_sound_path
            and not self._playing_error_cue
        ):
            self._playing_error_cue = True
            try:
                self._play_wav(self.fail_sound_path, block_until_played=False)
            finally:
                self._playing_error_cue = False
        if self.on_error:
            try:
                self.on_error(payload)
                return
            except Exception:
                pass
        print("[StreamedSpeechPipelineClient ERROR]", payload)

    def _write_transcript(self, prefix: str, text: str):
        line = f"{prefix}{text}"
        if self.PRINT_TRANSCRIPT:
            print(line, flush=True)
        if self.TRANSCRIPT_FILE:
            try:
                Path(self.TRANSCRIPT_FILE).parent.mkdir(parents=True, exist_ok=True)
                with open(self.TRANSCRIPT_FILE, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception as e:
                self._emit_error({"error": f"transcript write failed: {e}"})

    def start(self):
        if self._running.is_set():
            return
        self._debug("lifecycle", "starting client")
        self._running.set()
        self._start_tts_worker()
        if self.USE_WAKE_WORD:
            self._start_keyword_listener_thread()

    def start_background(self):
        if self._loop_thread and self._loop_thread.is_alive():
            return
        self._loop_thread = threading.Thread(target=self.run_forever, name="speech-pipeline-loop", daemon=True)
        self._loop_thread.start()

    def stop(self):
        if not self._running.is_set():
            return
        self._debug("lifecycle", "stopping client")
        self._running.clear()
        self._wake_event.set()
        self._tts_queue.put(None)
        try:
            if self._loop_thread and self._loop_thread.is_alive():
                self._loop_thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            if self._keyword_thread and self._keyword_thread.is_alive():
                self._keyword_thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            if self._tts_thread and self._tts_thread.is_alive():
                self._tts_thread.join(timeout=2.0)
        except Exception:
            pass
        try:
            self.player.close()
        except Exception:
            pass
        try:
            if self.mic_writer:
                self.mic_writer.close()
        except Exception:
            pass

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()

    def wake(self, *, play_cue: bool = True):
        self._debug("wake", f"wake requested play_cue={play_cue}")
        if play_cue:
            self._play_wav(self.wake_sound_path, block_until_played=False)

    def request_listen(self, *, play_cue: bool = True, respond: bool = True):
        self._debug("wake", f"listen requested play_cue={play_cue} respond={respond}")
        self.wake(play_cue=play_cue)
        self._next_listen_should_respond = bool(respond)
        self._wake_event.set()

    def register_tool_handler(self, name: str, fn: Callable[..., Dict[str, Any]]):
        self._debug("tools", f"registered tool handler: {name}")
        self._tool_handlers[name] = fn

    def speak_openai(self, text: str):
        self._debug("tts", f"queue direct speech len={len(text)}")
        self._queue_tts(text)
        self._wait_for_tts_completion()

    def send_text(self, text: str, *, role: str = "user", speak: bool = True):
        if role != "user":
            text = f"{role}: {text}"
        if speak:
            return self.respond_once(text)
        return self._run_nonstream_completion(text)

    def _filtered_tools(self) -> list:
        filtered = []
        for tool in self.tools or []:
            if not isinstance(tool, dict):
                continue
            name = self._tool_name(tool)
            if name and name in self._tool_handlers:
                filtered.append(tool)
        return filtered

    def _tool_name(self, tool: Dict[str, Any]) -> Optional[str]:
        if not isinstance(tool, dict):
            return None
        name = tool.get("name")
        if name:
            return str(name)
        fn = tool.get("function") or {}
        name = fn.get("name")
        return str(name) if name else None

    def _normalize_tool_schema(self, tool: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(tool, dict):
            return None
        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            fn = tool["function"]
            if fn.get("name"):
                return {
                    "type": "function",
                    "function": {
                        "name": fn["name"],
                        "description": fn.get("description", ""),
                        "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
                    },
                }
            return None

        name = tool.get("name")
        if not name:
            return None
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
            },
        }

    def _safe_tool_choice(self):
        if isinstance(self.tool_choice, dict):
            name = self.tool_choice.get("name")
            if not name:
                fn = self.tool_choice.get("function") or {}
                name = fn.get("name")
            if name and name not in self._tool_handlers:
                return "auto"
        return self.tool_choice

    def run_forever(self):
        self.start()
        if not self.USE_WAKE_WORD:
            self.request_listen(play_cue=False)

        while self._running.is_set():
            if not self._wake_event.wait(timeout=0.1):
                continue
            self._debug("wake", "wake event received")
            self._wake_event.clear()
            should_respond = self._next_listen_should_respond
            self._next_listen_should_respond = True
            if not self._running.is_set():
                break
            try:
                if self.WAKE_CAPTURE_DELAY_MS > 0:
                    self._debug("wake", f"sleeping {self.WAKE_CAPTURE_DELAY_MS} ms before capture")
                    time.sleep(self.WAKE_CAPTURE_DELAY_MS / 1000.0)
                with self._conversation_lock:
                    transcript = self.listen_once()
                if not transcript:
                    self._debug("stt", "no transcript captured after wake")
                    if self.fail_sound_path:
                        self._play_wav(self.fail_sound_path, block_until_played=False)
                    if self.USE_WAKE_WORD and self._running.is_set():
                        self._start_keyword_listener_thread()
                    continue
                if should_respond:
                    with self._conversation_lock:
                        self.respond_once(transcript)
                    if self.sleep_sound_path:
                        self._play_wav(self.sleep_sound_path, block_until_played=False)
                else:
                    self._debug("wake", "listen request completed without response")
            except KeyboardInterrupt:
                break
            except Exception as e:
                self._emit_error({"error": f"pipeline failed: {e}"})
            finally:
                if self.USE_WAKE_WORD and self._running.is_set():
                    self._debug("wake", f"re-arming wake-word listener after {self.WAKE_REARM_DELAY_MS} ms")
                    self._schedule_keyword_listener(delay_ms=self.WAKE_REARM_DELAY_MS)

    def listen_once(self) -> str:
        started = time.time()
        self._debug("stt", "starting utterance capture")
        audio_path = self._capture_utterance_to_tempfile()
        if not audio_path:
            self._debug("stt", f"capture produced no audio in {time.time() - started:.2f}s")
            return ""
        try:
            self._debug("stt", f"audio captured -> {audio_path}")
            transcript = self._transcribe_audio(audio_path)
        finally:
            try:
                os.unlink(audio_path)
            except OSError:
                pass
        transcript = (transcript or "").strip()
        if transcript:
            self._debug("stt", f"transcript ready in {time.time() - started:.2f}s len={len(transcript)}")
            self._write_transcript("[user] ", transcript)
            if self.on_transcript:
                try:
                    self.on_transcript(transcript)
                except Exception:
                    pass
        else:
            self._debug("stt", f"transcription empty after {time.time() - started:.2f}s")
        return transcript

    def respond_once(self, user_text: str) -> str:
        if self._conversation_lock.acquire(blocking=False):
            try:
                full_text = self._respond_once_locked(user_text)
            finally:
                self._conversation_lock.release()
            return full_text
        return self._respond_once_locked(user_text)

    def _respond_once_locked(self, user_text: str) -> str:
        self._debug("llm", f"starting response for user text len={len(user_text)}")
        started = time.time()
        full_text = self._stream_or_tool_roundtrip(user_text)
        self._wait_for_tts_completion()
        if full_text:
            self._debug("llm", f"response complete in {time.time() - started:.2f}s len={len(full_text)}")
            self._write_transcript("[assistant] ", full_text)
            if self.on_text_completed:
                try:
                    self.on_text_completed(full_text)
                except Exception:
                    pass
        else:
            self._debug("llm", f"response empty after {time.time() - started:.2f}s")
        return full_text

    def _run_nonstream_completion(self, user_text: str) -> str:
        content, _ = self._create_chat_completion(
            self._build_messages(user_text),
            stream=False,
            allow_tools=False,
        )
        return self._strip_hidden_reasoning_complete(content)

    def _capture_utterance_to_tempfile(self) -> Optional[str]:
        frames_per_block = max(1, int(self.INPUT_SAMPLE_RATE * (self.INPUT_BLOCK_MS / 1000.0)))
        max_blocks = max(1, int(self.MAX_RECORD_SECONDS * 1000 / self.INPUT_BLOCK_MS))
        min_voice_blocks = max(
            int(self.START_TRIGGER_BLOCKS),
            max(1, int(self.MIN_SPEECH_SECONDS * 1000 / self.INPUT_BLOCK_MS)),
        )
        silence_blocks = max(1, int(self.END_SILENCE_SECONDS * 1000 / self.INPUT_BLOCK_MS))
        preroll_blocks = max(1, int(self.INPUT_PREROLL_MS / self.INPUT_BLOCK_MS))
        heard_voice = False
        voice_run = 0
        silence_run = 0
        chunks: list[np.ndarray] = []
        preroll_chunks: deque[np.ndarray] = deque(maxlen=preroll_blocks)

        if self.DEBUG_AUDIO_IN:
            print("[Mic] listening for utterance")
        self._debug(
            "mic",
            f"capture config block_ms={self.INPUT_BLOCK_MS} preroll_ms={self.INPUT_PREROLL_MS} "
            f"min_voice_blocks={min_voice_blocks} silence_blocks={silence_blocks} rms_threshold={self.MIN_SPEECH_RMS}",
        )

        with sd.InputStream(
            channels=1,
            samplerate=self.INPUT_SAMPLE_RATE,
            dtype="float32",
            blocksize=frames_per_block,
            latency="low",
        ) as stream:
            for _ in range(max_blocks):
                if not self._running.is_set():
                    return None
                data, _ = stream.read(frames_per_block)
                mono = data.reshape(-1).astype(np.float32)
                if self.mic_writer:
                    self.mic_writer.write_float32(mono)
                rms, peak = rms_peak(mono)
                active = rms >= self.MIN_SPEECH_RMS or peak >= min(0.99, self.MIN_SPEECH_RMS * 4.0)
                if not heard_voice:
                    preroll_chunks.append(mono)
                else:
                    chunks.append(mono)
                if active:
                    voice_run += 1
                    silence_run = 0
                elif heard_voice:
                    silence_run += 1
                if not heard_voice and voice_run >= min_voice_blocks:
                    heard_voice = True
                    chunks = list(preroll_chunks)
                    preroll_chunks.clear()
                    chunks.append(mono)
                    self._debug("mic", f"speech detected after {len(chunks)} buffered chunks")
                if self.DEBUG_AUDIO_IN:
                    print(f"[Mic] rms={rms:.4f} peak={peak:.4f} active={active} heard={heard_voice}")
                if heard_voice and silence_run >= silence_blocks:
                    self._debug("mic", f"ending capture after silence_run={silence_run}")
                    break

        if not heard_voice:
            self._debug("mic", "speech not detected")
            return None

        audio = np.concatenate(chunks, axis=0)
        fd, tmp_path = tempfile.mkstemp(prefix="speech_pipeline_", suffix=".wav")
        os.close(fd)
        sf.write(tmp_path, audio, self.INPUT_SAMPLE_RATE, subtype="PCM_16")
        self._debug("mic", f"wrote wav samples={audio.size} duration={audio.size / self.INPUT_SAMPLE_RATE:.2f}s")
        return tmp_path

    def _transcribe_audio(self, audio_path: str) -> str:
        if not self.openai_api_key:
            raise RuntimeError("OpenAI API key is required for speech-to-text.")
        started = time.time()
        self._debug("stt", f"POST {self.stt_base_url}/audio/transcriptions model={self.stt_model}")
        headers = {"Authorization": f"Bearer {self.openai_api_key}"}
        data = {"model": self.stt_model}
        if self.stt_language:
            data["language"] = self.stt_language
        with open(audio_path, "rb") as f:
            files = {"file": (Path(audio_path).name, f, "audio/wav")}
            response = self.session.post(
                f"{self.stt_base_url}/audio/transcriptions",
                headers=headers,
                data=data,
                files=files,
                timeout=self.REQUEST_TIMEOUT_S,
            )
        response.raise_for_status()
        payload = response.json()
        self._debug("stt", f"transcription HTTP completed in {time.time() - started:.2f}s")
        return str(payload.get("text", "") or "")

    def _llm_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if self.llm_provider == "openai":
            api_key = self.llm_api_key or self.openai_api_key
            if not api_key:
                raise RuntimeError("OpenAI API key is required for OpenAI chat streaming.")
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _should_autoresolve_lmstudio_model(self) -> bool:
        name = str(self.llm_model or "").strip().lower()
        return self.llm_provider == "lmstudio" and name in ("", "auto", "current", "currently_loaded")

    def _get_effective_llm_model(self) -> str:
        if not self._should_autoresolve_lmstudio_model():
            return str(self.llm_model)
        if self._resolved_llm_model:
            return self._resolved_llm_model

        started = time.time()
        response = self.session.get(
            f"{self.llm_base_url}/models",
            headers={"Accept": "application/json"},
            timeout=min(self.REQUEST_TIMEOUT_S, 15.0),
        )
        response.raise_for_status()
        payload = response.json()
        models = payload.get("data") or []
        if not models:
            raise RuntimeError("LM Studio /models returned no loaded models.")
        model_id = str(models[0].get("id") or "").strip()
        if not model_id:
            raise RuntimeError("LM Studio /models returned a model without an id.")
        self._resolved_llm_model = model_id
        self._debug("llm", f"resolved LM Studio model='{model_id}' in {time.time() - started:.2f}s")
        return model_id

    def _build_messages(self, user_text: str, extra_messages: Optional[list] = None) -> list:
        message_text = user_text
        if self.reinforced_instructions:
            message_text = f"{user_text}\n{self.reinforced_instructions}"
        messages = [{"role": "system", "content": self.instructions}, {"role": "user", "content": message_text}]
        if extra_messages:
            messages.extend(extra_messages)
        return messages

    def _stream_or_tool_roundtrip(self, user_text: str) -> str:
        if not self._filtered_tools():
            self._debug("tools", "no eligible tool handlers for this request")
            return self._stream_llm_and_queue_tts(self._build_messages(user_text))

        self._debug("tools", f"tool-enabled request with handlers={sorted(self._tool_handlers.keys())}")
        content, tool_calls = self._create_chat_completion(
            self._build_messages(user_text),
            stream=False,
            allow_tools=True,
        )
        if not tool_calls:
            self._debug("tools", "model returned no tool calls")
            if content:
                self._emit_text_locally(content)
            return content.strip()

        self._debug("tools", f"model returned {len(tool_calls)} tool call(s)")
        followup_messages = [
            {
                "role": "assistant",
                "content": content or "",
                "tool_calls": tool_calls,
            }
        ]
        suppress_followup_response = False
        for tool_call in tool_calls:
            tool_result = self._run_tool_call(tool_call)
            if isinstance(tool_result, dict):
                suppress_followup_response = suppress_followup_response or bool(
                    tool_result.get("suppress_followup_response", False)
                )
            followup_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.get("id"),
                    "content": json.dumps(tool_result),
                }
            )
        if suppress_followup_response:
            self._debug("tools", "post-tool LLM/TTS response suppressed by tool result")
            return ""
        return self._stream_llm_and_queue_tts(self._build_messages(user_text, extra_messages=followup_messages), allow_tools=False)

    def _emit_text_locally(self, text: str):
        pending_tts = ""
        full_text = ""
        text = self._strip_hidden_reasoning_complete(text)
        for delta in text:
            if not delta:
                continue
            full_text += delta
            pending_tts += delta
            if self.on_text_delta:
                try:
                    self.on_text_delta(delta)
                except Exception:
                    pass
            if self.PRINT_TRANSCRIPT:
                print(delta, end="", flush=True)
            flush_text, pending_tts = self._split_tts_text(pending_tts)
            if flush_text.strip():
                self._queue_tts(flush_text.strip())
        if self.PRINT_TRANSCRIPT and full_text:
            print("", flush=True)
        if pending_tts.strip():
            self._queue_tts(pending_tts.strip())
        return

    def _stream_llm_and_queue_tts(self, messages: list, *, allow_tools: bool = False) -> str:
        self._reset_stream_reasoning_filter()
        started = time.time()
        model_name = self._get_effective_llm_model()
        self._debug("llm", f"stream chat request model={model_name} allow_tools={allow_tools} messages={len(messages)}")
        response = self.session.post(
            f"{self.llm_base_url}/chat/completions",
            headers=self._llm_headers(),
            json=self._chat_payload(messages, stream=True, allow_tools=allow_tools, model_name=model_name),
            timeout=self.REQUEST_TIMEOUT_S,
            stream=True,
        )
        response.raise_for_status()

        full_text = ""
        pending_tts = ""
        chunk_count = 0
        for event in self._iter_sse_json(response):
            raw_delta = self._extract_chat_delta_text(event)
            delta = self._consume_visible_stream_text(raw_delta)
            if not delta:
                continue
            chunk_count += 1
            full_text += delta
            pending_tts += delta
            if self.on_text_delta:
                try:
                    self.on_text_delta(delta)
                except Exception:
                    pass
            if self.PRINT_TRANSCRIPT:
                print(delta, end="", flush=True)
            flush_text, pending_tts = self._split_tts_text(pending_tts)
            if flush_text.strip():
                self._debug("tts", f"queue streamed sentence len={len(flush_text.strip())}")
                self._queue_tts(flush_text.strip())
        if self.PRINT_TRANSCRIPT and full_text:
            print("", flush=True)
        if pending_tts.strip():
            self._debug("tts", f"queue final streamed tail len={len(pending_tts.strip())}")
            self._queue_tts(pending_tts.strip())
        self._debug("llm", f"stream completed in {time.time() - started:.2f}s chunks={chunk_count} chars={len(full_text)}")
        return full_text.strip()

    def _reset_stream_reasoning_filter(self):
        self._stream_raw_text = ""
        self._stream_visible_chars = 0

    def _consume_visible_stream_text(self, raw_delta: str) -> str:
        if not raw_delta:
            return ""
        self._stream_raw_text += raw_delta
        visible = self._strip_hidden_reasoning_partial(self._stream_raw_text)
        if len(visible) <= self._stream_visible_chars:
            return ""
        new_text = visible[self._stream_visible_chars :]
        self._stream_visible_chars = len(visible)
        return new_text

    def _strip_hidden_reasoning_complete(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r"(?is)<think>.*?</think>", "", text)
        text = re.sub(r"(?is)<tool_call>.*?</tool_call>", "", text)
        text = re.sub(r'(?im)^\s*<tool_call>\s*$', "", text)
        text = re.sub(r'(?im)^\s*\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{.*\}\s*\}\s*$', "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _strip_hidden_reasoning_partial(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r"(?is)<tool_call>.*?</tool_call>", "", text)
        text = re.sub(r'(?im)^\s*<tool_call>\s*$', "", text)
        text = re.sub(r'(?im)^\s*\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{.*\}\s*\}\s*$', "", text)
        visible_parts: list[str] = []
        idx = 0
        while idx < len(text):
            start = text.find("<think>", idx)
            if start == -1:
                tail = text[idx:]
                if "<think".startswith(tail):
                    break
                visible_parts.append(tail)
                break
            visible_parts.append(text[idx:start])
            end = text.find("</think>", start + len("<think>"))
            if end == -1:
                break
            idx = end + len("</think>")
        visible = "".join(visible_parts)
        visible = visible.replace("<think>", "").replace("</think>", "")
        return visible

    def _chat_payload(self, messages: list, *, stream: bool, allow_tools: bool, model_name: Optional[str] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model_name or self._get_effective_llm_model(),
            "stream": stream,
            "temperature": self.llm_temperature,
            "messages": messages,
        }
        if allow_tools:
            filtered_tools = self._filtered_tools()
            if filtered_tools:
                payload["tools"] = [t for t in (self._normalize_tool_schema(tool) for tool in filtered_tools) if t]
                payload["tool_choice"] = self._safe_tool_choice()
        return payload

    def _create_chat_completion(self, messages: list, *, stream: bool, allow_tools: bool) -> tuple[str, list]:
        started = time.time()
        model_name = self._get_effective_llm_model()
        self._debug("llm", f"create completion model={model_name} stream={stream} allow_tools={allow_tools}")
        response = self.session.post(
            f"{self.llm_base_url}/chat/completions",
            headers=self._llm_headers(),
            json=self._chat_payload(messages, stream=stream, allow_tools=allow_tools, model_name=model_name),
            timeout=self.REQUEST_TIMEOUT_S,
            stream=stream,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as e:
            body = ""
            try:
                body = response.text[:1200]
            except Exception:
                pass
            raise RuntimeError(f"{e}; response_body={body}") from e
        if stream:
            raise ValueError("_create_chat_completion(stream=True) is not supported")
        payload = response.json()
        self._debug("llm", f"completion HTTP finished in {time.time() - started:.2f}s")
        choices = payload.get("choices") or []
        if not choices:
            return "", []
        message = choices[0].get("message") or {}
        content = message.get("content") or ""
        tool_calls = self._extract_tool_calls(message)
        if tool_calls:
            self._debug("tools", f"normalized tool call payload count={len(tool_calls)}")
        return str(content), tool_calls

    def _extract_tool_calls(self, message: Dict[str, Any]) -> list[Dict[str, Any]]:
        tool_calls = message.get("tool_calls") or []
        normalized: list[Dict[str, Any]] = []
        if isinstance(tool_calls, list):
            for idx, tool_call in enumerate(tool_calls):
                norm = self._normalize_tool_call(tool_call, default_id=f"tool_call_{idx}")
                if norm:
                    normalized.append(norm)
        function_call = message.get("function_call")
        if function_call:
            norm = self._normalize_tool_call({"function": function_call}, default_id="function_call_0")
            if norm:
                normalized.append(norm)
        return normalized

    def _normalize_tool_call(self, tool_call: Any, *, default_id: str) -> Optional[Dict[str, Any]]:
        if not isinstance(tool_call, dict):
            return None
        fn = tool_call.get("function")
        if isinstance(fn, dict) and fn.get("name"):
            return {
                "id": tool_call.get("id") or default_id,
                "type": tool_call.get("type") or "function",
                "function": {
                    "name": fn.get("name"),
                    "arguments": fn.get("arguments") or "{}",
                },
            }
        name = tool_call.get("name")
        if name:
            return {
                "id": tool_call.get("id") or default_id,
                "type": tool_call.get("type") or "function",
                "function": {
                    "name": name,
                    "arguments": tool_call.get("arguments") or "{}",
                },
            }
        return None

    def _run_tool_call(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        fn = tool_call.get("function") or {}
        name = fn.get("name")
        raw_arguments = fn.get("arguments") or "{}"
        self._debug("tools", f"running tool {name} raw_args={raw_arguments}")
        try:
            arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else dict(raw_arguments or {})
        except Exception as e:
            return {"error": f"tool args parse failed for {name}: {e}"}
        handler = self._tool_handlers.get(name)
        if not handler:
            self._emit_error(
                {
                    "error": f"Unknown tool: {name}",
                    "known_handlers": sorted(self._tool_handlers.keys()),
                    "tool_call": tool_call,
                }
            )
            return {"error": f"Unknown tool: {name}"}
        try:
            started = time.time()
            result = handler(**arguments)
            self._debug("tools", f"tool {name} completed in {time.time() - started:.2f}s")
            if isinstance(result, dict):
                return result
            return {"result": result}
        except Exception as e:
            return {"error": f"Tool {name} crashed: {repr(e)}"}

    def _iter_sse_json(self, response: requests.Response):
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            line = raw_line.strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                yield json.loads(data)
            except json.JSONDecodeError:
                continue

    def _extract_chat_delta_text(self, event: Dict[str, Any]) -> str:
        choices = event.get("choices") or []
        if not choices:
            return ""
        delta = choices[0].get("delta") or {}
        content = delta.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text" and item.get("text"):
                        text_parts.append(str(item["text"]))
                    elif item.get("type") == "output_text" and item.get("text"):
                        text_parts.append(str(item["text"]))
            return "".join(text_parts)
        return ""

    def _split_tts_text(self, pending: str) -> tuple[str, str]:
        if len(pending) < self.SENTENCE_MIN_CHARS and not re.search(r"[.!?]\s*$", pending):
            return "", pending
        match = None
        for m in re.finditer(r".*?[.!?](?:\s+|$)", pending, flags=re.S):
            match = m
        if match:
            return pending[: match.end()], pending[match.end() :]
        if len(pending) >= self.SENTENCE_MIN_CHARS:
            split_at = pending.rfind(" ")
            if split_at > int(self.SENTENCE_MIN_CHARS * 0.6):
                return pending[:split_at], pending[split_at + 1 :]
        return "", pending

    def _start_tts_worker(self):
        if self._tts_thread and self._tts_thread.is_alive():
            return
        self._debug("tts", "starting TTS worker thread")
        self._tts_thread = threading.Thread(target=self._tts_worker, name="speech-tts", daemon=True)
        self._tts_thread.start()

    def _queue_tts(self, text: str):
        if not self.tts_enabled:
            self._debug("tts", "speech suppressed because tts_enabled=false")
            return
        text = self._strip_hidden_reasoning_complete(text)
        if not text:
            return
        with self._tts_pending_lock:
            self._tts_work_pending += 1
            self._tts_pending_empty.clear()
        self._debug("tts", f"queued text len={len(text)} pending={self._tts_work_pending}")
        self._tts_queue.put(text)

    def _mark_tts_done(self):
        with self._tts_pending_lock:
            self._tts_work_pending = max(0, self._tts_work_pending - 1)
            if self._tts_work_pending == 0:
                self._tts_pending_empty.set()

    def _wait_for_tts_completion(self):
        self._debug("tts", "waiting for queued speech to finish")
        synthesis_done = self._tts_pending_empty.wait(timeout=self.REQUEST_TIMEOUT_S)
        if not synthesis_done:
            with self._tts_pending_lock:
                pending = self._tts_work_pending
            self._emit_error(
                {
                    "error": "timed out waiting for TTS synthesis to finish",
                    "pending_segments": pending,
                    "timeout_s": self.REQUEST_TIMEOUT_S,
                }
            )
        playback_done = self.player.wait_until_idle(timeout=self.REQUEST_TIMEOUT_S)
        if not playback_done:
            self._emit_error(
                {
                    "error": "timed out waiting for audio playback to drain",
                    "queued_audio_s": round(self.player.pending_duration_s(), 3),
                    "timeout_s": self.REQUEST_TIMEOUT_S,
                }
            )
        self._debug("tts", "playback queue drained")

    def _tts_worker(self):
        self._debug("tts", "TTS worker active")
        while True:
            text = self._tts_queue.get()
            if text is None:
                self._debug("tts", "TTS worker stopping")
                break
            try:
                self._debug("tts", f"synthesizing text len={len(text)}")
                self._stream_tts_audio(text)
            except Exception as e:
                self._emit_error({"error": f"tts failed: {e}", "text": text})
            finally:
                self._mark_tts_done()

    def _stream_tts_audio(self, text: str):
        if not self.openai_api_key:
            raise RuntimeError("OpenAI API key is required for text-to-speech.")
        started = time.time()
        self._debug("tts", f"POST {self.tts_base_url}/audio/speech model={self.tts_model} voice={self.tts_voice}")
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.tts_model,
            "voice": self.tts_voice,
            "input": text,
            "response_format": self.tts_response_format,
        }
        if self.tts_instructions:
            payload["instructions"] = self.tts_instructions
        response = self.session.post(
            f"{self.tts_base_url}/audio/speech",
            headers=headers,
            json=payload,
            timeout=self.REQUEST_TIMEOUT_S,
            stream=True,
        )
        response.raise_for_status()

        leftover = b""
        chunk_count = 0
        for chunk in response.iter_content(chunk_size=4096):
            if not chunk:
                continue
            chunk_count += 1
            pcm_bytes = leftover + chunk
            aligned = (len(pcm_bytes) // 2) * 2
            if aligned == 0:
                leftover = pcm_bytes
                continue
            audio = float32_from_pcm16(pcm_bytes[:aligned])
            self.player.enqueue(audio)
            leftover = pcm_bytes[aligned:]
        if leftover:
            audio = float32_from_pcm16(leftover + (b"\x00" if len(leftover) % 2 else b""))
            self.player.enqueue(audio)
        self._debug("tts", f"synthesis stream completed in {time.time() - started:.2f}s chunks={chunk_count}")

    def _start_keyword_listener_thread(self):
        if self._keyword_thread and self._keyword_thread.is_alive():
            return
        self._debug("wake", "starting wake-word listener thread")
        self._keyword_thread = threading.Thread(target=self._keyword_waiter, name="wakeword", daemon=True)
        self._keyword_thread.start()

    def _schedule_keyword_listener(self, *, delay_ms: int = 0):
        def _worker():
            if delay_ms > 0:
                time.sleep(delay_ms / 1000.0)
            if not self._running.is_set():
                return
            self._start_keyword_listener_thread()

        threading.Thread(target=_worker, name="wakeword-rearm", daemon=True).start()

    def _keyword_waiter(self):
        if speechsdk is None:
            self._emit_error({"error": "azure.cognitiveservices.speech is not installed; wake-word mode is unavailable."})
            return
        if not self.KEYWORD_TABLE_PATH:
            self._emit_error({"error": "Wake word enabled but no keyword_table_path provided/found."})
            return
        if not os.path.exists(self.KEYWORD_TABLE_PATH):
            self._emit_error({"error": f"Wake-word file not found: {self.KEYWORD_TABLE_PATH}"})
            return
        try:
            audio_cfg = speechsdk.audio.AudioConfig(use_default_microphone=True)
            kw_model = speechsdk.KeywordRecognitionModel(self.KEYWORD_TABLE_PATH)
            if self.SPEECH_KEY and self.SPEECH_REGION:
                speechsdk.SpeechConfig(subscription=self.SPEECH_KEY, region=self.SPEECH_REGION)
            recognizer = speechsdk.KeywordRecognizer(audio_config=audio_cfg)
            if self.DEBUG_AUDIO_IN:
                print("[WakeWord] listening:", self.KEYWORD_TABLE_PATH)
            self._debug("wake", f"armed keyword recognizer file={self.KEYWORD_TABLE_PATH}")
            result = recognizer.recognize_once_async(model=kw_model).get()
            if result.reason == speechsdk.ResultReason.RecognizedKeyword and self._running.is_set():
                self._debug("wake", "keyword recognized")
                self.request_listen(play_cue=True)
            else:
                self._debug("wake", f"keyword recognizer finished with reason={result.reason}")
        except Exception as e:
            self._emit_error({"error": f"Wake-word thread crashed: {repr(e)}"})
        finally:
            self._keyword_thread = None

    def _play_wav(self, path: Optional[str], *, block_until_played: bool = True):
        if not path:
            return
        try:
            self._debug("tts", f"playing wav cue path={path} block={block_until_played}")
            data, samplerate = sf.read(path, dtype="float32")
            if isinstance(data, np.ndarray) and data.ndim > 1:
                data = data[:, 0]
            audio = np.asarray(data, dtype=np.float32).reshape(-1)
            audio = resample_audio(audio, int(samplerate), self.TTS_SAMPLE_RATE)
            audio = apply_fade(audio, self.TTS_SAMPLE_RATE, self.FADE_MS)
            self.player.enqueue(audio)
            if block_until_played:
                self.player.wait_until_idle(timeout=self.REQUEST_TIMEOUT_S)
        except Exception as e:
            self._emit_error({"error": f"wav playback failed: {e}", "path": path})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrate", action="store_true", help="Measure mic levels and recommend speech_pipeline settings.")
    parser.add_argument(
        "--calibration-profile",
        choices=["fast", "balanced", "safe"],
        default="balanced",
        help="Responsiveness profile used for recommended settings.",
    )
    parser.add_argument(
        "--write-config",
        action="store_true",
        help="Write recommended calibration values into llm/config/config.yaml without prompting.",
    )
    args = parser.parse_args()

    if args.calibrate:
        raise SystemExit(run_calibration_mode(write_config=args.write_config, profile=args.calibration_profile))

    client = StreamedSpeechPipelineClient.load()
    client.on_error = lambda e: print("[App] error:", e)
    client.on_text_delta = lambda s: None
    client.on_text_completed = lambda s: None

    print("Speech pipeline client ready. Commands: /wake, /listen, /say <text>, /quit")

    client.start()
    input_thread_stop = threading.Event()

    def input_loop():
        while not input_thread_stop.is_set():
            line = sys.stdin.readline()
            if not line:
                time.sleep(0.05)
                continue
            msg = line.strip()
            if not msg:
                continue
            if msg.lower() in ("/q", "/quit", "quit", "exit"):
                input_thread_stop.set()
                client.stop()
                return
            if msg.startswith("/wake"):
                client.wake()
                continue
            if msg.startswith("/listen"):
                client.request_listen(play_cue=True)
                continue
            if msg.startswith("/say "):
                text = msg[5:].strip()
                if text:
                    client.respond_once(text)
                continue
            print("Unknown command. Use /wake, /listen, /say <text>, or /quit")

    t = threading.Thread(target=input_loop, name="stdin-loop", daemon=True)
    t.start()

    try:
        client.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        input_thread_stop.set()
        client.stop()


if __name__ == "__main__":
    main()
