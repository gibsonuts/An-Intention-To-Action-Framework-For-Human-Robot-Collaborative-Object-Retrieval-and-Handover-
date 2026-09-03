
#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import base64
import json
import os
import queue
import threading
import time
from typing import Any, Callable, Dict, Optional
import numpy as np
import sounddevice as sd
import soundfile as sf
import websocket
import azure.cognitiveservices.speech as speechsdk
import yaml, re
from commons.grasp_utils import check_path_exists
from pathlib import Path
from llm.websearcher import WebSearcher
from openai import OpenAI

# Hard-coded YAML config path (no alternative or env override)
CFG_PATH = "config/config.yaml"


def pcm16_from_float32(x: np.ndarray) -> bytes:
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767).astype(np.int16).tobytes()

def float32_from_pcm16(b: bytes) -> np.ndarray:
    if not b:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(b, dtype=np.int16).astype(np.float32) / 32767.0

def rms_peak(x: np.ndarray):
    if x.size == 0:
        return 0.0, 0.0
    r = float(np.sqrt(np.mean(np.square(x))))
    p = float(np.max(np.abs(x)))
    return r, p

def apply_fade(audio_f32: np.ndarray, sr: int, ms: int):
    if audio_f32.size == 0 or ms <= 0:
        return audio_f32
    n = min(audio_f32.size, int(sr * (ms / 1000.0)))
    if n <= 1:
        return audio_f32
    fade_in = np.linspace(0.0, 1.0, n, dtype=np.float32)
    fade_out = np.linspace(1.0, 0.0, n, dtype=np.float32)
    y = audio_f32.copy()
    y[:n] *= fade_in
    y[-n:] *= fade_out
    return y

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
            self.wf.setsampwidth(2)  # 16-bit
            self.wf.setframerate(samplerate)

    def write_float32(self, x: np.ndarray):
        if not self.wf:
            return
        with self.lock:
            self.wf.writeframes(pcm16_from_float32(x))

    def close(self):
        if self.wf:
            with self.lock:
                self.wf.close()
            self.wf = None

class OutputPlayer:
    """Low-latency, de-jittered playback with stats."""

    def __init__(self, samplerate: int, block_ms: int = 20, debug: bool = False, dump_path: Optional[str] = None,
                 on_audio_enqueued: Optional[Callable[[np.ndarray], None]] = None):
        self.sr = samplerate
        self.block = int(self.sr * (block_ms / 1000.0))
        self.queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=512)
        self.leftover = np.zeros(0, dtype=np.float32)
        self.underruns = 0
        self.debug = debug
        self._last_stat = time.time()
        self._frames_out = 0
        self.writer = RollingWavWriter(dump_path, samplerate) if dump_path else None
        self._on_audio_enqueued = on_audio_enqueued

        def callback(outdata, frames, time_info, status):
            if status and self.debug:
                print("OUT status:", status)
            out = np.zeros(frames, dtype=np.float32)
            filled = 0
            while filled < frames:
                if self.leftover.size == 0:
                    try:
                        chunk = self.queue.get_nowait()
                        self.leftover = chunk
                    except queue.Empty:
                        self.underruns += 1
                        break
                take = min(frames - filled, self.leftover.size)
                out[filled : filled + take] = self.leftover[:take]
                self.leftover = self.leftover[take:]
                filled += take

            outdata[:, 0] = out
            self._frames_out += frames
            if self.writer and filled > 0:
                self.writer.write_float32(out)

            if self._on_audio_enqueued and filled > 0:
                # notify once per callback when audio has been output
                try:
                    self._on_audio_enqueued(out)
                except Exception:
                    pass

            now = time.time()
            if self.debug and (now - self._last_stat) >= 1.0:
                qsz = self.queue.qsize()
                dur = self._frames_out / self.sr
                print(f"[OUT] q={qsz} underruns={self.underruns} played={dur:.2f}s")
                self._last_stat = now
                self._frames_out = 0

        self.stream = sd.OutputStream(
            channels=1,
            samplerate=self.sr,
            dtype="float32",
            blocksize=self.block,
            callback=callback,
        )
        self.stream.start()

    def enqueue(self, audio_f32: np.ndarray):
        idx = 0
        n = audio_f32.size
        while idx < n:
            end = min(n, idx + self.block)
            block = audio_f32[idx:end]
            self.queue.put(block, block=True)
            idx = end

    def qsize(self):
        return self.queue.qsize()

    def close(self):
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass
        if self.writer:
            self.writer.close()

def is_client_realtime_running(client: "OpenAIRealtimeClient") -> bool:
    """Return True if the realtime pipeline appears to be running."""
    try:
        running_flag = getattr(client, "_running", None)
        if not (running_flag and running_flag.is_set()):
            return False

        ws_thread_ok = bool(getattr(client, "_ws_thread", None) and client._ws_thread.is_alive())
        ws_ok = bool(
            getattr(client, "_wsapp", None)
            and getattr(client._wsapp, "sock", None)
            and getattr(client._wsapp.sock, "connected", False)
        )
        audio_ok = bool(
            getattr(client, "player", None)
            and getattr(client.player, "stream", None)
            and getattr(client.player.stream, "active", False)
        )
        return (ws_thread_ok or ws_ok) and audio_ok
    except Exception:
        return False

# ---------------------- Main Client (public API) ---------------------- #

ToolHandler = Callable[..., Dict[str, Any]]
TextCallback = Callable[[str], None]
AudioCallback = Callable[[np.ndarray], None]
EventCallback = Callable[[Dict[str, Any]], None]

class OpenAIRealtimeClient:
    """
    A reusable realtime WebSocket client supporting audio input, audio output with
    ducking/half-duplex gating, and function-calling via Python callbacks.

    Optional wake-word mode via Microsoft Speech SDK:
      - Set use_wake_word=True (or speechsdk.enable_wake_word: true in YAML).
      - Provide keyword_table_path to your .table file (or speechsdk.keyword_table).
      - Optionally set AZURE_SPEECH_KEY / AZURE_SPEECH_REGION or provide in YAML.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-realtime-preview",
        voice: str = "marin",
        ws_url: Optional[str] = None,
        instructions: Optional[str] = None,
        reinforced_instructions:  Optional[str] = None,
        websearch_instructions:  Optional[str] = None,
        tools: Optional[list] = None,
        tool_choice: str | Dict[str, Any] = "auto",
        # Audio config
        sample_rate: int = 24000,
        chunk_ms: int = 200,
        out_block_ms: int = 20,
        fade_ms: int = 5,
        mic_latency: str = "low",
        # Echo handling
        echo_handling: str = "half_duplex",  # 'off' | 'half_duplex' | 'attenuate'
        echo_guard_ms: int = 700,            # base guard (will be auto-extended)
        attenuation_db: int = 18,
        enable_websearch: bool = True,
        # Debug
        debug_audio_in: bool = False,
        debug_audio_out: bool = False,
        dump_mic_wav: Optional[str] = None,
        dump_out_wav: Optional[str] = None,
        wake_sound_path: Optional[str] = None,
        sleep_sound_path: Optional[str] = None,
        fail_sound_path: Optional[str] = None,

        # -------------------- NEW: Wake-word / idle options -------------------- #
        use_wake_word: bool = False,
        speech_key: Optional[str] = None,          # Azure Speech key (or env)
        speech_region: Optional[str] = None,       # Azure region (or env)
        keyword_table_path: Optional[str] = None,  # path to .table
        idle_timeout_s: float = 8.0,               # seconds of user silence
        rms_voice_threshold: float = 0.01,         # ~0.005–0.02 typical

        # -------------------- NEW: Transcript options -------------------- #
        show_transcript: bool = False,
        transcript_mode: str = "final",            # "final" | "stream" | "both"
        transcript_file: Optional[str] = None,
    ):
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.ws_url = ws_url or f"wss://api.openai.com/v1/realtime?model={model}"
        self.instructions = instructions or ""
        self.reinforced_instructions = reinforced_instructions
        self.tools = tools or []
        self.tool_choice = tool_choice

        self.openai_sdk_client = OpenAI(api_key=api_key)

        # Audio / echo
        self.SAMPLE_RATE = sample_rate
        self.CHUNK_MS = chunk_ms
        self.OUT_BLOCK_MS = out_block_ms
        self.FADE_MS = fade_ms
        self.MIC_LATENCY = mic_latency
        self.ECHO_HANDLING = (echo_handling or "half_duplex").lower()
        self.ECHO_GUARD_MS = int(echo_guard_ms)
        self.ATTENUATION_DB = attenuation_db

        # Debug
        self.DEBUG_AUDIO_IN = debug_audio_in
        self.DEBUG_AUDIO_OUT = debug_audio_out

        self.mic_writer = RollingWavWriter(dump_mic_wav, self.SAMPLE_RATE) if dump_mic_wav else None
        self.out_dump_path = dump_out_wav

        # State
        self._wsapp: Optional[websocket.WebSocketApp] = None
        self._ws_thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._running.clear()

        self.frames_per_chunk = int(self.SAMPLE_RATE * (self.CHUNK_MS / 1000.0))
        self.current_buf = bytearray()
        self._pending_tool_calls: Dict[str, Dict[str, Any]] = {}
        self._tool_handlers: Dict[str, ToolHandler] = {}
        self.enable_mic = False
        # Callbacks
        self.on_text_delta: Optional[TextCallback] = None
        self.on_text_completed: Optional[TextCallback] = None
        self.on_audio_enqueued: Optional[AudioCallback] = None
        self.on_event: Optional[EventCallback] = None
        self.on_error: Optional[EventCallback] = None

        # ----------- NEW: Robust TTS activity tracking -----------
        # The mic is considered unsafe to open while now < _tts_active_until
        self._tts_active_until: float = 0.0

        # Audio output player
        # When audio is *actually* written to device, bump _tts_active_until too
        def _on_out_audio(_audio: np.ndarray):
            # protect until a bit past the current device callback
            tail_ms = self._dynamic_guard_tail_ms()
            self._tts_active_until = max(self._tts_active_until, time.time() + tail_ms / 1000.0)

        self.player = OutputPlayer(
            self.SAMPLE_RATE,
            block_ms=self.OUT_BLOCK_MS,
            debug=self.DEBUG_AUDIO_OUT,
            dump_path=self.out_dump_path,
            on_audio_enqueued=_on_out_audio,
        )

        self.MIC_STREAM_RUNNING  = False

        self.wake_sound_path = wake_sound_path
        self.sleep_sound_path = sleep_sound_path
        self.fail_sound_path = fail_sound_path

        # Wake-word / idle state
        self.USE_WAKE_WORD = use_wake_word
        self.SPEECH_KEY = speech_key
        self.SPEECH_REGION = speech_region
        self.KEYWORD_TABLE_PATH = str(keyword_table_path) if keyword_table_path else None
        self.IDLE_TIMEOUT_S = float(idle_timeout_s)
        self.RMS_VOICE_THRESHOLD = float(rms_voice_threshold)

        self._keyword_thread: Optional[threading.Thread] = None
        self._mic_thread: Optional[threading.Thread] = None
        self._last_voice_ts: float = 0.0

        # Transcript settings
        self.SHOW_TRANSCRIPT = bool(show_transcript)
        mode = (transcript_mode or "final").lower()
        self.TRANSCRIPT_MODE = mode if mode in ("final", "stream", "both") else "final"
        self.TRANSCRIPT_FILE = transcript_file
        self._stream_line_open = False  # pretty-print state for streaming

        if enable_websearch:
            self.searcher = WebSearcher()
            self.register_tool_handler("web_search", self.get_websearch_handler)
            self.instructions = self.instructions + '\n' + (websearch_instructions or "")
        else:
            self.instructions += '\n cannot access time-sensitive information from web'

    # ---------------------- Helpers ---------------------- #
    def speak_openai(self,text):
            #check if mp3 file for text exists in the directory
        file_path = os.path.join(r"", "data/voice_samples/"+text+".wav")
      
        if os.path.exists(file_path):
            self._play_wav(file_path, block_until_played=True)
            return
    
        print('speak ',text)
        if self.openai_sdk_client is None:
            # Fallback: no-op if no key present
            print(f"[TTS disabled] {text}")
            return
        model = 'tts-1-hd'
        response = self.openai_sdk_client.audio.speech.create(
            model=model,
            voice=self.voice,
            input=text
        )
        response.stream_to_file(file_path)
        self._play_wav(file_path, block_until_played=True)
        



    def _dynamic_guard_tail_ms(self) -> int:
        """
        Dynamic guard = max(configured base, 2*block + 120ms)
        so that larger output blocks automatically extend the mute window.
        """
        return int(max(self.ECHO_GUARD_MS, (2 * self.OUT_BLOCK_MS) + 120))

    # ---------------------- Public API ---------------------- #

    @classmethod
    def load(cls) -> "OpenAIRealtimeClient":
        """Build a client from a YAML config file.
        Supports ${VARNAME} environment expansions in string fields.
        """
        cfg = {}
        cfg_file = check_path_exists(CFG_PATH,__file__)
        if cfg_file:
            with cfg_file.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        else:
            print('ERROR no cfg file',cfg_file)
            exit()

        def expand_env(x):
            if isinstance(x, str):
                return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", lambda m: os.environ.get(m.group(1), ""), x)
            if isinstance(x, list):
                return [expand_env(i) for i in x]
            if isinstance(x, dict):
                return {k: expand_env(v) for k, v in x.items()}
            return x

        cfg = expand_env(cfg)
        params = cfg.get("realtime_client") or cfg.get("client", {})
        tools = cfg.get("tools", [])
        instructions = cfg.get("instructions")
        websearch_instructions = cfg.get("websearch_instructions")
        reinforced_instructions = cfg.get("reinforced_instructions")
        preset = cfg.get("presets", {})
        wake_sound_path = preset.get("wake_sound")
        sleep_sound_path = preset.get("sleep_sound")
        fail_sound_path = preset.get("fail_sound")

        # speechsdk config
        speech_cfg = cfg.get("speechsdk", {})
        use_wake_word = bool(speech_cfg.get("enable_wake_word", False))
        speech_key = speech_cfg.get("subscription_key")
        speech_region = speech_cfg.get("region")
        keyword_table_path= speech_cfg.get("keyword_table")  # e.g., "config/hey_computer.table"
        idle_timeout_s = speech_cfg.get("idle_timeout_seconds", 8)
        rms_voice_threshold = speech_cfg.get("rms_voice_threshold", 0.01)

        return cls(
            api_key=params.get("api_key"),
            model=params.get("model", "gpt-realtime"),
            voice=params.get("voice", "marin"),
            ws_url=params.get("ws_url"),
            instructions=instructions,
            reinforced_instructions=reinforced_instructions,
            websearch_instructions= websearch_instructions,
            tools=tools,
            tool_choice=params.get("tool_choice", "auto"),
            sample_rate=params.get("sample_rate", 24000),
            chunk_ms=params.get("chunk_ms", 200),
            out_block_ms=params.get("out_block_ms", 20),
            fade_ms=params.get("fade_ms", 5),
            mic_latency=params.get("mic_latency", "low"),
            echo_handling=params.get("echo_handling", "half_duplex"),
            echo_guard_ms=params.get("echo_guard_ms", 700),
            attenuation_db=params.get("attenuation_db", 18),
            debug_audio_in=params.get("debug_audio_in", False),
            debug_audio_out=params.get("debug_audio_out", False),
            dump_mic_wav=params.get("dump_mic_wav"),
            dump_out_wav=params.get("dump_out_wav"),
            enable_websearch=params.get("enable_websearch",True),
            wake_sound_path = check_path_exists(wake_sound_path,__file__),
            sleep_sound_path = check_path_exists(sleep_sound_path,__file__),
            fail_sound_path = check_path_exists(fail_sound_path,__file__),

            # Wake word
            use_wake_word=use_wake_word,
            speech_key=speech_key,
            speech_region=speech_region,
            keyword_table_path=check_path_exists(keyword_table_path,__file__),
            idle_timeout_s=idle_timeout_s,
            rms_voice_threshold=rms_voice_threshold,

            # Transcript config comes from the YAML `realtime_client:` block.
            show_transcript=params.get("show_transcript", False),
            transcript_mode=params.get("transcript_mode", "final"),
            transcript_file=params.get("transcript_file"),
        )

    def get_websearch_handler(self,query: str):
        print(f"[TOOL:search web] {query}")
        result = self.searcher.ask(query)
        response = result["output_text"]
        print(response)
        return {"status": "ok", "result": response, "send_to_model":True}

    def register_tool_handler(self, name: str, fn: ToolHandler):
        """Register a Python callback for a tool/function call from the model."""
        self._tool_handlers[name] = fn

    def _filtered_tools(self) -> list:
        """Return only tools whose names have a registered handler."""
        filtered: list = []
        for t in (self.tools or []):
            if not isinstance(t, dict):
                continue
            name = t.get("name")
            if name and name in self._tool_handlers:
                filtered.append(t)
        return filtered

    def _safe_tool_choice(self):
        """Ensure tool_choice doesn't reference a filtered-out tool."""
        tc = self.tool_choice
        if isinstance(tc, dict):
            name = tc.get("name")
            if name and name not in self._tool_handlers:
                return "auto"
        return tc

    def start(self) -> None:
        """Start microphone streaming and connect the WebSocket in a background thread."""
        if self._running.is_set():
            return
        self._running.set()
        headers = [
            f"Authorization: Bearer {self.api_key}",
            "OpenAI-Beta: realtime=v1",
        ]
        self._wsapp = websocket.WebSocketApp(
            self.ws_url,
            header=headers,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        self._ws_thread = threading.Thread(target=self._wsapp.run_forever, name="realtime-ws", daemon=True)
        self._ws_thread.start()
        #wait until everything is running
        while not is_client_realtime_running(self):
            time.sleep(0.1)
        print("[Client] Realtime client started.")

    def stop(self, wait: bool = True) -> None:
        """Stop streaming and close the connection quickly and safely."""
        if not self._running.is_set():
            return

        self._running.clear()

        try:
            if self._wsapp:
                try:
                    self._wsapp.keep_running = False
                except Exception:
                    pass
                try:
                    self._wsapp.close(status=1001, reason="shutdown")
                except Exception:
                    pass
        except Exception:
            pass

        try:
            if getattr(self, "player", None):
                self.player.close()
        except Exception:
            pass

        try:
            if self.mic_writer:
                self.mic_writer.close()
        except Exception:
            pass

        try:
            if wait and self._ws_thread and self._ws_thread.is_alive():
                self._ws_thread.join(timeout=1.5)
        except Exception:
            pass

        try:
            sd.stop()
        except Exception:
            pass

    def send_text(self, text: str, *, role: str = "user", speak: bool = True) -> None:
        ws = self._wsapp
        if not ws:
            raise RuntimeError("WebSocket not started. Call start() first.")
        content_type = "input_text" if role == "user" else "text"
        text = text + ' \n ' + self.reinforced_instructions #alway add reinforcement instruction to input
        ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": role,
                "content": [{"type": content_type, "text": text}],
            },
        }))
        ws.send(json.dumps({
            "type": "response.create",
            "response": {"modalities": (["audio", "text"] if speak else ["text"])}
        }))
        print('[send_text]: ',text)

    # ---------------------- NEW: image sending ---------------------- #
    def send_image(
        self,
        *,
        image_bytes: bytes,
        mime_type: str = "image/jpeg",
        prompt_text: Optional[str] = None,
        speak: bool = False,
        role: str = "user",
    ) -> None:
        """Sends an image (as a data URI) to the model, optionally with a text prompt."""
        if not self._wsapp:
            raise RuntimeError("WebSocket not started. Call start() first.")

        b64 = base64.b64encode(image_bytes).decode("ascii")
        data_uri = f"data:{mime_type};base64,{b64}"

        content = []
        if prompt_text:
            content.append({"type": "input_text", "text": prompt_text})
        content.append({"type": "input_image", "image_url": data_uri})

        self._wsapp.send(json.dumps({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": role,
                "content": content,
            },
        }))
        self._wsapp.send(json.dumps({
            "type": "response.create",
            "response": {"modalities": (["audio", "text"] if speak else ["text"])}
        }))
        print('[send_image]: ',prompt_text)

    def force_reply(self) -> None:
        """Force the model to respond using the current buffered mic input."""
        ws = self._wsapp
        if not ws:
            raise RuntimeError("WebSocket not started. Call start() first.")
        try:
            ws.send(json.dumps({"type": "response.create", "response": {"instructions":self.reinforced_instructions,"modalities": ["audio", "text"]}}))
        except Exception as e:
            self._emit_error({"error": f"force_reply failed: {e}"})

    # Context manager helpers
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()

    # ---------------------- Transcript helper ---------------------- #
    def _write_transcript(self, text: str, end: str = "\n", stream_chunk: bool = False):
        try:
            print(text, end=end, flush=True)
            if stream_chunk:
                self._stream_line_open = True
            elif end == "\n":
                self._stream_line_open = False

            if self.TRANSCRIPT_FILE:
                try:
                    Path(self.TRANSCRIPT_FILE).parent.mkdir(parents=True, exist_ok=True)
                except Exception:
                    pass
                with open(self.TRANSCRIPT_FILE, "a", encoding="utf-8") as f:
                    f.write(text)
                    if end != "":
                        f.write("\n")
        except Exception:
            pass

    # ---------------------- Internal callbacks ---------------------- #

    def _on_open(self, ws):
        try:
            default_in, default_out = sd.default.device
            if self.DEBUG_AUDIO_OUT:
                print("Default devices -> IN:", default_in, "OUT:", default_out)
        except Exception:
            pass

        filtered_tools = self._filtered_tools()
        safe_tool_choice = self._safe_tool_choice()
        filtered_tool_names = [b.get("name", "") for b in (filtered_tools or []) if "name" in b and b.get("name")]
        print('llm tools being used:',filtered_tool_names)

        session_update = {
            "type": "session.update",
            "session": {
                "modalities": ["audio", "text"],
                "voice": self.voice,
                "turn_detection": {"type": "server_vad"},
                "instructions": self.instructions,
                "tools": filtered_tools,
                "tool_choice": safe_tool_choice,
            },
        }
        ws.send(json.dumps(session_update))

        if self.USE_WAKE_WORD:
            self._start_keyword_listener_thread()
            self.MIC_STREAM_RUNNING = False
        else:
            self.MIC_STREAM_RUNNING = True

        self._start_mic_stream_thread(ws)

    def _on_message(self, ws, message):
        try:
            event = json.loads(message)
        except Exception:
            return

        etype = event.get("type")
        if self.on_event:
            try:
                self.on_event(event)
            except Exception:
                pass

        if etype == "session.updated":
            return

        # ----- Text streaming (optional pretty transcript) -----
        if etype == "response.audio_transcript.delta":
            if self.on_text_delta:
                delta = event.get("delta", "")
                try:
                    self.on_text_delta(delta)
                except Exception:
                    pass
            if self.SHOW_TRANSCRIPT and self.TRANSCRIPT_MODE in ("stream", "both"):
                delta = event.get("delta", "")
                if delta:
                    if not self._stream_line_open:
                        self._write_transcript("\n[assistant•stream] ", end="")
                    self._write_transcript(delta, end="", stream_chunk=True)
            return

        if etype in "response.audio_transcript.done":
            if self.on_text_completed:
                try:
                    txt = event.get("transcript", "")
                    self.on_text_completed(txt)
                except Exception:
                    pass
            if self.SHOW_TRANSCRIPT and self.TRANSCRIPT_MODE in ("final", "both"):
                txt = event.get("transcript", "")
                if txt:
                    if self._stream_line_open:
                        self._write_transcript("")
                    self._write_transcript(f"[assistant] {txt}")
            return

        # ----- Audio streaming from server (TTS) -----
        if etype in ("response.audio.delta", "response.output_audio.delta"):
            # As soon as we see *any* output audio, extend the protection window
            tail_ms = self._dynamic_guard_tail_ms()
            self._tts_active_until = max(self._tts_active_until, time.time() + tail_ms / 1000.0)

            delta_b64 = event.get("delta")
            if delta_b64:
                self.current_buf.extend(base64.b64decode(delta_b64))
            return

        if etype in ("response.audio.done", "response.output_audio.done"):
            # When a clip is completed, we enqueue it and extend guard again
            if self.current_buf:
                audio_f32 = float32_from_pcm16(bytes(self.current_buf))
                audio_f32 = apply_fade(audio_f32, self.SAMPLE_RATE, self.FADE_MS)
                if self.DEBUG_AUDIO_OUT:
                    dur = audio_f32.size / self.SAMPLE_RATE
                    r, p = rms_peak(audio_f32)
                    print(f"[OUT] clip received: {dur*1000:.0f} ms, rms={r:.3f}, peak={p:.3f}")
                self.player.enqueue(audio_f32)
                self.current_buf = bytearray()

            # Extend the guard a little more after 'done' to cover device flush
            tail_ms = self._dynamic_guard_tail_ms()
            self._tts_active_until = max(self._tts_active_until, time.time() + tail_ms / 1000.0)
            return

        # ----- Function calling lifecycle -----
        if etype == "response.output_item.added":
            item = event.get("item", {})            
            if item.get("type") == "function_call":
                call_id = item.get("call_id") or item.get("id")
                name = item.get("name")
                if call_id:
                    self._pending_tool_calls[call_id] = {"name": name, "args": ""}
            return

        if etype == "response.function_call_arguments.delta":
            call_id = event.get("call_id")
            delta = event.get("delta", "")
            rec = self._pending_tool_calls.setdefault(call_id, {"name": event.get("name"), "args": ""})
            rec["args"] += delta
            return

        if etype == "response.function_call_arguments.done":
            call_id = event.get("call_id")
            rec = self._pending_tool_calls.get(call_id, {})
            name = rec.get("name") or event.get("name")
            print(rec)
            arg_str = rec.get("args") or event.get("arguments", "")
            try:
                args = json.loads(arg_str) if arg_str else {}
            except Exception as e:
                print({"error": f"tool args JSON parse error for {name}: {e}", "raw": arg_str} )
                self._emit_error({"error": f"tool args JSON parse error for {name}: {e}", "raw": arg_str})
                args = {}
            handler = self._tool_handlers.get(name)
            if not handler:
                tool_result: Dict[str, Any] = {"error": f"Unknown tool: {name}"}
                print("error: not handler")
            else:
                try:
                    print('args',args)
                    tool_result = handler(**args)
                    # if not isinstance(tool_result, dict):
                    #     tool_result = {"result": tool_result}
                    print('tool_result',tool_result)
                except Exception as e:
                    print( {"error": f"Tool {name} crashed: {repr(e)}"})
                    tool_result = {"error": f"Tool {name} crashed: {repr(e)}"}

            send_to_model = False
            if "send_to_model" in tool_result:
                if isinstance(tool_result["send_to_model"], bool):
                   send_to_model = tool_result["send_to_model"]
            if send_to_model:
                try:
                    ws.send(
                        json.dumps(
                            {
                                "type": "conversation.item.create",
                                "item": {
                                    "type": "function_call_output",
                                    "call_id": call_id,
                                    "output": json.dumps(tool_result),
                                },
                            }
                        )
                    )
                    ws.send(json.dumps({"type": "response.create", "response": {"instructions":self.reinforced_instructions,"modalities": ["audio", "text"]}}))
                except Exception as e:
                    self._emit_error({"error": f"failed to send tool result: {e}"})
                finally:
                    self._pending_tool_calls.pop(call_id, None)
                return

        if etype == "error":
            self._emit_error(event)
            return

    def _on_error(self, ws, error):
        self._emit_error({"error": str(error)})

    def _on_close(self, ws, code, reason):
        try:
            if self.mic_writer:
                self.mic_writer.close()
        except Exception:
            pass
        try:
            self.player.close()
        except Exception:
            pass
        if self._running.is_set():
            self._emit_error({"error": f"WebSocket closed ({code}): {reason}"})
        self._running.clear()

    # ---------------------- Mic worker & echo logic ---------------------- #

    def _output_is_active(self) -> bool:
        # Hard gate while any TTS is *likely* audible
        if time.time() < self._tts_active_until:
            return True
        # Also gate while there is buffered audio pending
        if self.player.qsize() > 0:
            return True
        return False

    def _attenuation_gain(self) -> float:
        return 10 ** (-float(self.ATTENUATION_DB) / 20.0)

    def _start_mic_stream_thread(self, ws):
        if self._mic_thread and self._mic_thread.is_alive():
            return
        self._mic_thread = threading.Thread(
            target=self._mic_stream, args=(ws,), name="mic-stream", daemon=True
        )
        self._mic_thread.start()

    def _mic_stream(self, ws):
        frames = self.frames_per_chunk
        self._last_voice_ts = time.time()  # start timer when conversation begins
        # simple smoothing for attenuate mode (avoid pumping)
        fade_len = max(8, int(0.5 * frames))  # ~half a chunk
        fade = np.linspace(1.0, self._attenuation_gain(), fade_len).astype(np.float32)
        
        with sd.InputStream(
            channels=1,
            samplerate=self.SAMPLE_RATE,
            dtype="float32",
            blocksize=frames,
            latency=self.MIC_LATENCY,
        ) as istream:
            try:
               
                while self._running.is_set() and ws.keep_running:
                    audio_chunk, _ = istream.read(frames)
                    mono = audio_chunk.flatten().astype(np.float32)

                    output_active = self._output_is_active()
                    if output_active:
                        self._last_voice_ts = time.time()

                    if output_active and self.ECHO_HANDLING != "off":
                        if self.ECHO_HANDLING == "half_duplex":
                            # Drop mic while TTS is playing — bulletproof against feedback
                            continue
                        elif self.ECHO_HANDLING == "attenuate":
                            # Smooth ramp for first samples of chunk
                            if mono.size >= fade_len:
                                mono[:fade_len] *= fade
                                mono[fade_len:] *= self._attenuation_gain()
                            else:
                                g = self._attenuation_gain()
                                mono *= g
                    
                    if self.MIC_STREAM_RUNNING:
                        if self.mic_writer:
                            self.mic_writer.write_float32(mono)

                        try:
                            ws.send(
                                json.dumps(
                                    {
                                        "type": "input_audio_buffer.append",
                                        "audio": base64.b64encode(pcm16_from_float32(mono)).decode("ascii"),
                                    }
                                )
                            )
                        except Exception as e:
                            self._emit_error({"error": f"mic send failed: {e}"})
                            break

                    # NEW: idle timeout -> return to wake word
                    if self.USE_WAKE_WORD and self.MIC_STREAM_RUNNING:
                        if (time.time() - self._last_voice_ts) >= self.IDLE_TIMEOUT_S:
                            print('going to sleep')
                            try:
                                self._play_wav(self.sleep_sound_path, block_until_played=True)
                            except Exception:
                                pass
                            
                            self.MIC_STREAM_RUNNING = False  
                            if self._running.is_set() and ws.keep_running:
                                self._start_keyword_listener_thread() 

            finally:
                print('mic stream stopped for some reason')
                self.MIC_STREAM_RUNNING = False
                # if self.USE_WAKE_WORD and self._running.is_set() and ws.keep_running:
                    # self._start_keyword_listener_thread()

    # ---------------------- Wake-word handling (Speech SDK) ---------------------- #

    def _start_keyword_listener_thread(self):
        if self._keyword_thread and self._keyword_thread.is_alive():
            return
        t = threading.Thread(target=self._keyword_waiter, name="wakeword", daemon=True)
        self._keyword_thread = t
        t.start()

    def _keyword_waiter(self):
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
                _ = speechsdk.SpeechConfig(subscription=self.SPEECH_KEY, region=self.SPEECH_REGION)

            recognizer = speechsdk.KeywordRecognizer(audio_config=audio_cfg)

            if self.DEBUG_AUDIO_IN:
                print("[WakeWord] Listening for keyword via:", self.KEYWORD_TABLE_PATH)

            result = recognizer.recognize_once_async(model=kw_model).get()

            if result.reason == speechsdk.ResultReason.RecognizedKeyword:
                if not self.MIC_STREAM_RUNNING:
                    try:
                        self._arm_local_audio_guard_for_wav(self.wake_sound_path)
                        self._play_wav(self.wake_sound_path, block_until_played=False)
                    except Exception:
                        pass
                    if self._wsapp and getattr(self._wsapp, "sock", None):
                        self._last_voice_ts = time.time()
                        self.MIC_STREAM_RUNNING = True
                        # self._start_mic_stream_thread(self._wsapp)
            else:
                if self._running.is_set():
                    if self.DEBUG_AUDIO_IN:
                        print(f"[WakeWord] Non-Recognized result: {result.reason}. Re-arming…")
                    self._start_keyword_listener_thread()
        except Exception as e:
            self._emit_error({"error": f"Wake-word thread crashed: {repr(e)}"})
            if self._running.is_set():
                self._start_keyword_listener_thread()

    # ---------------------- Error surfacing ---------------------- #

    def _emit_error(self, payload: Dict[str, Any]):
        if self.on_error:
            try:
                self.on_error(payload)
            except Exception:
                pass
        else:
            print("[OpenAIRealtimeClient ERROR]", payload)

    def _arm_local_audio_guard_for_wav(self, path: Optional[str], *, extra_tail_ms: Optional[int] = None) -> None:
        """Gate the mic during locally played cues (ding/beep)."""
        if not path:
            return
        try:
            info = sf.info(path)
            dur_s = float(getattr(info, "duration", 0.0) or 0.0)
        except Exception:
            dur_s = 0.0
        tail_ms = int(self._dynamic_guard_tail_ms() if extra_tail_ms is None else extra_tail_ms)
        self._tts_active_until = max(self._tts_active_until, time.time() + dur_s + (tail_ms / 1000.0))

    def _play_wav(self, path: Optional[str], *, block_until_played: bool = True):
        try:
            if not path:
                return
            data, samplerate = sf.read(path, dtype='float32')
            sd.play(data, samplerate)
            if block_until_played:
                sd.wait()
        except FileNotFoundError:
            print(f"Error: The file '{path}' was not found.")
        except Exception as e:
            print(f"An error occurred: {e}")

    def wake(self, *, play_cue: bool = True) -> None:
        if not self._wsapp or not getattr(self._wsapp, "sock", None):
            self._emit_error({"error": "Wake ignored: WebSocket not connected."})
            return

        if self.MIC_STREAM_RUNNING:
            self._last_voice_ts = time.time()
            if self.DEBUG_AUDIO_IN:
                print("[Wake] Stream already active — idle timer refreshed.")
            return

        if play_cue:
            try:
                self._arm_local_audio_guard_for_wav(self.wake_sound_path)
                self._play_wav(self.wake_sound_path, block_until_played=False)
            except Exception:
                pass

        # self._start_mic_stream_thread(self._wsapp)
        self._last_voice_ts = time.time()
        self.MIC_STREAM_RUNNING = True
        if self.DEBUG_AUDIO_IN:
            print("[Wake] Mic stream started manually.")

# ---------------------- Minimal inline demo ---------------------- #
if __name__ == "__main__":
    # Minimal demo using the YAML config at CFG_PATH.

    def get_object_handler(description: str):
        print(f"[TOOL:get_object] {description}")
        time.sleep(0.2)
        return {"status": "ok", "action": "in_progress"}

    def stop_action_handler(reason: str = ""):
        print(f"[TOOL:stop_action] {reason=}")
        time.sleep(0.2)
        return {"status": "ok", "action": "done"}

    def camera_snapshot_handler(request: str, reason: str) -> Dict[str, Any]:
        return {"status": "ok", "sent_bytes": None}

    client = OpenAIRealtimeClient.load()
    client.register_tool_handler("get_object", get_object_handler)
    client.register_tool_handler("stop_action", stop_action_handler)
    client.register_tool_handler("camera_snapshot", camera_snapshot_handler)

    client.on_error = lambda e: print("[App] error:", e)
    client.on_text_delta = lambda s: None

    client.start()
    print("🎙️ Realtime client started. Type to chat. Commands: /snap <request>, /wake, /quit")

    def input_loop():
        try:
            while True:
                line = sys.stdin.readline()
                if not line:
                    time.sleep(0.05)
                    continue
                msg = line.strip()
                if not msg:
                    continue
                if msg.lower() in ("/q", "/quit", "quit", "exit"):
                    break
                if msg.startswith("/snap"):
                    request = msg[len("/snap"):].strip() or "the requested item"
                    res = camera_snapshot_handler(request=request, reason="manual keyboard snapshot")
                    if res.get("status") != "ok":
                        print("[/snap ERROR]", res)
                    continue
                try:
                    client.send_text(msg, role="user", speak=True)
                except Exception as e:
                    print("[send_text ERROR]", e)
        except KeyboardInterrupt:
            pass

    t = threading.Thread(target=input_loop, name="stdin-loop", daemon=True)
    t.start()

    try:
        while t.is_alive():
            t.join(timeout=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        print("\nShutting down...")
        try:
            client.stop()
        except Exception:
            pass
