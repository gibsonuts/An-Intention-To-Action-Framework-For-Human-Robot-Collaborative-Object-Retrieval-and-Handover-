from __future__ import annotations

import base64
import io
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image
import yaml


class OpenAiQueryClient:
    """
    Small reusable helper for OpenAI Responses API queries.

    Supports text-only or text+image queries, optional RGB re-encode for OpenCV images,
    and model fallback retries.
    """

    def __init__(
        self,
        openai_sdk_client: Any = None,
        *,
        default_model: str = "gpt-4.1",
        fallback_models: Optional[List[str]] = None,
        reencode_rgb: bool = True,
        jpeg_quality: int = 95,
        responses_create_defaults: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.openai_sdk_client = openai_sdk_client
        self.default_model = default_model
        self.fallback_models = fallback_models or ["gpt-5.2", "gpt-4.1", "gpt-4o", "gpt-4.1-mini"]
        self.reencode_rgb = bool(reencode_rgb)
        self.jpeg_quality = int(jpeg_quality)
        self.responses_create_defaults = dict(responses_create_defaults or {})
        self._client_init_error: Optional[str] = None

    @classmethod
    def from_config_dict(
        cls,
        cfg: Optional[Dict[str, Any]] = None,
        *,
        openai_sdk_client: Any = None,
    ) -> "OpenAiQueryClient":
        cfg = dict(cfg or {})
        return cls(
            openai_sdk_client=openai_sdk_client,
            default_model=str(cfg.get("default_model", "gpt-4.1")),
            fallback_models=list(cfg.get("fallback_models", ["gpt-5.2", "gpt-4.1", "gpt-4o", "gpt-4.1-mini"])),
            reencode_rgb=bool(cfg.get("reencode_rgb", True)),
            jpeg_quality=int(cfg.get("jpeg_quality", 95)),
            responses_create_defaults=dict(cfg.get("responses_create", {}) or {}),
        )

    def set_client(self, openai_sdk_client: Any) -> None:
        self.openai_sdk_client = openai_sdk_client
        self._client_init_error = None

    @staticmethod
    def _expand_env_str(value: str) -> str:
        return re.sub(
            r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}",
            lambda m: os.environ.get(m.group(1), ""),
            value,
        )

    @classmethod
    def _load_api_key_from_llm_config(cls) -> Optional[str]:
        cfg_path = Path(__file__).resolve().parent / "config" / "config.yaml"
        if not cfg_path.exists():
            return None
        try:
            with cfg_path.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            client_cfg = (
                cfg.get("realtime_client") or cfg.get("client", {})
            ) if isinstance(cfg, dict) else {}
            api_key = client_cfg.get("api_key") if isinstance(client_cfg, dict) else None
            if not api_key or not isinstance(api_key, str):
                return None
            api_key = cls._expand_env_str(api_key).strip()
            return api_key or None
        except Exception:
            return None

    def _ensure_client(self) -> bool:
        if self.openai_sdk_client is not None:
            return True
        try:
            from openai import OpenAI  # lazy import

            api_key = self._load_api_key_from_llm_config()
            self.openai_sdk_client = OpenAI(api_key=api_key) if api_key else OpenAI()
            self._client_init_error = None
            return True
        except Exception as e:
            self._client_init_error = str(e)
            return False

    def _normalize_image_bytes(self, image_bytes: bytes, *, mime_type: str = "image/jpeg") -> bytes:
        if not self.reencode_rgb:
            return image_bytes
        if mime_type.lower() not in ("image/jpeg", "image/jpg", "image/png"):
            return image_bytes
        try:
            dec = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if dec is None:
                return image_bytes
            rgb = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            buf = io.BytesIO()
            if mime_type.lower() == "image/png":
                pil_img.save(buf, format="PNG")
            else:
                pil_img.save(buf, format="JPEG", quality=self.jpeg_quality)
            return buf.getvalue()
        except Exception:
            return image_bytes

    @staticmethod
    def _to_data_uri(image_bytes: bytes, *, mime_type: str) -> str:
        return f"data:{mime_type};base64," + base64.b64encode(image_bytes).decode("ascii")

    def query(
        self,
        *,
        prompt_text: str,
        image_bytes: Optional[bytes] = None,
        image_items: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        mime_type: str = "image/jpeg",
        fallback_models: Optional[List[str]] = None,
        response_create_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self._ensure_client():
            reason = "OpenAI SDK client is not available."
            if self._client_init_error:
                reason += f" Auto-init failed: {self._client_init_error}"
            return {"ok": False, "reason": reason}

        final_model = model or self.default_model
        model_candidates: List[str] = []
        for name in [final_model] + list(fallback_models or self.fallback_models):
            if name and name not in model_candidates:
                model_candidates.append(name)

        data_uris: List[str] = []
        if image_bytes is not None:
            api_image_bytes = self._normalize_image_bytes(image_bytes, mime_type=mime_type)
            data_uris.append(self._to_data_uri(api_image_bytes, mime_type=mime_type))
        for item in list(image_items or []):
            if not isinstance(item, dict):
                continue
            item_bytes = item.get("image_bytes")
            if item_bytes is None:
                continue
            item_mime_type = str(item.get("mime_type", mime_type))
            api_image_bytes = self._normalize_image_bytes(item_bytes, mime_type=item_mime_type)
            data_uris.append(self._to_data_uri(api_image_bytes, mime_type=item_mime_type))

        resp = None
        model_used = None
        model_errors: List[str] = []
        for model_name in model_candidates:
            try:
                content = [{"type": "input_text", "text": prompt_text}]
                for data_uri in data_uris:
                    content.append({"type": "input_image", "image_url": data_uri})
                create_kwargs = dict(self.responses_create_defaults)
                create_kwargs.update(dict(response_create_kwargs or {}))
                create_kwargs = {
                    k: v
                    for k, v in create_kwargs.items()
                    if v is not None and not (k == "tools" and isinstance(v, list) and len(v) == 0)
                }
                create_kwargs["model"] = model_name
                create_kwargs["input"] = [{"role": "user", "content": content}]
                resp = self.openai_sdk_client.responses.create(**create_kwargs)
                model_used = model_name
                break
            except Exception as e:
                msg = str(e)
                model_errors.append(f"{model_name}: {msg}")
                if "model_not_found" in msg or "does not exist" in msg:
                    continue
                return {
                    "ok": False,
                    "reason": f"Query failed: {e}",
                    "requested_model": final_model,
                    "tried_models": model_candidates,
                    "errors": model_errors,
                }

        if resp is None:
            return {
                "ok": False,
                "reason": "Query failed: no available model from fallback list.",
                "requested_model": final_model,
                "tried_models": model_candidates,
                "errors": model_errors,
            }

        return {
            "ok": True,
            "response": resp,
            "output_text": getattr(resp, "output_text", "") or "",
            "model_used": model_used or final_model,
            "requested_model": final_model,
            "tried_models": model_candidates,
            "errors": model_errors,
        }

    def query_image(
        self,
        *,
        prompt_text: str,
        image_bytes: Optional[bytes] = None,
        image_items: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        mime_type: str = "image/jpeg",
        fallback_models: Optional[List[str]] = None,
        response_create_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self.query(
            prompt_text=prompt_text,
            image_bytes=image_bytes,
            image_items=image_items,
            model=model,
            mime_type=mime_type,
            fallback_models=fallback_models,
            response_create_kwargs=response_create_kwargs,
        )


__all__ = ["OpenAiQueryClient"]
