#!/usr/bin/env python3
"""
Generic SAM3 detection utilities.

Usage example:

    from sam3_detector import Sam3Detector, filter_masks_by_overlap

    detector = Sam3Detector(device="cuda")
    masks = detector.segment(color_bgr, "screw head", category="screw head")
"""

from contextlib import nullcontext
import base64
import os
import re
import time
from typing import List, Dict, Any, Tuple, Optional, Union

import numpy as np
import cv2
from PIL import Image
import requests


# ============================================================
# Low-level helpers (mask / geometry)
# ============================================================

def mask_to_contour(
    mask: np.ndarray,
    min_area: int = 50,
    dilate_iters: int = 1,
    close_iters: int = 1,
) -> Optional[np.ndarray]:
    """
    Convert a boolean mask to a single, blob-like contour.
    - fills holes/small gaps via morphology
    - returns convex hull of largest component
    """
    mask = np.squeeze(mask)

    if mask.ndim != 2:
        raise ValueError(f"mask_to_contour expected 2D mask, got {mask.shape}")

    mask_u8 = (mask.astype(bool) * 255).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)

    if close_iters > 0:
        mask_u8 = cv2.morphologyEx(
            mask_u8, cv2.MORPH_CLOSE, kernel, iterations=close_iters
        )

    if dilate_iters > 0:
        mask_u8 = cv2.dilate(mask_u8, kernel, iterations=dilate_iters)

    contours, _ = cv2.findContours(
        mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < min_area:
        return None

    return cv2.convexHull(contour)


def masks_overlap(m1: np.ndarray, m2: np.ndarray) -> bool:
    """
    Check if two segmentation masks overlap in image space.
    Each mask is expected to be a boolean or 0/1 array with the same HxW.
    """
    m1 = np.squeeze(m1).astype(bool)
    m2 = np.squeeze(m2).astype(bool)

    if m1.ndim != 2 or m2.ndim != 2:
        raise ValueError(f"masks_overlap expected 2D masks, got {m1.shape}, {m2.shape}")

    if m1.shape != m2.shape:
        raise ValueError(f"masks_overlap shape mismatch: {m1.shape} vs {m2.shape}")

    return bool(np.any(m1 & m2))


def mask_centroid(mask: np.ndarray) -> Tuple[float, float]:
    """
    Compute centroid (u, v) of a 2D mask in pixel space.

    Returns:
        (u, v) = (x, y) pixel coordinates as floats.
    """
    mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise ValueError(f"mask_centroid expected 2D mask, got {mask.shape}")
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        raise ValueError("mask_centroid: empty mask (no True pixels)")
    return float(xs.mean()), float(ys.mean())


def project_pixel_to_cam(
    u: float,
    v: float,
    depth_value: float,
    intr: Dict[str, float],
) -> np.ndarray:
    """
    Back-project a pixel (u, v) with depth into camera coordinates.

    intr: dict with fx, fy, cx, cy
    depth_value: depth at (u, v) in meters.
    """
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    x = (u - cx) * depth_value / fx
    y = (v - cy) * depth_value / fy
    z = depth_value
    return np.array([x, y, z], dtype=np.float32)


def apply_mask_overlay(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    color_bgr: Tuple[int, int, int],
    alpha_fg: float = 0.6,
) -> np.ndarray:
    """
    Simple colored overlay for visualization (BGR color).
    """
    mask = np.squeeze(mask).astype(bool)
    overlay = image_bgr.copy().astype(np.float32)
    overlay[mask] = overlay[mask] * (1.0 - alpha_fg) + np.array(color_bgr, np.float32) * alpha_fg
    return overlay.astype(image_bgr.dtype)


def draw_mask_debug(
    image_bgr: np.ndarray,
    masks: List[Dict[str, Any]],
    centroids: Optional[List[Tuple[float, float]]] = None,
    output_path: str = "data/image_samples/sam3_debug.png",
    category_colors: Optional[Dict[str, Tuple[int, int, int]]] = None,
):
    """
    Draw mask overlay + contour + label and optionally save the visualization.

    Returns:
        Annotated image in BGR format.
    """
    if centroids is None:
        centroids = [mask_centroid(m["segmentation"]) for m in masks]

    if category_colors is None:
        category_colors = {}

    vis = image_bgr.copy()

    for m, (cx, cy) in zip(masks, centroids):
        mask = m["segmentation"]
        cat = m.get("category", "obj")
        color = category_colors.get(cat, (0, 0, 255))

        vis = apply_mask_overlay(vis, mask, color)

        contour = m.get("contour", None)

        if contour is not None:
            cv2.drawContours(vis, [contour], -1, color, 2)
            x, y, w, h = cv2.boundingRect(contour)
            cv2.circle(vis, (int(cx), int(cy)), 5, (255, 255, 255), -1)
        # else:
        x, y, w, h = m["bbox"]
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)

        cv2.putText(vis, cat, (x, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imwrite(output_path, vis)
    print(f"[DEBUG] Saved mask debug image to: {output_path}")

    return vis   # <--- REQUIRED


# ============================================================
# Extra helpers for bbox math (used by refine_boxes_with_str_prompts)
# ============================================================

def _round(x: Union[int, float]) -> int:
    return int(round(float(x)))


def _to_xyxy_from_boxdict(b: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    """
    Convert a bbox dict into (x1, y1, x2, y2).

    Supported formats (all length-4 lists/tuples):
      - 'xyxy': [x1, y1, x2, y2]
      - 'xywh': [x, y, w, h]
      - 'bbox': [x, y, w, h]   # <- your current format

    Returns None if no supported key is present.
    """
    # Already in xyxy
    if "xyxy" in b and b["xyxy"] is not None:
        x1, y1, x2, y2 = b["xyxy"]
        return float(x1), float(y1), float(x2), float(y2)

    # xywh or bbox (both treated as xywh)
    for key in ("xywh", "bbox"):
        if key in b and b[key] is not None:
            x, y, w, h = b[key]
            x1 = float(x)
            y1 = float(y)
            x2 = x1 + float(w)
            y2 = y1 + float(h)
            return x1, y1, x2, y2

    return None


def _iou(
    b1_xyxy: Tuple[float, float, float, float],
    b2_xyxy: Tuple[float, float, float, float],
) -> float:
    """
    IoU of two boxes in (x1, y1, x2, y2) format.
    """
    x1a, y1a, x2a, y2a = b1_xyxy
    x1b, y1b, x2b, y2b = b2_xyxy

    inter_x1 = max(x1a, x1b)
    inter_y1 = max(y1a, y1b)
    inter_x2 = min(x2a, x2b)
    inter_y2 = min(y2a, y2b)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = (x2a - x1a) * (y2a - y1a)
    area_b = (x2b - x1b) * (y2b - y1b)

    union = max(area_a + area_b - inter_area, 1e-6)
    return float(inter_area / union)


def _area_ratio_xyxy(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    W: int,
    H: int,
) -> float:
    """
    Area of box (x1,y1,x2,y2) divided by total image area W*H.
    """
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    box_area = w * h
    img_area = float(max(W * H, 1))
    return float(box_area / img_area)


def filter_masks_by_non_overlap(
    sources: List[Dict[str, Any]],
    targets: List[Dict[str, Any]],
    use_mask_overlap: bool = True,
    use_centroid_in_contour: bool = True,
    min_contour_area: int = 50,
    filter_side: str = "sources",
) -> List[Dict[str, Any]]:
    """
    Keep only masks from the chosen side (`filter_side`) that do NOT overlap
    with any mask on the other side.

    Non-overlap means:
      - No binary mask pixel overlap (if use_mask_overlap=True), AND
      - Centroid is NOT inside any reference contour (if use_centroid_in_contour=True).

    Args:
        sources: first list of mask dicts.
        targets: second list of mask dicts.
        use_mask_overlap: enable/disable pixel-wise mask overlap test.
        use_centroid_in_contour: enable/disable centroid-in-contour test.
        min_contour_area: minimum contour area to consider valid.
        filter_side: which list to filter:
            - "sources":  keep non-overlapping masks from `sources` (default)
            - "targets":  keep non-overlapping masks from `targets`

    Returns:
        List of mask dicts from the side specified by `filter_side`.
    """
    if filter_side not in ("sources", "targets"):
        raise ValueError(f"filter_side must be 'sources' or 'targets', got: {filter_side}")

    if not sources or not targets:
        # Nothing to compare against -> return the requested side unchanged
        return sources if filter_side == "sources" else targets

    # Decide which list is being filtered (candidates) and which is reference
    if filter_side == "sources":
        candidates = sources
        refs = targets
        side_name = "sources"
    else:
        candidates = targets
        refs = sources
        side_name = "targets"

    # Ensure reference masks have valid contours
    valid_refs = []
    for r in refs:
        contour = r.get("contour", None)
        if contour is None:
            contour = mask_to_contour(
                r["segmentation"],
                min_area=min_contour_area,
                dilate_iters=1,
                close_iters=1,
            )
            r["contour"] = contour

        if contour is not None and cv2.contourArea(contour) >= min_contour_area:
            valid_refs.append(r)

    if not valid_refs:
        print(f"[WARN] filter_masks_by_non_overlap: no valid reference contours; returning all {side_name}.")
        return candidates

    filtered: List[Dict[str, Any]] = []

    for c in candidates:
        c_mask = c["segmentation"]
        cx, cy = mask_centroid(c_mask)

        has_overlap = False

        for r in valid_refs:
            r_mask = r["segmentation"]

            # Pixel overlap test
            if use_mask_overlap and masks_overlap(c_mask, r_mask):
                has_overlap = True
                break

            # Centroid-in-contour test
            if use_centroid_in_contour:
                contour = r["contour"]
                inside = cv2.pointPolygonTest(contour, (cx, cy), False)
                if inside >= 0:
                    has_overlap = True
                    break

        if not has_overlap:
            filtered.append(c)

    print(
        f"[INFO] Non-overlap filter ({side_name}): kept {len(filtered)} "
        f"out of {len(candidates)}"
    )
    return filtered

def filter_masks_by_overlap(
    heads: List[Dict[str, Any]],
    stems: List[Dict[str, Any]],
    min_contour_area: int = 50,
) -> List[Dict[str, Any]]:
    """
    Keep only head masks whose centroid lies inside the contour of
    at least one stem mask.
    """
    if not heads or not stems:
        return []

    # Ensure stems have contours
    valid_stems = []
    for s in stems:
        contour = s.get("contour", None)
        if contour is None:
            # Fall back to computing contour if missing
            contour = mask_to_contour(
                s["segmentation"],
                min_area=min_contour_area,
                dilate_iters=1,
                close_iters=1,
            )
            s["contour"] = contour

        if contour is not None and cv2.contourArea(contour) >= min_contour_area:
            valid_stems.append(s)

    if not valid_stems:
        print("[WARN] filter_heads_inside_stems: no valid stem contours.")
        return []

    filtered_heads: List[Dict[str, Any]] = []

    for head in heads:
        h_mask = head["segmentation"]
        cx, cy = mask_centroid(h_mask)

        keep = False
        for stem in valid_stems:
            contour = stem["contour"]
            inside = cv2.pointPolygonTest(contour, (cx, cy), False)
            if inside >= 0:
                keep = True
                break

        if keep:
            filtered_heads.append(head)

    print(
        f"[INFO] Heads-in-stem-contours filter: "
        f"{len(heads)} -> {len(filtered_heads)} kept"
    )
    return filtered_heads


# ============================================================
# Generic SAM3 Detection Class
# ============================================================

class Sam3Detector:
    """
    Generic wrapper around SAM3 for text-prompt segmentation.

    Features:
      - text-prompt segmentation (single or multiple prompts)
      - returns masks with bbox + contour
      - helper to pick closest mask by depth
      - helper to compute 3D target in camera frame
      - helper to "refine" bboxes using text prompts + IoU (GroundingDINO-like)
    """
    def __init__(
        self,
        device: str = "cuda",
        default_conf_threshold: float = 0.2,
        backend: Optional[str] = None,
        roboflow_api_key: Optional[str] = None,
        roboflow_api_key_env: str = "ROBOFLOW_API_KEY",
        roboflow_api_url: str = "https://serverless.roboflow.com/sam3/concept_segment",
        roboflow_model_id: str = "sam3/sam3_final",
        roboflow_timeout: float = 60.0,
        roboflow_jpeg_quality: int = 95,
    ):
        self.backend = str(backend or os.getenv("SAM3_BACKEND", "local")).strip().lower()
        if self.backend not in ("local", "roboflow"):
            raise ValueError("SAM3 backend must be either 'local' or 'roboflow'")

        self.device = device
        self.default_conf_threshold = float(default_conf_threshold)
        self.roboflow_api_url = str(roboflow_api_url).rstrip("?")
        self.roboflow_model_id = str(roboflow_model_id)
        self.roboflow_timeout = max(1.0, float(roboflow_timeout))
        self.roboflow_jpeg_quality = int(np.clip(roboflow_jpeg_quality, 1, 100))
        self.roboflow_api_key_env = str(roboflow_api_key_env or "ROBOFLOW_API_KEY")
        if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", self.roboflow_api_key_env):
            raise ValueError(
                "sam3.roboflow.api_key_env must be an environment-variable name "
                "such as ROBOFLOW_API_KEY, not the API key itself."
            )
        self.roboflow_api_key = roboflow_api_key or os.getenv(self.roboflow_api_key_env)

        if self.backend == "roboflow":
            if not self.roboflow_api_key:
                raise RuntimeError(
                    "Roboflow SAM3 is selected but no API key is available. "
                    "Set the environment variable named by "
                    "sam3.roboflow.api_key_env (normally ROBOFLOW_API_KEY)."
                )
            self._roboflow_session = requests.Session()
            self.processor = None
            self._amp_dtype = None
            print(
                "[INFO] SAM3 backend=roboflow; inference will run through "
                f"{self.roboflow_api_url}."
            )
            return

        # Keep SAM3's newer PyTorch modules out of processes that only use the
        # lightweight geometry helpers in this file.
        import torch
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        self._torch = torch
        self._amp_dtype = self._torch.bfloat16 if device == "cuda" else None

        print(f"[INFO] Building SAM3 image model on {device}...")
        model = build_sam3_image_model()
        model.to(device)
        model.eval()

        self.processor = Sam3Processor(
            model,
            confidence_threshold=default_conf_threshold,
        )
        print(f"[INFO] SAM3 model loaded with default_conf_threshold={default_conf_threshold}")

        # ------------------------------------------------------
        # 🔥 WARM UP SAM3 (important for first-run performance)
        # ------------------------------------------------------
        try:
            print("[INFO] Warming up SAM3 model...")

            # Create a dummy RGB image: 256×256 solid color
            dummy_image = np.zeros((256, 256, 3), dtype=np.uint8)
            dummy_pil = Image.fromarray(dummy_image)

            # Encode dummy image
            warm_state = self._run_sam3_set_image(dummy_pil)

            # Run a dummy prompt through the model
            _ = self._run_sam3_text_prompt(state=warm_state, prompt="object")

            print("[INFO] Warm-up completed.")
        except Exception as e:
            print(f"[WARN] Warm-up failed (not fatal): {e}")

    @classmethod
    def from_config(cls, config: Optional[Dict[str, Any]] = None) -> "Sam3Detector":
        """Build a detector from the ``sam3`` section in cycles.yaml."""
        config = config or {}
        roboflow = config.get("roboflow", {}) or {}
        return cls(
            backend=config.get("backend"),
            device=str(config.get("device", "cuda")),
            default_conf_threshold=float(config.get("default_conf_threshold", 0.2)),
            roboflow_api_key=roboflow.get("api_key") or None,
            roboflow_api_key_env=str(roboflow.get("api_key_env", "ROBOFLOW_API_KEY")),
            roboflow_api_url=str(
                roboflow.get(
                    "api_url",
                    "https://serverless.roboflow.com/sam3/concept_segment",
                )
            ),
            roboflow_model_id=str(roboflow.get("model_id", "sam3/sam3_final")),
            roboflow_timeout=float(roboflow.get("timeout_s", 60.0)),
            roboflow_jpeg_quality=int(roboflow.get("jpeg_quality", 95)),
        )

    def _inference_context(self):
        if self.device == "cuda":
            return self._torch.autocast("cuda", dtype=self._amp_dtype)
        return nullcontext()

    def _run_sam3_set_image(self, image_pil: Image.Image):
        with self._torch.inference_mode(), self._inference_context():
            return self.processor.set_image(image_pil)

    def _run_sam3_text_prompt(self, state: Any, prompt: str):
        with self._torch.inference_mode(), self._inference_context():
            return self.processor.set_text_prompt(
                state=state,
                prompt=prompt,
            )

    def _tensor_to_numpy(self, tensor: Any, *, dtype: Optional[Any] = None) -> np.ndarray:
        if not self._torch.is_tensor(tensor):
            return np.asarray(tensor)

        tensor = tensor.detach()
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        return tensor.cpu().numpy()

    @staticmethod
    def _mask_result(
        mask: np.ndarray,
        bbox_xyxy: Tuple[float, float, float, float],
        score: float,
        text_prompt: str,
        category: Optional[str],
        orientation_align: str,
    ) -> Optional[Dict[str, Any]]:
        """Convert a binary mask into the result shape used by the robot pipeline."""
        contour = mask_to_contour(mask)
        if contour is None or len(contour) < 3:
            return None

        (cx, cy), (w_raw, h_raw), angle_raw = cv2.minAreaRect(contour)
        width = float(w_raw)
        height = float(h_raw)
        angle = float(angle_raw)

        def normalize_angle(value: float) -> float:
            if value <= -90.0:
                value += 180.0
            elif value > 90.0:
                value -= 180.0
            return value

        if orientation_align == "long":
            if height > width:
                aligned_angle = angle + 90.0
                aligned_width, aligned_height = height, width
            else:
                aligned_angle = angle
                aligned_width, aligned_height = width, height
        elif orientation_align == "short":
            if width > height:
                aligned_angle = angle + 90.0
                aligned_width, aligned_height = height, width
            else:
                aligned_angle = angle
                aligned_width, aligned_height = width, height
        else:
            raise ValueError(f"Invalid orientation_align: {orientation_align}")

        box_points = cv2.boxPoints(
            ((cx, cy), (aligned_width, aligned_height), aligned_angle)
        )
        x0, y0, x1, y1 = bbox_xyxy
        x0_i, y0_i = int(x0), int(y0)
        x1_i, y1_i = int(x1), int(y1)
        result: Dict[str, Any] = {
            "segmentation": np.squeeze(mask).astype(bool),
            "bbox": [x0_i, y0_i, max(1, x1_i - x0_i), max(1, y1_i - y0_i)],
            "score": float(score),
            "phrase": text_prompt,
            "contour": contour,
            "center": (float(cx), float(cy)),
            "angle_deg": normalize_angle(aligned_angle),
            "rect_wh": (float(aligned_width), float(aligned_height)),
            "oriented_box": box_points.astype(np.float32),
        }
        if category is not None:
            result["category"] = category
        return result

    def _roboflow_segment_prompts(
        self,
        image_rgb: np.ndarray,
        prompts: List[str],
        confidence_thresholds: List[Optional[float]],
        categories: List[Optional[str]],
        orientation_align: str,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Run Roboflow's hosted SAM3 concept-segmentation endpoint."""
        if not prompts:
            return {}

        thresholds = [
            float(self.default_conf_threshold if value is None else value)
            for value in confidence_thresholds
        ]
        request_threshold = float(np.clip(min(thresholds), 0.0, 1.0))
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        encoded, buffer = cv2.imencode(
            ".jpg",
            image_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.roboflow_jpeg_quality],
        )
        if not encoded:
            raise RuntimeError("Failed to JPEG-encode the image for Roboflow SAM3")

        payload = {
            "image": {
                "type": "base64",
                "value": base64.b64encode(buffer).decode("ascii"),
            },
            "prompts": [{"type": "text", "text": prompt} for prompt in prompts],
            "output_prob_thresh": request_threshold,
            "format": "polygon",
            "model_id": self.roboflow_model_id,
        }
        response = None
        started = time.monotonic()
        try:
            response = self._roboflow_session.post(
                self.roboflow_api_url,
                params={"api_key": self.roboflow_api_key},
                json=payload,
                timeout=self.roboflow_timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            status = getattr(response, "status_code", None)
            detail = f" HTTP {status}" if status is not None else ""
            raise RuntimeError(
                f"Roboflow SAM3 request failed{detail} ({type(exc).__name__})"
            ) from None

        try:
            data = response.json()
        except ValueError:
            raise RuntimeError("Roboflow SAM3 returned a non-JSON response") from None
        if not isinstance(data, dict):
            raise RuntimeError("Roboflow SAM3 returned an invalid JSON response")
        prompt_results = data.get("prompt_results")
        if not isinstance(prompt_results, list):
            raise RuntimeError(
                "Roboflow SAM3 response is missing the `prompt_results` list"
            )

        height, width = image_rgb.shape[:2]
        results: Dict[str, List[Dict[str, Any]]] = {prompt: [] for prompt in prompts}
        for prompt_result in prompt_results:
            if not isinstance(prompt_result, dict):
                continue
            index = prompt_result.get("prompt_index")
            if index is None:
                index = (prompt_result.get("echo") or {}).get("prompt_index")
            try:
                index = int(index)
            except (TypeError, ValueError):
                continue
            if index < 0 or index >= len(prompts):
                continue

            for prediction in prompt_result.get("predictions", []) or []:
                if not isinstance(prediction, dict):
                    continue
                score = float(prediction.get("confidence", 0.0))
                if score < thresholds[index]:
                    continue
                segmentation = np.zeros((height, width), dtype=np.uint8)
                for polygon in prediction.get("masks", []) or []:
                    points = np.asarray(polygon, dtype=np.float32)
                    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] < 2:
                        continue
                    points = np.rint(points[:, :2]).astype(np.int32)
                    points[:, 0] = np.clip(points[:, 0], 0, max(width - 1, 0))
                    points[:, 1] = np.clip(points[:, 1], 0, max(height - 1, 0))
                    cv2.fillPoly(segmentation, [points], 1)
                if not np.any(segmentation):
                    continue
                ys, xs = np.where(segmentation)
                formatted = self._mask_result(
                    segmentation.astype(bool),
                    (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)),
                    score,
                    prompts[index],
                    categories[index],
                    orientation_align,
                )
                if formatted is not None:
                    results[prompts[index]].append(formatted)

        elapsed = time.monotonic() - started
        server_time = data.get("time")
        print(
            f"[INFO] Roboflow SAM3 completed {len(prompts)} prompt(s) in "
            f"{elapsed:.2f}s (server_time={server_time!r})."
        )
        for prompt in prompts:
            print(
                f"[INFO] SAM3 found {len(results[prompt])} instances for prompt "
                f"'{prompt}' via Roboflow"
            )
        return results

    # ---------------- core segmentation ----------------
    def _segment_single_prompt(
        self,
        image_rgb: np.ndarray,
        text_prompt: str,
        confidence_threshold: Optional[float] = None,
        category: Optional[str] = None,
        orientation_align: str = "long",  
    ) -> List[Dict[str, Any]]:
        """
        Internal helper: run SAM3 on an RGB image for a single prompt.
        """
        if self.backend == "roboflow":
            return self._roboflow_segment_prompts(
                image_rgb,
                [text_prompt],
                [confidence_threshold],
                [category],
                orientation_align,
            )[text_prompt]

        image_pil = Image.fromarray(image_rgb)

        # Encode image
        state = self._run_sam3_set_image(image_pil)

        # Optionally override confidence threshold for this call
        if confidence_threshold is not None:
            self.processor.confidence_threshold = confidence_threshold
        else:
            self.processor.confidence_threshold = self.default_conf_threshold

        # Text prompt -> segmentation
        output = self._run_sam3_text_prompt(state=state, prompt=text_prompt)

        masks = output["masks"]   # [N, H, W]
        boxes = output["boxes"]   # [N, 4] (x0, y0, x1, y1)
        scores = output["scores"] # [N]

        if masks is None or len(masks) == 0:
            print(f"[WARN] SAM3: no masks found for prompt: '{text_prompt}'")
            return []

        masks_np = self._tensor_to_numpy(masks).astype(bool)
        boxes_np = self._tensor_to_numpy(boxes, dtype=self._torch.float32)
        scores_np = self._tensor_to_numpy(scores, dtype=self._torch.float32)

        results: List[Dict[str, Any]] = []

        for i in range(masks_np.shape[0]):
            x0, y0, x1, y1 = boxes_np[i]
            x0_i, y0_i, x1_i, y1_i = int(x0), int(y0), int(x1), int(y1)
            w_i = max(1, x1_i - x0_i)
            h_i = max(1, y1_i - y0_i)

            mask_i = masks_np[i]

            # -------------------------------------------------
            # Contour + oriented box / rotation
            # -------------------------------------------------
            contour = mask_to_contour(mask_i)

            angle_deg = None
            center = None
            rect_wh = None
            oriented_box = None

            if contour is not None and len(contour) >= 3:
                (cx, cy), (w_raw, h_raw), angle_raw = cv2.minAreaRect(contour)

                w = float(w_raw)
                h = float(h_raw)
                angle = float(angle_raw)

                def _normalize_angle(a: float) -> float:
                    # Normalize into [-90, 90) for convenience
                    if a <= -90.0:
                        a += 180.0
                    elif a > 90.0:
                        a -= 180.0
                    return a

                if orientation_align == "long":
                    # angle_deg should align with the LONGER side
                    if h > w:
                        angle_new = angle + 90.0
                        long_side = h
                        short_side = w
                    else:
                        angle_new = angle
                        long_side = w
                        short_side = h

                    angle_deg = _normalize_angle(angle_new)
                    rect_wh = (float(long_side), float(short_side))
                    box_pts = cv2.boxPoints(((cx, cy), (long_side, short_side), angle_new))

                elif orientation_align == "short":
                    # angle_deg should align with the SHORTER side
                    if w > h:
                        angle_new = angle + 90.0
                        short_side = h
                        long_side = w
                    else:
                        angle_new = angle
                        short_side = w
                        long_side = h

                    angle_deg = _normalize_angle(angle_new)
                    rect_wh = (float(short_side), float(long_side))
                    box_pts = cv2.boxPoints(((cx, cy), (short_side, long_side), angle_new))

                else:
                    raise ValueError(f"Invalid orientation_align: {orientation_align}")

                center = (float(cx), float(cy))
                oriented_box = box_pts.astype(np.float32)


                m: Dict[str, Any] = {
                    "segmentation": mask_i,
                    "bbox": [x0_i, y0_i, w_i, h_i],  # axis-aligned xywh
                    "score": float(scores_np[i]),
                    "phrase": text_prompt,
                    "contour": contour,
                    # New orientation-related fields:
                    "center": center,               # (cx, cy) in pixels
                    "angle_deg": angle_deg,         # in-plane rotation
                    "rect_wh": rect_wh,             # (w, h) of oriented box
                    "oriented_box": oriented_box,   # 4 points of rotated rect
                }

                if category is not None:
                    m["category"] = category

                results.append(m)

        print(f"[INFO] SAM3 found {len(results)} instances for prompt '{text_prompt}'")
        return results

    def segment(
        self,
        color_bgr: np.ndarray,
        text_prompt: str,
        confidence_threshold: Optional[float] = None,
        category: Optional[str] = None,
        orientation_align: str = "long",
    ) -> List[Dict[str, Any]]:
        """
        Segment objects using a text prompt.

        Args:
            color_bgr: OpenCV BGR image (H, W, 3).
            text_prompt: text query for SAM3.
            confidence_threshold: optional override threshold for this prompt.
            category: optional label to attach to returned masks.

        Returns:
            List of mask dicts.
        """
        print('prompt',text_prompt,'conf',confidence_threshold)
        image_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        return self._segment_single_prompt(
            image_rgb=image_rgb,
            text_prompt=text_prompt,
            confidence_threshold=confidence_threshold,
            category=category,
            orientation_align=orientation_align,
        )

    def segment_with_annotated_image(
        self,
        color_bgr: np.ndarray,
        text_prompt: str,
        confidence_threshold: Optional[float] = None,
        category: Optional[str] = None,
        orientation_align: str = "long",
        output_path: Optional[str] = "data/image_samples/sam3_segment_annotated.png",
        category_colors: Optional[Dict[str, Tuple[int, int, int]]] = None,
        show_window: bool = False,
        window_name: str = "SAM3 Annotated",
        wait_ms: int = 0,
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Segment objects and produce an annotated BGR image with contour + label.

        Returns:
            (masks, annotated_bgr)
        """
        masks = self.segment(
            color_bgr=color_bgr,
            text_prompt=text_prompt,
            confidence_threshold=confidence_threshold,
            category=category,
            orientation_align=orientation_align,
        )
        annotated = draw_mask_debug(
            image_bgr=color_bgr,
            masks=masks,
            output_path=output_path if output_path is not None else "data/image_samples/sam3_segment_annotated.png",
            category_colors=category_colors,
        )
        if show_window:
            cv2.imshow(window_name, annotated)
            cv2.waitKey(int(wait_ms))
        return masks, annotated

    def segment_multi(
        self,
        color_bgr: np.ndarray,
        prompts: List[str],
        conf_thresholds: Optional[List[Optional[float]]] = None,
        categories: Optional[List[Optional[str]]] = None,
        orientation_align: str = "long",
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run multiple text prompts on the same image.

        Args:
            color_bgr: OpenCV BGR image.
            prompts: list of text prompts.
            conf_thresholds: list of thresholds (same length or None).
            categories: list of category labels (same length or None).

        Returns:
            dict: prompt -> list of mask dicts
        """
        image_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

        if conf_thresholds is None:
            conf_thresholds = [None] * len(prompts)
        if categories is None:
            categories = [None] * len(prompts)

        if len(conf_thresholds) != len(prompts):
            raise ValueError("conf_thresholds must have the same length as prompts")
        if len(categories) != len(prompts):
            raise ValueError("categories must have the same length as prompts")
        if self.backend == "roboflow":
            return self._roboflow_segment_prompts(
                image_rgb,
                prompts,
                conf_thresholds,
                categories,
                orientation_align,
            )

        results: Dict[str, List[Dict[str, Any]]] = {}
        for prompt, conf, cat in zip(prompts, conf_thresholds, categories):
            masks = self._segment_single_prompt(
                image_rgb=image_rgb,
                text_prompt=prompt,
                confidence_threshold=conf,
                category=cat,
                orientation_align=orientation_align,
            )
            results[prompt] = masks
        return results

    # ---------------- depth-based utilities ----------------

    def pick_closest_by_depth(
        self,
        candidates: List[Dict[str, Any]],
        depth: np.ndarray,
        depth_scale: float = 1.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Pick the candidate whose mask has the smallest median depth.

        Args:
            candidates: list of mask dicts.
            depth: depth map aligned with the masks (H, W).
            depth_scale: factor to convert raw depth units to meters.

        Returns:
            dict:
                {
                  "mask": chosen mask dict,
                  "depth_raw": raw median depth (same units as depth),
                  "depth_m": median depth in meters
                }
            or None if no valid candidate.
        """
        if not candidates:
            return None

        best_mask = None
        best_depth = float("inf")

        for c in candidates:
            mask = np.squeeze(c["segmentation"])
            if mask.ndim != 2:
                raise ValueError(f"Expected 2D mask, got {mask.shape}")

            ys, xs = np.where(mask)
            if xs.size == 0 or ys.size == 0:
                continue

            depth_vals = depth[ys, xs]
            depth_vals = depth_vals[depth_vals > 0]
            if depth_vals.size == 0:
                continue

            d = float(np.median(depth_vals))
            if d < best_depth:
                best_depth = d
                best_mask = c

        if best_mask is None:
            print("[WARN] pick_closest_by_depth: all candidates had invalid depth.")
            return None

        depth_m = best_depth * depth_scale
        print(f"[DEBUG] Closest object depth: raw={best_depth}, meters={depth_m}")
        return {
            "mask": best_mask,
            "depth": best_depth,
            "depth_m": depth_m,
        }

    def compute_target_from_mask(
        self,
        mask: Dict[str, Any],
        depth: np.ndarray,
        intr: Dict[str, float],
        depth_scale: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Compute 3D target in camera frame using mask centroid.
        """
        seg = np.squeeze(mask["segmentation"]).astype(bool)
        if seg.ndim != 2:
            raise ValueError(f"compute_target_from_mask expected 2D mask, got {seg.shape}")

        preferred = mask.get("target_pixel", None)
        if preferred is not None and len(preferred) >= 2:
            u = float(preferred[0])
            v = float(preferred[1])
            target_src = mask.get("target_pixel_source", "target_pixel")
        else:
            u, v = mask_centroid(seg)
            target_src = "centroid"

        H, W = depth.shape[:2]
        u_int = int(np.clip(round(u), 0, max(W - 1, 0)))
        v_int = int(np.clip(round(v), 0, max(H - 1, 0)))

        depth_raw = float(depth[v_int, u_int])
        depth_m = depth_raw * depth_scale

        # If preferred point is invalid, pick nearest valid depth pixel inside the mask.
        if depth_m <= 0:
            ys, xs = np.where(seg)
            if xs.size > 0:
                xs_i = np.clip(xs.astype(int), 0, max(W - 1, 0))
                ys_i = np.clip(ys.astype(int), 0, max(H - 1, 0))
                dvals = depth[ys_i, xs_i].astype(np.float32) * float(depth_scale)
                valid = dvals > 0
                if np.any(valid):
                    xs_v = xs_i[valid].astype(np.float32)
                    ys_v = ys_i[valid].astype(np.float32)
                    d2 = (xs_v - float(u)) ** 2 + (ys_v - float(v)) ** 2
                    j = int(np.argmin(d2))
                    u_int = int(xs_v[j])
                    v_int = int(ys_v[j])
                    u = float(u_int)
                    v = float(v_int)
                    depth_raw = float(depth[v_int, u_int])
                    depth_m = depth_raw * depth_scale
                    target_src = f"{target_src}_nearest_valid_depth"

        if depth_m <= 0:
            print(
                f"Invalid depth at target point: raw={depth_raw}, meters={depth_m}, "
                f"v_int={v_int}, u_int={u_int}, source={target_src}"
            )
            return None

        target_cam = project_pixel_to_cam(u, v, depth_m, intr)
        print(f"[DEBUG] Target (camera frame): {target_cam} source={target_src} pixel=({u:.1f}, {v:.1f})")

        return {
            "target_cam": target_cam,
            "pixel": (u, v),
            "depth_m": depth_m,
            'angle_cam': mask["angle_deg"],
            "target_pixel_source": target_src,
        }

    def detect_bbox(
        self,
        prompts: List[str],
        color: np.ndarray,
        conf_list:  List[float],
        debug: bool = False
    ) -> List[Dict[str, Any]]:

        seg_dict = self.segment_multi(
            color,
            prompts=prompts,
            conf_thresholds=conf_list,  # use default per prompt
            categories=prompts,
        )

        new_bboxes = []
        for prompt, masks in seg_dict.items():
            for m in masks:
                print(f"[DEBUG] prompt '{prompt}': {len(masks)} masks")        
                new_bboxes.append(
                    {
                        "label": m.get("category", prompt),
                        "confidence": float(m["score"]),
                        "xywh": m["bbox"],
                        "center": m.get("center"),
                        "angle_deg": m.get("angle_deg"),
                        "rect_wh": m.get("rect_wh"),
                        "oriented_box": m.get("oriented_box"),
                    }
                )

      
        return new_bboxes
    

    # ---------------- NEW: refine_boxes_with_str_prompts ----------------
    def refine_boxes_with_str_prompts(
        self,
        image: str,
        bbox: Optional[List[Dict[str, Any]]] = None,
    ):

        # --- load image ---
        img_bgr = cv2.imread(image)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not read image: {image}")
     
        # -----------------------------
        # Build prompts from box labels
        # -----------------------------
       
        labels_as_list = [
            b.get("label", "")
            for b in (bbox or [])
            if "label" in b and b.get("label")
        ]
        labels_as_list = list(dict.fromkeys(labels_as_list))  # deduplicate
        print('labels_as_list',labels_as_list)

        # If we have no labels, just use a broad generic prompt.
        if not labels_as_list:
            prompts = ["object"]
            categories = ["object"]
        else:
            prompts = labels_as_list
            categories = labels_as_list

        # -----------------------------
        # Run SAM3 for all prompts
        # -----------------------------
        seg_dict = self.segment_multi(
            img_bgr,
            prompts=prompts,
            conf_thresholds=None,  # use default per prompt
            categories=categories,
        )

        # Flatten all detections into arrays
        pred_xyxy: List[Tuple[float, float, float, float]] = []
        pred_labels: List[str] = []
        pred_scores: List[float] = []

        for prompt, masks in seg_dict.items():
            for m in masks:
                x, y, w, h = m["bbox"]
                x1, y1, x2, y2 = float(x), float(y), float(x + w), float(y + h)
                pred_xyxy.append((x1, y1, x2, y2))
                pred_labels.append(m.get("category", prompt))
                pred_scores.append(float(m["score"]))

        new_bboxes = []
        for prompt, masks in seg_dict.items():
            print(f"[DEBUG] prompt '{prompt}': {len(masks)} masks")        
            new_bboxes.append(
                {
                    "label": m.get("category", prompt),
                    # copy confidence from the best-matching input box
                    "confidence":float(m["score"]),
                    "xywh": m["bbox"]
                    ,
                }
            )

        debug_image = img_bgr.copy()
        for b in new_bboxes:
            x1, y1, w, h = b["xywh"]
            x2 = x1 + w
            y2 = y1 + h
            lab = b["label"]
            score = b["confidence"]
            cv2.rectangle(
                debug_image,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 0, 255),
                2,
            )
            cv2.putText(
                debug_image,
                f"{lab}:{score:.2f}",
                (int(x1), max(int(y1) - 5, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )
        print(f"[DEBUG] Refined {len(new_bboxes)} boxes using SAM3 prompts.")
        return new_bboxes, debug_image


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Quick demo / smoke test for Sam3Detector."
    )
    parser.add_argument("image", help="Path to input image (BGR).")
    parser.add_argument("prompt", help='Text prompt, e.g. "screw head".')
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--conf", type=float, default=0.2)
    parser.add_argument("--save-debug",
                        type=str,
                        default=None,
                        help="Optional path to save debug image.")

    args = parser.parse_args()

    if not os.path.isfile(args.image):
        raise SystemExit(f"[ERROR] Image does not exist: {args.image}")

    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        raise SystemExit(f"[ERROR] cv2 failed to read: {args.image}")

    print(f"[INFO] Running Sam3Detector with prompt: '{args.prompt}'")

    detector = Sam3Detector(
        device=args.device,
        default_conf_threshold=args.conf,
    )

    masks = detector.segment(
        color_bgr=img_bgr,
        text_prompt=args.prompt,
        confidence_threshold=None,
        category=args.prompt,
    )

    print(f"[INFO] Found {len(masks)} masks")

    if masks:
        # Create visualization
        vis = draw_mask_debug(
            image_bgr=img_bgr,
            masks=masks,
            output_path=args.save_debug if args.save_debug else "sam3_debug_tmp.png",
        )

        # Show the image in a debug window
        cv2.imshow("SAM3 Debug Visualization", vis)
        print("[INFO] Press any key to close debug window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print("[INFO] No masks found.")
