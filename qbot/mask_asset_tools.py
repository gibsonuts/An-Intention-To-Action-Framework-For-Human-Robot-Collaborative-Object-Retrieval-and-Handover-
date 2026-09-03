#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import json

import cv2
import numpy as np


ASSETS_DIR = Path(__file__).resolve().parent / "assets"


@dataclass
class MaskAssetPaths:
    image: Path
    mask: Path
    masked: Path
    meta: Path
    depth_raw: Path
    masked_depth_raw: Path


def ensure_assets_dir(assets_dir: Optional[Path | str] = None) -> Path:
    out_dir = Path(assets_dir) if assets_dir is not None else ASSETS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _sanitize_asset_name(asset_name: str) -> str:
    safe = "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in asset_name.strip())
    if not safe:
        raise ValueError("asset_name is empty after sanitization")
    return safe


def get_mask_asset_paths(asset_name: str, assets_dir: Optional[Path | str] = None) -> MaskAssetPaths:
    safe_name = _sanitize_asset_name(asset_name)
    base_dir = ensure_assets_dir(assets_dir)
    return MaskAssetPaths(
        image=base_dir / f"{safe_name}_image.png",
        mask=base_dir / f"{safe_name}_mask.png",
        masked=base_dir / f"{safe_name}_masked.png",
        meta=base_dir / f"{safe_name}_meta.json",
        depth_raw=base_dir / f"{safe_name}_depth.npy",
        masked_depth_raw=base_dir / f"{safe_name}_masked_depth.npy",
    )


def _validate_color_image(image_rgb: np.ndarray) -> np.ndarray:
    if image_rgb is None:
        raise ValueError("image_rgb is None")
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB image, got shape={getattr(image_rgb, 'shape', None)}")
    if image_rgb.dtype != np.uint8:
        image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image_rgb)


def _validate_depth_image(depth_u16: np.ndarray) -> np.ndarray:
    if depth_u16 is None:
        raise ValueError("depth_u16 is None")
    if depth_u16.ndim != 2:
        raise ValueError(f"Expected HxW depth image, got shape={getattr(depth_u16, 'shape', None)}")
    if depth_u16.dtype != np.uint16:
        depth_u16 = np.clip(depth_u16, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    return np.ascontiguousarray(depth_u16)


def normalize_mask(mask: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    if mask is None:
        raise ValueError("mask is None")
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape={mask.shape}")
    if mask.shape != shape_hw:
        mask = cv2.resize(mask, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    if mask.dtype != np.uint8:
        mask = np.clip(mask, 0, 255).astype(np.uint8)
    return np.where(mask > 127, 255, 0).astype(np.uint8)


def apply_binary_mask(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    image_rgb = _validate_color_image(image_rgb)
    mask_u8 = normalize_mask(mask, image_rgb.shape[:2])
    masked = cv2.bitwise_and(image_rgb, image_rgb, mask=mask_u8)
    return masked


def apply_binary_mask_depth(depth_u16: np.ndarray, mask: np.ndarray) -> np.ndarray:
    depth_u16 = _validate_depth_image(depth_u16)
    mask_u8 = normalize_mask(mask, depth_u16.shape[:2])
    masked = np.where(mask_u8 > 0, depth_u16, 0).astype(np.uint16)
    return masked


def depth_to_vis_bgr(depth_u16: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Visualize uint16 depth (millimeters) as a BGR heatmap."""
    depth_u16 = _validate_depth_image(depth_u16)
    valid = depth_u16 > 0
    if mask is not None:
        mask_u8 = normalize_mask(mask, depth_u16.shape[:2])
        valid &= mask_u8 > 0

    vis_u8 = np.zeros(depth_u16.shape, dtype=np.uint8)
    if np.any(valid):
        vals = depth_u16[valid].astype(np.float32)
        vmin = float(np.percentile(vals, 5))
        vmax = float(np.percentile(vals, 95))
        if vmax <= vmin + 1.0:
            vmax = vmin + 1.0
        scaled = (depth_u16.astype(np.float32) - vmin) / (vmax - vmin)
        vis_u8 = np.clip(scaled * 255.0, 0, 255).astype(np.uint8)
    vis_bgr = cv2.applyColorMap(vis_u8, cv2.COLORMAP_TURBO)
    vis_bgr[~valid] = (0, 0, 0)
    return vis_bgr


def save_mask_asset(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    asset_name: str,
    depth_u16: Optional[np.ndarray] = None,
    assets_dir: Optional[Path | str] = None,
    camera: Optional[str] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    image_rgb = _validate_color_image(image_rgb)
    mask_u8 = normalize_mask(mask, image_rgb.shape[:2])
    paths = get_mask_asset_paths(asset_name, assets_dir=assets_dir)
    meta: Dict[str, Any] = {}

    if depth_u16 is not None:
        depth_u16 = _validate_depth_image(depth_u16)
        if depth_u16.shape != image_rgb.shape[:2]:
            raise ValueError(
                f"Depth shape {depth_u16.shape} does not match display image shape {image_rgb.shape[:2]}"
            )
        masked_depth_u16 = apply_binary_mask_depth(depth_u16, mask_u8)
        np.save(paths.depth_raw, depth_u16)
        np.save(paths.masked_depth_raw, masked_depth_u16)
        depth_vis_bgr = depth_to_vis_bgr(depth_u16)
        masked_depth_vis_bgr = depth_to_vis_bgr(masked_depth_u16, mask=mask_u8)
        ok_image = cv2.imwrite(str(paths.image), depth_vis_bgr)
        ok_masked = cv2.imwrite(str(paths.masked), masked_depth_vis_bgr)
        meta["reference_mode"] = "depth"
        meta["depth_units"] = "mm_uint16"
        meta["depth_paths"] = {
            "depth_raw": str(paths.depth_raw),
            "masked_depth_raw": str(paths.masked_depth_raw),
        }
    else:
        masked_rgb = apply_binary_mask(image_rgb, mask_u8)
        ok_image = cv2.imwrite(str(paths.image), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        ok_masked = cv2.imwrite(str(paths.masked), cv2.cvtColor(masked_rgb, cv2.COLOR_RGB2BGR))
        meta["reference_mode"] = "rgb"

    ok_mask = cv2.imwrite(str(paths.mask), mask_u8)
    if not (ok_image and ok_mask and ok_masked):
        raise RuntimeError("Failed to save one or more asset images")

    mask_pixels = int(np.count_nonzero(mask_u8))
    total_pixels = int(mask_u8.shape[0] * mask_u8.shape[1])
    meta.update({
        "asset_name": _sanitize_asset_name(asset_name),
        "camera": camera,
        "image_shape_hw3": list(image_rgb.shape),
        "mask_pixels": mask_pixels,
        "mask_coverage": (mask_pixels / total_pixels) if total_pixels else 0.0,
        "similarity_depth_scale_mm": 100.0,
    })
    if extra_meta:
        meta.update(extra_meta)
    paths.meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {
        "image_path": str(paths.image),
        "mask_path": str(paths.mask),
        "masked_path": str(paths.masked),
        "meta_path": str(paths.meta),
        "depth_path": str(paths.depth_raw),
        "masked_depth_path": str(paths.masked_depth_raw),
    }


def load_mask_asset(asset_name: str, assets_dir: Optional[Path | str] = None) -> Dict[str, Any]:
    paths = get_mask_asset_paths(asset_name, assets_dir=assets_dir)
    if not paths.image.exists():
        raise FileNotFoundError(f"Missing asset image: {paths.image}")
    if not paths.mask.exists():
        raise FileNotFoundError(f"Missing asset mask: {paths.mask}")
    if not paths.masked.exists():
        raise FileNotFoundError(f"Missing asset masked image: {paths.masked}")

    image_bgr = cv2.imread(str(paths.image), cv2.IMREAD_COLOR)
    masked_bgr = cv2.imread(str(paths.masked), cv2.IMREAD_COLOR)
    mask_u8 = cv2.imread(str(paths.mask), cv2.IMREAD_GRAYSCALE)
    if image_bgr is None or masked_bgr is None or mask_u8 is None:
        raise RuntimeError("Failed to load one or more asset files")

    meta: Dict[str, Any] = {}
    if paths.meta.exists():
        try:
            meta = json.loads(paths.meta.read_text(encoding="utf-8"))
        except Exception:
            meta = {}

    out = {
        "paths": {
            "image": str(paths.image),
            "mask": str(paths.mask),
            "masked": str(paths.masked),
            "meta": str(paths.meta),
            "depth_raw": str(paths.depth_raw),
            "masked_depth_raw": str(paths.masked_depth_raw),
        },
        "image_rgb": cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
        "masked_rgb": cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2RGB),
        "mask": normalize_mask(mask_u8, image_bgr.shape[:2]),
        "meta": meta,
    }
    if paths.depth_raw.exists():
        out["depth_u16"] = _validate_depth_image(np.load(paths.depth_raw))
    if paths.masked_depth_raw.exists():
        out["masked_depth_u16"] = _validate_depth_image(np.load(paths.masked_depth_raw))
    return out


def compare_image_to_saved_mask_asset(
    image_rgb: np.ndarray,
    asset_name: str,
    assets_dir: Optional[Path | str] = None,
    resize_input: bool = True,
) -> Dict[str, Any]:
    image_rgb = _validate_color_image(image_rgb)
    asset = load_mask_asset(asset_name, assets_dir=assets_dir)
    ref_mask = asset["mask"]
    ref_masked_rgb = asset["masked_rgb"]
    ref_shape = ref_masked_rgb.shape[:2]

    input_image_rgb = image_rgb
    resized = False
    if input_image_rgb.shape[:2] != ref_shape:
        if not resize_input:
            raise ValueError(
                f"Input image shape {input_image_rgb.shape[:2]} does not match reference shape {ref_shape}"
            )
        input_image_rgb = cv2.resize(
            input_image_rgb, (ref_shape[1], ref_shape[0]), interpolation=cv2.INTER_LINEAR
        )
        resized = True

    input_masked_rgb = apply_binary_mask(input_image_rgb, ref_mask)
    ref_masked_rgb = _validate_color_image(ref_masked_rgb)
    ref_mask = normalize_mask(ref_mask, ref_shape)
    valid = ref_mask > 0
    if not np.any(valid):
        raise ValueError("Saved mask is empty (no painted pixels)")

    curr_vals = input_masked_rgb[valid].astype(np.float32)
    ref_vals = ref_masked_rgb[valid].astype(np.float32)
    diff = curr_vals - ref_vals

    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    similarity = max(0.0, 1.0 - (mae / 255.0))

    return {
        "similarity_score": similarity,
        "similarity_percent": similarity * 100.0,
        "mae": mae,
        "rmse": rmse,
        "masked_pixel_count": int(np.count_nonzero(valid)),
        "input_resized": resized,
        "reference_masked_rgb": ref_masked_rgb,
        "input_masked_rgb": input_masked_rgb,
        "mask": ref_mask,
        "asset_meta": asset.get("meta", {}),
        "asset_paths": asset["paths"],
    }


def compare_depth_to_saved_mask_asset(
    depth_u16: np.ndarray,
    asset_name: str,
    assets_dir: Optional[Path | str] = None,
    resize_input: bool = True,
) -> Dict[str, Any]:
    depth_u16 = _validate_depth_image(depth_u16)
    asset = load_mask_asset(asset_name, assets_dir=assets_dir)
    if "masked_depth_u16" not in asset:
        raise ValueError(
            "Saved asset does not contain depth reference data. Recreate it with utilties/create_mask_asset.py."
        )

    ref_mask = asset["mask"]
    ref_masked_depth = _validate_depth_image(asset["masked_depth_u16"])
    ref_depth = _validate_depth_image(asset.get("depth_u16", ref_masked_depth))
    ref_h, ref_w = ref_mask.shape[:2]

    input_depth = depth_u16
    resized = False
    if input_depth.shape != (ref_h, ref_w):
        if not resize_input:
            raise ValueError(f"Input depth shape {input_depth.shape} does not match reference {(ref_h, ref_w)}")
        input_depth = cv2.resize(input_depth, (ref_w, ref_h), interpolation=cv2.INTER_NEAREST)
        input_depth = _validate_depth_image(input_depth)
        resized = True

    input_masked_depth = apply_binary_mask_depth(input_depth, ref_mask)

    valid = (ref_mask > 0) & (ref_masked_depth > 0) & (input_masked_depth > 0)
    if not np.any(valid):
        raise ValueError("No valid (non-zero) depth pixels inside saved mask for comparison")

    curr_vals = input_masked_depth[valid].astype(np.float32)
    ref_vals = ref_masked_depth[valid].astype(np.float32)
    diff = curr_vals - ref_vals

    mae_mm = float(np.mean(np.abs(diff)))
    rmse_mm = float(np.sqrt(np.mean(diff * diff)))
    scale_mm = float((asset.get("meta") or {}).get("similarity_depth_scale_mm", 100.0))
    if scale_mm <= 0:
        scale_mm = 100.0
    similarity = max(0.0, 1.0 - (mae_mm / scale_mm))

    diff_mm = np.zeros(ref_mask.shape, dtype=np.float32)
    diff_mm[valid] = np.abs(input_masked_depth[valid].astype(np.float32) - ref_masked_depth[valid].astype(np.float32))
    diff_vis_u8 = np.clip((diff_mm / scale_mm) * 255.0, 0, 255).astype(np.uint8)
    diff_vis_bgr = cv2.applyColorMap(diff_vis_u8, cv2.COLORMAP_TURBO)
    diff_vis_bgr[~valid] = (0, 0, 0)

    ref_masked_vis_bgr = depth_to_vis_bgr(ref_masked_depth, mask=ref_mask)
    input_masked_vis_bgr = depth_to_vis_bgr(input_masked_depth, mask=ref_mask)

    return {
        "similarity_score": similarity,
        "similarity_percent": similarity * 100.0,
        "mae": mae_mm,
        "rmse": rmse_mm,
        "mae_mm": mae_mm,
        "rmse_mm": rmse_mm,
        "masked_pixel_count": int(np.count_nonzero(valid)),
        "input_resized": resized,
        "reference_depth_u16": ref_depth,
        "reference_masked_depth_u16": ref_masked_depth,
        "input_masked_depth_u16": input_masked_depth,
        "reference_masked_vis_bgr": ref_masked_vis_bgr,
        "input_masked_vis_bgr": input_masked_vis_bgr,
        "diff_vis_bgr": diff_vis_bgr,
        "mask": ref_mask,
        "asset_meta": asset.get("meta", {}),
        "asset_paths": asset["paths"],
    }
