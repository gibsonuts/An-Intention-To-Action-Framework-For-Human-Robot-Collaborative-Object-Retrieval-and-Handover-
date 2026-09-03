"""Run SAM3 inference in a separate Conda environment, one request at a time."""

import json
import os
import pickle
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional

import numpy as np

from detectors.sam3_object_detection import mask_centroid, project_pixel_to_cam


class Sam3SubprocessDetector:
    def __init__(
        self,
        env_name: str = "quendabot_demo",
        python_path: Optional[str] = None,
        device: str = "cuda",
        timeout: float = 300.0,
    ):
        self.python_path = Path(
            python_path
            or os.environ.get("SAM3_PYTHON", "")
            or Path(sys.executable).resolve().parents[2] / env_name / "bin" / "python"
        )
        if not self.python_path.is_file():
            raise FileNotFoundError(
                f"SAM3 Python not found at {self.python_path}. "
                "Set SAM3_PYTHON to the quendabot_demo Python executable."
            )

        self.device = device
        self.timeout = timeout
        self.worker_path = Path(__file__).resolve().with_name("sam3_worker.py")

    def _run(self, operation: str, color: np.ndarray, params: Dict[str, Any]):
        with tempfile.TemporaryDirectory(prefix="qbot_sam3_") as temp_dir:
            temp_path = Path(temp_dir)
            image_path = temp_path / "color.npy"
            params_path = temp_path / "params.json"
            output_path = temp_path / "result.pkl"

            np.save(image_path, color, allow_pickle=False)
            params_path.write_text(json.dumps(params), encoding="utf-8")

            command = [
                str(self.python_path),
                str(self.worker_path),
                "--operation",
                operation,
                "--image",
                str(image_path),
                "--params",
                str(params_path),
                "--output",
                str(output_path),
                "--device",
                self.device,
            ]
            print(f"[SAM3] Running one-shot inference in {self.python_path.parent.parent.name}...")
            completed = subprocess.run(
                command,
                cwd=str(self.worker_path.parent.parent),
                text=True,
                capture_output=True,
                timeout=self.timeout,
                check=False,
            )
            if completed.stdout:
                print(completed.stdout, end="")
            if completed.returncode != 0:
                detail = completed.stderr.strip() or "SAM3 worker exited without an error message"
                raise RuntimeError(f"SAM3 subprocess failed ({completed.returncode}): {detail}")
            if not output_path.is_file():
                raise RuntimeError("SAM3 subprocess did not create a result file")

            with output_path.open("rb") as result_file:
                return pickle.load(result_file)

    def segment(
        self,
        color_bgr: np.ndarray,
        text_prompt: str,
        confidence_threshold: Optional[float] = None,
        category: Optional[str] = None,
        orientation_align: str = "long",
    ) -> List[Dict[str, Any]]:
        return self._run(
            "segment",
            color_bgr,
            {
                "text_prompt": text_prompt,
                "confidence_threshold": confidence_threshold,
                "category": category,
                "orientation_align": orientation_align,
            },
        )

    def detect_bbox(
        self,
        prompts: List[str],
        color: np.ndarray,
        conf_list: List[float],
        debug: bool = False,
    ) -> List[Dict[str, Any]]:
        return self._run(
            "detect_bbox",
            color,
            {"prompts": prompts, "conf_list": conf_list, "debug": debug},
        )

    def compute_target_from_mask(
        self,
        mask: Dict[str, Any],
        depth: np.ndarray,
        intr: Dict[str, float],
        depth_scale: float = 1.0,
    ) -> Optional[Dict[str, Any]]:
        seg = np.squeeze(mask["segmentation"]).astype(bool)
        if seg.ndim != 2:
            raise ValueError(f"compute_target_from_mask expected 2D mask, got {seg.shape}")

        preferred = mask.get("target_pixel")
        if preferred is not None and len(preferred) >= 2:
            u, v = float(preferred[0]), float(preferred[1])
            target_src = mask.get("target_pixel_source", "target_pixel")
        else:
            u, v = mask_centroid(seg)
            target_src = "centroid"

        height, width = depth.shape[:2]
        u_int = int(np.clip(round(u), 0, max(width - 1, 0)))
        v_int = int(np.clip(round(v), 0, max(height - 1, 0)))
        depth_raw = float(depth[v_int, u_int])
        depth_m = depth_raw * depth_scale

        if depth_m <= 0:
            ys, xs = np.where(seg)
            if xs.size:
                xs = np.clip(xs.astype(int), 0, max(width - 1, 0))
                ys = np.clip(ys.astype(int), 0, max(height - 1, 0))
                values = depth[ys, xs].astype(np.float32) * float(depth_scale)
                valid = values > 0
                if np.any(valid):
                    valid_x = xs[valid].astype(np.float32)
                    valid_y = ys[valid].astype(np.float32)
                    nearest = int(np.argmin((valid_x - u) ** 2 + (valid_y - v) ** 2))
                    u, v = float(valid_x[nearest]), float(valid_y[nearest])
                    depth_m = float(depth[int(v), int(u)]) * depth_scale
                    target_src = f"{target_src}_nearest_valid_depth"

        if depth_m <= 0:
            print(f"[WARN] Invalid depth at SAM3 target pixel ({u:.1f}, {v:.1f})")
            return None

        return {
            "target_cam": project_pixel_to_cam(u, v, depth_m, intr),
            "pixel": (u, v),
            "depth_m": depth_m,
            "angle_cam": mask.get("angle_deg"),
            "target_pixel_source": target_src,
        }
