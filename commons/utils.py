from __future__ import annotations
import numpy as np
from typing import List, Tuple
import cv2
from pathlib import Path
import math
import os
# ---------------------------------------------------------------------------
# Utility: transform to UR pose
# ---------------------------------------------------------------------------

def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-8:
        raise ValueError("Cannot normalize zero-length vector")
    return v / n

def screw_direction(p_head: np.ndarray, p_stem: np.ndarray) -> np.ndarray:
    """
    Returns a unit vector along the screw axis, pointing from stem -> head.
    """
    return normalize(p_head - p_stem)

def point_above_head_along_axis(
    p_head: np.ndarray,
    d: np.ndarray,
    axis_offset: float = 0.01,
) -> np.ndarray:
    """
    Returns a point a small distance *along the screw axis* beyond the head.
    
    p_head: 3D position of the screw head (same frame as d)
    d: unit direction vector of the screw axis
    axis_offset: how far along d from the head (meters)
    """
    return p_head + axis_offset * d

def matrix_to_ur_pose(T: np.ndarray) -> List[float]:
    """Convert 4x4 transform to UR pose [x, y, z, Rx, Ry, Rz]."""
    x, y, z = T[:3, 3]
    rotation_vector, _ = cv2.Rodrigues(
        np.asarray(T[:3, :3], dtype=np.float64)
    )
    rx, ry, rz = rotation_vector.reshape(3,)
    return [
        float(x), float(y), float(z),
        float(rx), float(ry), float(rz),
    ]

def draw_point(
    image: np.ndarray,
    u: float,
    v: float,
    text: str = "target",
) -> None:
    """Draw 2D projected 3D target position for debugging."""

    vis = image.copy()
    cv2.circle(vis, (int(u), int(v)), 8, (255, 0, 0), -1)
    cv2.putText(
        vis, text, (int(u) + 10, int(v) - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2
    )
    cv2.imshow("3D_target", cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)


def _get_intrinsics_fx_fy_cx_cy(intr) -> Tuple[float, float, float, float]:
    """
    Helper to extract fx, fy, cx, cy from either a dict or an intrinsics object.
    Adjust this if your intrinsics type is different.
    """
    # Dict-style intrinsics (like in image-only mode)
    if isinstance(intr, dict):
        fx = float(intr["fx"])
        fy = float(intr["fy"])
        cx = float(intr["cx"])
        cy = float(intr["cy"])
        return fx, fy, cx, cy

    # Object-style intrinsics
    # (adjust attribute names if your camera intrinsics use different fields)
    fx = float(getattr(intr, "fx", None))
    fy = float(getattr(intr, "fy", None))
    cx = float(getattr(intr, "cx", None))
    cy = float(getattr(intr, "cy", None))
    if any(v is None for v in (fx, fy, cx, cy)):
        raise ValueError("Unsupported intrinsics format; please update _get_intrinsics_fx_fy_cx_cy().")
    return fx, fy, cx, cy

def check_path_exists(input_path,file=__file__):
    if not input_path:
        print('input path is none')
        return None

    base_dir = Path(file).resolve().parent  # folder this .py lives in

    p = Path(input_path)
    if not p.is_absolute():
        p = base_dir / p  # make it relative to this file's dir

    try:
        abs_path = p.resolve(strict=True)
        print(abs_path)
        return abs_path
    except FileNotFoundError:
        print(p, 'does not exist')
        return None

def read_image_file(image_path: str) -> np.ndarray:
    if image_path is not None:
        if not os.path.exists(image_path):
           return None

        color = cv2.imread(image_path)
        if color is None:
            raise ValueError(f"Failed to read image: {image_path}")
        
        return color
    return None


def rotz(rad: float) -> np.ndarray:
    """3×3 rotation matrix for a rotation about +Z by rad (radians)."""
    c = math.cos(rad)
    s = math.sin(rad)
    return np.array([
        [ c, -s, 0.0],
        [ s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)



def axis_angle_to_rot(r: np.ndarray) -> np.ndarray:
    """
    Convert a 3D axis-angle vector (rx, ry, rz) to a 3x3 rotation matrix.
    UR-style: axis = r / ||r||, angle = ||r||.
    """
    theta = float(np.linalg.norm(r))
    if theta < 1e-8:
        return np.eye(3)

    k = r / theta
    kx, ky, kz = k
    K = np.array([[0.0, -kz,  ky],
                  [kz,  0.0, -kx],
                  [-ky, kx,  0.0]], dtype=float)
    R = np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)
    return R


def rot_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """
    Convert a 3x3 rotation matrix to a 3D axis-angle vector (UR-style).
    """
    # Clamp for numerical safety
    trace = float(np.trace(R))
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = max(min(cos_theta, 1.0), -1.0)

    theta = math.acos(cos_theta)

    if theta < 1e-8:
        return np.zeros(3, dtype=float)

    rx = R[2, 1] - R[1, 2]
    ry = R[0, 2] - R[2, 0]
    rz = R[1, 0] - R[0, 1]
    axis = np.array([rx, ry, rz], dtype=float)
    axis /= (2.0 * math.sin(theta))

    return axis * theta


def fold_angle_for_symmetry(angle: float, symmetry_order: int = 2) -> float:
    """
    Fold an angle (radians) into the smallest-magnitude equivalent given
    N-fold rotational symmetry (symmetry_order).
    
    For symmetry_order = 2  => 180° periodicity.
    For symmetry_order = 4  => 90° periodicity, etc.
    """
    if symmetry_order <= 1:
        # No symmetry => nothing to fold
        return angle

    period = 2.0 * math.pi / symmetry_order
    # Bring into [-pi, pi]
    angle = (angle + math.pi) % (2.0 * math.pi) - math.pi

    # Shift by multiples of 'period' to get minimal magnitude
    k = round(angle / period)
    folded = angle - k * period
    # final fold again into [-pi, pi] for safety
    folded = (folded + math.pi) % (2.0 * math.pi) - math.pi
    return folded

def rotation_distance(R1: np.ndarray, R2: np.ndarray) -> float:
    """
    Smallest angle (in radians) between two rotation matrices.
    """
    R_err = R1.T @ R2
    # Clamp trace to valid range for numerical stability
    cos_theta = (np.trace(R_err) - 1.0) / 2.0
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return math.acos(cos_theta)
