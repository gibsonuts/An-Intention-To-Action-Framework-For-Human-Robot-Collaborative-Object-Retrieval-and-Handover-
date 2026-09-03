
#!/usr/bin/env python3
"""
grasp_utils.py
Common math, pose, projection, and visualization helpers for the grasp pipeline.
"""
# commons/utils.py
from __future__ import annotations
from pathlib import Path
import math
import numpy as np
from typing import Dict, Any
import math
import numpy as np
from typing import List, Dict, Optional, Tuple
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial.transform import Rotation as spipyR

def _validate_pts_cols(pts: np.ndarray, cols: Optional[np.ndarray]):
    print(cols)
    if pts is None or pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"`pts` must be an (N,3) array, got shape {None if pts is None else pts.shape}")
    if cols is not None:
        if cols.ndim != 2 or cols.shape[1] != 3:
            raise ValueError(f"`cols` must be an (N,3) array when provided, got shape {cols.shape}")
        if cols.shape[0] != pts.shape[0]:
            raise ValueError(f"`cols` and `pts` must have same length (N). pts:{pts.shape[0]} cols:{cols.shape[0]}")


def T_to_ur_pose(T):
    x, y, z = T[:3, 3]
    R = T[:3, :3]

    theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if theta < 1e-6:
        return [x, y, z, 0, 0, 0]

    rx = (R[2, 1] - R[1, 2]) / (2 * np.sin(theta))
    ry = (R[0, 2] - R[2, 0]) / (2 * np.sin(theta))
    rz = (R[1, 0] - R[0, 1]) / (2 * np.sin(theta))

    return [float(x), float(y), float(z),
            float(rx * theta), float(ry * theta), float(rz * theta)]

def statistical_outlier_removal_kdtree(
    pts: np.ndarray,
    k: int = 16,
    std_ratio: float = 2.0,
    cols: Optional[np.ndarray] = None,
    return_mask: bool = False,
    leafsize: int = 32,
    workers: int = -1,
):
    _validate_pts_cols(pts, cols)

    N = pts.shape[0]
    if N < 2:
        keep = np.ones(N, dtype=bool)
        if return_mask:
            return pts.astype(np.float32), (cols.astype(np.float32) if cols is not None else None), keep
        return pts.astype(np.float32), (cols.astype(np.float32) if cols is not None else None)

    k_eff = max(1, min(k, N - 1))
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(pts, leafsize=leafsize)
        dists, _ = tree.query(pts, k=k_eff + 1, workers=workers)
        if dists.ndim == 1:
            dists = dists[:, None]
        dists = dists[:, 1:]
    except Exception:
        try:
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=k_eff + 1, algorithm="kd_tree",
                                  n_jobs=workers if workers != -1 else None)
            nn.fit(pts)
            dists, _ = nn.kneighbors(pts, return_distance=True)
            dists = dists[:, 1:]
        except Exception as e:
            raise ImportError(
                "Install SciPy (preferred) or scikit-learn to use KD-tree SOR."
            ) from e

    mean_knn = dists.mean(axis=1).astype(np.float32)
    thresh = float(mean_knn.mean() + std_ratio * mean_knn.std(ddof=0))
    keep = mean_knn <= thresh

    pts_f = pts[keep].astype(np.float32)
    cols_f = (cols[keep].astype(np.float32) if cols is not None else None)

    if return_mask:
        return pts_f, cols_f, keep
    return pts_f, cols_f

def voxel_downsample(
    pts: np.ndarray,
    cols: Optional[np.ndarray] = None,
    voxel_size: float = 0.005
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Mean voxel-grid downsampling using sort/group-by with reduceat (fast, no deps).
    pts:  (N,3) float array
    cols: (N,3) float array or None
    returns: (pts_ds, cols_ds or None) as float32
    """
    if pts is None or pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"`pts` must be (N,3), got {None if pts is None else pts.shape}")
    if cols is not None and (cols.ndim != 2 or cols.shape[1] != 3 or cols.shape[0] != pts.shape[0]):
        raise ValueError("`cols` must be (N,3) and match pts length when provided")

    N = pts.shape[0]
    if N == 0:
        return pts, (cols if cols is not None else None)

    # 1) Quantize to voxel grid (int64), robust to negatives
    grid = np.floor(pts / voxel_size).astype(np.int64, copy=False)

    # 2) Sort by (x,y,z) so equal voxels are consecutive
    order = np.lexsort((grid[:, 2], grid[:, 1], grid[:, 0]))
    grid_s = grid[order]
    pts_s  = pts[order]
    cols_s = cols[order] if cols is not None else None

    # 3) Find group boundaries where voxel index changes
    change = np.any(np.diff(grid_s, axis=0) != 0, axis=1)
    # starts of each group
    starts = np.concatenate(([0], np.flatnonzero(change) + 1))
    # add sentinel end to make reduceat-friendly
    ends = np.concatenate((starts[1:], [N]))
    counts = (ends - starts).astype(np.int64)

    # 4) Per-voxel sums then means
    sum_pts = np.add.reduceat(pts_s, starts, axis=0)
    pts_ds = (sum_pts / counts[:, None]).astype(np.float32, copy=False)

    if cols_s is not None:
        sum_cols = np.add.reduceat(cols_s, starts, axis=0)
        cols_ds = (sum_cols / counts[:, None]).astype(np.float32, copy=False)
    else:
        cols_ds = None

    return pts_ds, cols_ds

# def voxel_downsample(pts: np.ndarray, cols: np.ndarray, voxel_size: float = 0.005) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Simple mean voxel-grid downsampling (no external deps).
#     """
#     if len(pts) == 0:
#         return pts, cols
#     grid = np.floor(pts / voxel_size).astype(np.int32)
#     uniq, inv = np.unique(grid, axis=0, return_inverse=True)
#     counts = np.bincount(inv)

#     pts_ds = np.stack([np.bincount(inv, weights=pts[:, d]) for d in range(3)], axis=1) / counts[:, None]
#     cols_ds = np.stack([np.bincount(inv, weights=cols[:, d]) for d in range(3)], axis=1) / counts[:, None]

#     return pts_ds.astype(np.float32), cols_ds.astype(np.float32)


# -------------------- Visualization --------------------

def show_image_with_boxes(image_path: str, boxes: List[Dict], title: Optional[str] = None, save_path: Optional[str] = None) -> None:
    """
    Display an image with bounding boxes overlaid.
    Optionally saves to `save_path` (PNG) instead of or in addition to showing.
    Each box is a dict with at least key "xywh". Optional keys: "label", "confidence".
    """
    img = Image.open(image_path).convert("RGB")
    fig, ax = plt.subplots()
    ax.imshow(img)

    for b in boxes or []:
        x, y, w, h = b["xywh"]
        label = b.get("label", "")
        conf  = b.get("confidence", None)

        rect = patches.Rectangle((x, y), w, h, fill=False, linewidth=2)
        ax.add_patch(rect)

        if label or conf is not None:
            txt = f"{label}".strip()
            if conf is not None:
                txt = f"{txt} ({conf:.2f})" if txt else f"{conf:.2f}"
            ax.text(x, max(0, y - 5), txt,
                    fontsize=9, color="white",
                    bbox=dict(facecolor="black", alpha=0.5, pad=2))

    if title:
        ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0, dpi=200)
        print(f"[\u2713] Saved visualization -> {save_path}")
    else:
        plt.show()


# -------------------- Math / Pose helpers --------------------

def so3_distance(Ra: np.ndarray, Rb: np.ndarray) -> float:
    M = Ra.T @ Rb
    c = (np.trace(M) - 1.0) * 0.5
    c = 1.0 if c > 1.0 else (-1.0 if c < -1.0 else c)
    return math.acos(c)


def rot_axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis) + 1e-12
    x, y, z = axis / n
    c, s = math.cos(angle), math.sin(angle)
    C = 1.0 - c
    return np.array([
        [c + x*x*C,     x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s,   c + y*y*C,   y*z*C - x*s],
        [z*x*C - y*s,   z*y*C + x*s, c + z*z*C  ]
    ], dtype=float)


def R_from_rpy(rx: float, ry: float, rz: float) -> np.ndarray:
    cz, sz = math.cos(rz), math.sin(rz)
    cy, sy = math.cos(ry), math.sin(ry)
    cx, sx = math.cos(rx), math.sin(rx)
    Rz = np.array([[cz, -sz, 0],[sz, cz, 0],[0, 0, 1]], dtype=float)
    Ry = np.array([[cy, 0, sy],[0, 1, 0],[-sy, 0, cy]], dtype=float)
    Rx = np.array([[1, 0, 0],[0, cx, -sx],[0, sx, cx]], dtype=float)
    return Rz @ Ry @ Rx  # ZYX


def T_from_rpy(xyz, rpy, degrees: bool=False) -> np.ndarray:
    x, y, z = [float(v) for v in xyz]
    rx, ry, rz = [float(v) for v in rpy]   # ZYX order by your convention
    if degrees:
        rz, ry, rx = map(math.radians, (rz, ry, rx))
    R = R_from_rpy(rx, ry, rz)
    T = np.eye(4); T[:3,:3] = R; T[:3,3] = [x, y, z]
    return T


def rotvec_from_R(R: np.ndarray) -> np.ndarray:
    t = (np.trace(R) - 1.0) / 2.0
    t = 1.0 if t > 1.0 else (-1.0 if t < -1.0 else t)
    th = math.acos(t)
    if abs(th) < 1e-12:
        return np.zeros(3, dtype=float)
    rx = (R[2,1]-R[1,2])/(2.0*math.sin(th))
    ry = (R[0,2]-R[2,0])/(2.0*math.sin(th))
    rz = (R[1,0]-R[0,1])/(2.0*math.sin(th))
    return np.array([rx, ry, rz], dtype=float) * th


def pose_from_T(T: np.ndarray) -> List[float]:
    R = T[:3,:3]
    p = T[:3,3]
    rvec = rotvec_from_R(R)
    return [float(p[0]), float(p[1]), float(p[2]), float(rvec[0]), float(rvec[1]), float(rvec[2])]


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3,:3] = R
    T[:3, 3] = t
    return T


def load_yaml_pose(d: Dict, default_degrees: bool=False) -> np.ndarray:
    if d is None:
        raise ValueError("Pose dict is None")
    xyz = d.get('translation')
    rpy = d.get('rpy')  # ZYX order (named rpy_xyz in YAML)
    if xyz is None or rpy is None:
        raise ValueError(f"Pose dict missing 'translation' or 'rpy': {d}")
    use_deg = bool(d.get('degrees', default_degrees))
    return T_from_rpy(xyz, rpy, degrees=use_deg)


def axis_angle_from_rot(R: np.ndarray) -> np.ndarray:
    theta = math.acos(max(min((np.trace(R) - 1) / 2.0, 1.0), -1.0))
    if abs(theta) < 1e-9:
        return np.zeros(3)
    rx = (R[2,1] - R[1,2]) / (2*math.sin(theta))
    ry = (R[0,2] - R[2,0]) / (2*math.sin(theta))
    rz = (R[1,0] - R[0,1]) / (2*math.sin(theta))
    axis = np.array([rx, ry, rz])
    return axis * theta


def rpy_from_R_zyx(R: np.ndarray) -> Tuple[float, float, float]:
    r20 = -R[2, 0]
    r20 = -1.0 if r20 < -1.0 else (1.0 if r20 > 1.0 else r20)
    pitch = math.asin(r20)
    cp = math.cos(pitch)
    eps = 1e-8
    if abs(cp) > eps:
        roll  = math.atan2(R[2, 1], R[2, 2])  # X
        yaw   = math.atan2(R[1, 0], R[0, 0])  # Z
    else:
        roll  = 0.0
        yaw   = math.atan2(-R[0, 1], R[1, 1])
    return roll, pitch, yaw


def grasp_T_from_grasp_obj(g) -> np.ndarray:
    """Build 4x4 grasp pose (camera frame) from Grasp-like object."""

    R = None; t = None
    for key in ("rotation", "R", "rot", "rotation_matrix"):
        if hasattr(g, key):
            R = np.array(getattr(g, key), dtype=float)
            break
    if R is None and hasattr(g, "rotationMatrix"):
        R = np.array(g.rotationMatrix(), dtype=float)
    for key in ("translation", "t", "centre", "translation_vector"):
        if hasattr(g, key):
            t = np.array(getattr(g, key), dtype=float).reshape(3)
            break
    if t is None and hasattr(g, "translationVector"):
        t = np.array(g.translationVector(), dtype=float).reshape(3)
    if R is None or t is None:
        raise ValueError("Could not find rotation/translation on the Grasp object")
    T = np.eye(4, dtype=float)
    T[:3,:3] = R
    T[:3, 3] = t
    return T

def set_T_on_grasp_group(grasp_group, idx, T) -> None:
    """
    Set a new 4x4 pose T for grasp_group[idx].

    Assumes a GraspGroup-like API with:
        grasp_group.rotation_matrices  -> (N, 3, 3) array
        grasp_group.translations       -> (N, 3) array
    """
    T = np.asarray(T, dtype=float)

    if T.shape == (4, 4):
        R = T[:3, :3]
        t = T[:3, 3]
    elif T.shape == (3, 4):
        R = T[:, :3]
        t = T[:, 3]
    else:
        raise ValueError(f"Expected T to have shape (4, 4) or (3, 4), got {T.shape}")

    t = np.asarray(t, dtype=float).reshape(3)

    # These should be numpy arrays of shape (N, 3, 3) and (N, 3)
    if hasattr(grasp_group, "rotation_matrices"):
        grasp_group.rotation_matrices[idx] = R
    elif hasattr(grasp_group, "R"):
        grasp_group.R[idx] = R
    else:
        raise ValueError("GraspGroup has no rotation_matrices or R attribute")

    if hasattr(grasp_group, "translations"):
        grasp_group.translations[idx] = t
    elif hasattr(grasp_group, "t"):
        grasp_group.t[idx] = t
    else:
        raise ValueError("GraspGroup has no translations or t attribute")
    
# -------------------- Camera <-> Pixel helpers --------------------

def project_cam_to_pixel(pt_cam, fx: float, fy: float, cx: float, cy: float):
    X, Y, Z = float(pt_cam[0]), float(pt_cam[1]), float(pt_cam[2])
    if Z <= 1e-9:
        return None
    u = fx * (X / Z) + cx
    v = fy * (Y / Z) + cy
    return (u, v)


def depth_to_points(depths: np.ndarray, fx: float, fy: float, cx: float, cy: float, depth_scale: float) -> np.ndarray:
    """
    Convert a depth image (uint16 or float) into Nx3 3D points (meters) in the camera frame.
    """
    H, W = depths.shape
    xmap, ymap = np.meshgrid(np.arange(W), np.arange(H))
    points_z = depths.astype(np.float32) * depth_scale
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z
    pts = np.stack([points_x, points_y, points_z], axis=-1).reshape(-1, 3)
    return pts


def point_in_any_box(u: float, v: float, boxes: List[Dict], margin: int= 0) -> bool:
    """Check if pixel (u,v) lies in any [x,y,w,h] box (optionally grown by 'margin' px)."""
    for b in boxes or []:
        x, y, w, h = b["xywh"]
        x0 = x - margin; y0 = y - margin
        x1 = x0 + w + margin*2; y1 = y0 + h + margin*2
        if (u >= x0) and (u <= x1) and (v >= y0) and (v <= y1):
            return True
    return False


def rotate_grasp_objects(gg,tf_rotation):
    for i,g in enumerate(gg):
        T = grasp_T_from_grasp_obj(g)
        T=tf_rotation@T
        T_new = T.copy()
        set_T_on_grasp_group(gg,i,T_new)

    return gg


def filter_tilted_grasps(gg, max_angle_deg):
    filtered = []
    for indx,g in enumerate(gg):
        T = grasp_T_from_grasp_obj(g)  # 4x4 transform, R in T[:3, :3]
        # R_y90 = np.array([
        #     [0, 0, 1, 0],
        #     [0, 1, 0, 0],
        #     [-1, 0, 0, 0],
        #     [0, 0, 0, 1]
        # ], dtype=float)
        # R_z90 = np.array([
        #     [0, -1, 0, 0],
        #     [1,  0, 0, 0],
        #     [0,  0, 1, 0],
        #     [0,  0, 0, 1]
        # ], dtype=float)
        # rotatedT = T @ R_y90 @ R_z90

        R = spipyR.from_matrix(T[:3,:3])
        yaw, pitch, roll = R.as_euler('zyx', degrees=True)
        rotate_angle =  (abs(pitch) +  abs(roll))/2
        print(f"Rotation applied (degrees):Yaw (Z):   {yaw:.3f} Pitch (Y): {pitch:.3f} Roll (X):  {roll:.3f}, rotate_angle {rotate_angle:.3f}")
        if rotate_angle<= max_angle_deg:
            filtered.append(indx)
    new_gg = gg[filtered]

    return new_gg


def indices_of_grasps_in_boxes(gg, fx: float, fy: float, cx: float, cy: float, boxes: List[Dict], margin: int= 0):
    keep = []
    for i, g in enumerate(gg):
        T = grasp_T_from_grasp_obj(g)
 
        t = T[:3, 3]  # camera-frame translation (meters)
            
        pv = project_cam_to_pixel(t, fx, fy, cx, cy)
        if pv is None:
            continue
        u, v = pv
        if point_in_any_box(u, v, boxes, margin=margin):
            keep.append(i)
    return keep

def _get_grasp_score(g) -> float:
    """
    Try to extract a scalar score from a grasp object.
    Customize if your score lives somewhere else.
    """
    # Common patterns:
    for attr in ("score", "confidence", "q", "quality"):
        if hasattr(g, attr):
            val = getattr(g, attr)
            if isinstance(val, (int, float)):
                return float(val)
    # Dict-like
    try:
        return float(g.get("score"))  # type: ignore[attr-defined]
    except Exception:
        pass
    # Fallback = 0 (so it won't beat any real scored grasp)
    return 0.0

def _get_grasp_width(g) -> float:
    """
    Try to extract a scalar score from a grasp object.
    Customize if your score lives somewhere else.
    """
    # Common patterns:
    if hasattr(g, 'width'):
        val = getattr(g, 'width')
        if isinstance(val, (int, float)):
            return float(val)
    # Dict-like
    try:
        return float(g.get("width"))  # type: ignore[attr-defined]
    except Exception:
        pass
    # Fallback = 0 (so it won't beat any real scored grasp)
    return 0.0

def _normalize(values: List[float]) -> List[float]:
    if not values:
        return values
    vmin, vmax = min(values), max(values)
    if math.isclose(vmin, vmax):
        return [0.5 for _ in values]  # all equal
    return [(v - vmin) / (vmax - vmin) for v in values]

# def best_grasp_indices_per_box(gg, fx: float, fy: float, cx: float, cy: float, boxes: List[Dict], margin: float=0.0, return_stats: bool=False):
    """
    For each box, choose the grasp whose projected centre is inside and closest to the box centre.
    """
    centres = []
    for b in boxes or []:
        x, y, w, h = [float(v) for v in b["xywh"]]
        centres.append((x + 0.5 * w, y + 0.5 * h))

    proj = []
    for g in gg:
        T = grasp_T_from_grasp_obj(g)
        t = T[:3, 3]
        pv = project_cam_to_pixel(t, fx, fy, cx, cy)
        proj.append(pv)

    best_idxs = [None] * len(centres)
    best_d2   = [float("inf")] * len(centres)

    for i, pv in enumerate(proj):
        if pv is None:
            continue
        u, v = pv
        for j, b in enumerate(boxes or []):
            x, y, w, h = [float(v) for v in b["xywh"]]
            x0, y0 = x - margin, y - margin
            x1, y1 = x + w + margin, y + h + margin
            if (u >= x0) and (u <= x1) and (v >= y0) and (v <= y1):
                cx_b, cy_b = centres[j]
                du, dv = (u - cx_b), (v - cy_b)
                d2 = math.sqrt(du * du + dv * dv)
                if d2 < best_d2[j]:
                    best_d2[j] = d2
                    best_idxs[j] = i

    if not return_stats:
        return best_idxs

    stats = []
    for j, idx in enumerate(best_idxs):
        info = {
            "box_index": j,
            "box_centre": centres[j],
            "chosen_grasp_idx": idx,
            "pixel_distance": (float(best_d2[j]) ** 0.5) if best_idxs[j] is not None else None
        }
        stats.append(info)
    return best_idxs, stats

# def best_grasp_indices_per_box(
#     gg,
#     fx: float,
#     fy: float,
#     cx: float,
#     cy: float,
#     boxes: List[Dict],
#     margin: float = 0.0,
#     return_stats: bool = False,
#     criterion: str = "distance",   # "distance" | "score" | "both"
#     alpha: List[float] = [0.5,0.5,0.5],            # tradeoff for "both": 0..1 (distance weight)
# ):
#     """
#     For each box, choose a grasp among those whose projected centre falls inside it.

#     criterion:
#       - "distance": pick min pixel distance to box centre (original behavior)
#       - "score":    pick max grasp score
#       - "both":     pick argmin of alpha*norm_distance + (1-alpha)*(1 - norm_score)

#     Returns:
#       - if return_stats=False: List[Optional[int]] chosen grasp index per box
#       - else: (best_idxs, stats)
#     """
#     # Compute box centres in pixel space
#     centres = []
#     for b in boxes or []:
#         x, y, w, h = [float(v) for v in b["xywh"]]
#         centres.append((x + 0.5 * w, y + 0.5 * h))

#     # Project each grasp to pixel coords
#     proj = []
#     scores = []
#     width = []
#     for g in gg:
#         T = grasp_T_from_grasp_obj(g)
#         t = T[:3, 3]
#         pv = project_cam_to_pixel(t, fx, fy, cx, cy)
#         proj.append(pv)
#         scores.append(_get_grasp_score(g))
#         width.append(_get_grasp_width(g))
        

#     chosen_idxs = [None] * len(centres)
#     # Track also distance and score of the chosen for stats
#     chosen_dist = [None] * len(centres)
#     chosen_score = [None] * len(centres)
#     chosen_width = [None] * len(centres)

#     # For "distance" and "score" we can keep a simple running best
#     best_d = [float("inf")] * len(centres)
#     best_s = [-float("inf")] * len(centres)
#     min_width = [float("inf")] * len(centres)

#     # For "both", we need to evaluate over in-box candidates per box
#     in_box_candidates = [[] for _ in range(len(centres))]  # list of (idx, dist, score)

#     for i, pv in enumerate(proj):
#         if pv is None:
#             continue
#         u, v = pv
#         for j, b in enumerate(boxes or []):
#             x, y, w, h = [float(val) for val in b["xywh"]]
#             x0, y0 = x - margin, y - margin
#             x1, y1 = x + w + margin, y + h + margin
#             if (u >= x0) and (u <= x1) and (v >= y0) and (v <= y1):
#                 cx_b, cy_b = centres[j]
#                 du, dv = (u - cx_b), (v - cy_b)
#                 d = math.hypot(du, dv)

#                 if criterion == "distance":
#                     if d < best_d[j]:
#                         best_d[j] = d
#                         chosen_idxs[j] = i
#                         chosen_dist[j] = d
#                         chosen_score[j] = scores[i]
#                 elif criterion == "score":
#                     if scores[i] > best_s[j]:
#                         best_s[j] = scores[i]
#                         chosen_idxs[j] = i
#                         chosen_dist[j] = d
#                         chosen_score[j] = scores[i]
#                 elif criterion == "width":
#                    # Pick global max mwidt
#                     if width[i] < min_width[j]:
#                             min_width[j] = width[i]
#                             chosen_idxs[j] = i
#                             chosen_dist[j] = d
#                             chosen_score[j] = width[i]

#                 else:  # "both" (collect; choose after loop)
#                     in_box_candidates[j].append((i, d, scores[i],width[i]))

#     if criterion == "both":
#         # For each box, normalize distances & scores among its candidates and pick the best composite
#         for j, cand in enumerate(in_box_candidates):
#             if not cand:
#                 continue
#             idxs, dists, scs, wcs = zip(*cand)
#             nd = _normalize(list(dists))
#             ns = _normalize(list(scs))
#             ws = _normalize(list(wcs))
#             # minimize: alpha * nd + (1 - alpha) * (1 - ns)
#             best_val = float("inf")
#             best_k = 0
#             for k in range(len(cand)):
#                 # val = alpha * nd[k] + (1.0 - alpha) * (1.0 - ns[k]) * (1.0 - ws[k])
#                 val = alpha[0] * nd[k] + (1.0 - alpha[1]) * (1.0 - ns[k]) + (1.0 - alpha[2]) *(ws[k])#width uses lowest
   
#                 if val < best_val:
#                     best_val = val
#                     best_k = k
#             chosen_idxs[j] = idxs[best_k]
#             chosen_dist[j] = dists[best_k]
#             chosen_score[j] = scs[best_k]
#             chosen_width[j] = wcs[best_k]

#     if not return_stats:
#         return chosen_idxs

#     # Build stats per box, including individual winners by each criterion
#     # Compute also the separate winners for transparency
#     per_box_distance_winner = [None] * len(centres)
#     per_box_score_winner = [None] * len(centres)

#     # Recompute simple winners (so stats are consistent regardless of 'criterion')
#     tmp_best_d = [float("inf")] * len(centres)
#     tmp_best_s = [-float("inf")] * len(centres)

#     for i, pv in enumerate(proj):
#         if pv is None:
#             continue
#         u, v = pv
#         for j, b in enumerate(boxes or []):
#             x, y, w, h = [float(val) for val in b["xywh"]]
#             x0, y0 = x - margin, y - margin
#             x1, y1 = x + w + margin, y + h + margin
#             if (u >= x0) and (u <= x1) and (v >= y0) and (v <= y1):
#                 cx_b, cy_b = centres[j]
#                 d = math.hypot(u - cx_b, v - cy_b)
#                 if d < tmp_best_d[j]:
#                     tmp_best_d[j] = d
#                     per_box_distance_winner[j] = i
#                 if scores[i] > tmp_best_s[j]:
#                     tmp_best_s[j] = scores[i]
#                     per_box_score_winner[j] = i

#     stats = []
#     for j, idx in enumerate(chosen_idxs):
#         info = {
#             "box_index": j,
#             "box_centre": centres[j],
#             "chosen_grasp_idx": idx,
#             "chosen_pixel_distance": float(chosen_dist[j]) if chosen_dist[j] is not None else None,
#             "chosen_score": float(chosen_score[j]) if chosen_score[j] is not None else None,
#             "winner_by_distance_idx": per_box_distance_winner[j],
#             "winner_by_score_idx": per_box_score_winner[j],
#             "criterion": criterion,
#             "alpha": alpha if criterion == "both" else None,
#         }
#         stats.append(info)

#     return chosen_idxs, stats
def best_grasp_indices_per_box(
    gg,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    boxes: List[Dict],
    margin: float = 0.0,
    return_stats: bool = False,
    criterion: str = "distance",   # "distance" | "score" | "width" | "both"
    alpha: List[float] = [0.5, 0.5, 0.5],  # tradeoff for "both"
):
    """
    For each box, choose a grasp among those whose projected centre falls inside it.

    criterion:
      - "distance": pick min pixel distance to box centre (original behavior)
      - "score":    pick max grasp score
      - "width":    pick grasp whose width is closest to the median width in that box
      - "both":     pick argmin of alpha*norm_distance + (1-alpha)*(1 - norm_score) etc.

    Returns:
      - if return_stats=False: List[Optional[int]] chosen grasp index per box
      - else: (best_idxs, stats)
    """
    # Compute box centres in pixel space
    centres = []
    for b in boxes or []:
        x, y, w, h = [float(v) for v in b["xywh"]]
        centres.append((x + 0.5 * w, y + 0.5 * h))

    # Project each grasp to pixel coords
    proj = []
    scores = []
    width = []
    for g in gg:
        T = grasp_T_from_grasp_obj(g)
        t = T[:3, 3]
        pv = project_cam_to_pixel(t, fx, fy, cx, cy)
        proj.append(pv)
        scores.append(_get_grasp_score(g))
        width.append(_get_grasp_width(g))

    chosen_idxs = [None] * len(centres)
    chosen_dist = [None] * len(centres)
    chosen_score = [None] * len(centres)
    chosen_width = [None] * len(centres)

    # For "distance" and "score" we can keep a simple running best
    best_d = [float("inf")] * len(centres)
    best_s = [-float("inf")] * len(centres)

    # For "both" and "width", collect candidates per box
    in_box_candidates = [[] for _ in range(len(centres))]   # (idx, dist, score, width)
    width_candidates = [[] for _ in range(len(centres))]    # (idx, dist, width)

    for i, pv in enumerate(proj):
        if pv is None:
            continue
        u, v = pv
        for j, b in enumerate(boxes or []):
            x, y, w, h = [float(val) for val in b["xywh"]]
            x0, y0 = x - margin, y - margin
            x1, y1 = x + w + margin, y + h + margin
            if (u >= x0) and (u <= x1) and (v >= y0) and (v <= y1):
                cx_b, cy_b = centres[j]
                d = math.hypot(u - cx_b, v - cy_b)

                if criterion == "distance":
                    if d < best_d[j]:
                        best_d[j] = d
                        chosen_idxs[j] = i
                        chosen_dist[j] = d
                        chosen_score[j] = scores[i]
                        chosen_width[j] = width[i]
                elif criterion == "score":
                    if scores[i] > best_s[j]:
                        best_s[j] = scores[i]
                        chosen_idxs[j] = i
                        chosen_dist[j] = d
                        chosen_score[j] = scores[i]
                        chosen_width[j] = width[i]
                elif criterion == "width":
                    # collect candidates; decide after loop using median width
                    width_candidates[j].append((i, d, width[i]))
                else:  # "both"
                    in_box_candidates[j].append((i, d, scores[i], width[i]))

    # Handle "width": choose candidate closest to median width per box
    if criterion == "width":
        for j, cand in enumerate(width_candidates):
            if not cand:
                continue
            idxs, dists, wcs = zip(*cand)
            median_w = float(np.median(wcs))
            best_val = float("inf")
            best_k = 0
            for k, wc in enumerate(wcs):
                val = abs(wc - median_w)
                if val < best_val:
                    best_val = val
                    best_k = k
            chosen_idxs[j] = idxs[best_k]
            chosen_dist[j] = dists[best_k]
            chosen_score[j] = wcs[best_k]   # you can treat width as the score here, or leave as None
            chosen_width[j] = wcs[best_k]

    # Handle "both": distance + score (+ width) multi-objective
    if criterion == "both":
        for j, cand in enumerate(in_box_candidates):
            if not cand:
                continue
            idxs, dists, scs, wcs = zip(*cand)
            nd = _normalize(list(dists))
            ns = _normalize(list(scs))
            ws = _normalize(list(wcs))
            best_val = float("inf")
            best_k = 0
            for k in range(len(cand)):
                # width uses lowest -> smaller ws is better
                val = alpha[0] * nd[k] + (1.0 - alpha[1]) * (1.0 - ns[k]) + (1.0 - alpha[2]) * ws[k]
                if val < best_val:
                    best_val = val
                    best_k = k
            chosen_idxs[j] = idxs[best_k]
            chosen_dist[j] = dists[best_k]
            chosen_score[j] = scs[best_k]
            chosen_width[j] = wcs[best_k]

    if not return_stats:
        return chosen_idxs

    # Build stats (including per-criterion simple winners for transparency)
    per_box_distance_winner = [None] * len(centres)
    per_box_score_winner = [None] * len(centres)
    tmp_best_d = [float("inf")] * len(centres)
    tmp_best_s = [-float("inf")] * len(centres)

    for i, pv in enumerate(proj):
        if pv is None:
            continue
        u, v = pv
        for j, b in enumerate(boxes or []):
            x, y, w, h = [float(val) for val in b["xywh"]]
            x0, y0 = x - margin, y - margin
            x1, y1 = x + w + margin, y + h + margin
            if (u >= x0) and (u <= x1) and (v >= y0) and (v <= y1):
                cx_b, cy_b = centres[j]
                d = math.hypot(u - cx_b, v - cy_b)
                if d < tmp_best_d[j]:
                    tmp_best_d[j] = d
                    per_box_distance_winner[j] = i
                if scores[i] > tmp_best_s[j]:
                    tmp_best_s[j] = scores[i]
                    per_box_score_winner[j] = i

    stats = []
    for j, idx in enumerate(chosen_idxs):
        info = {
            "box_index": j,
            "box_centre": centres[j],
            "chosen_grasp_idx": idx,
            "chosen_pixel_distance": float(chosen_dist[j]) if chosen_dist[j] is not None else None,
            "chosen_score": float(chosen_score[j]) if chosen_score[j] is not None else None,
            "chosen_width": float(chosen_width[j]) if chosen_width[j] is not None else None,
            "winner_by_distance_idx": per_box_distance_winner[j],
            "winner_by_score_idx": per_box_score_winner[j],
            "criterion": criterion,
            "alpha": alpha if criterion == "both" else None,
        }
        stats.append(info)

    return chosen_idxs, stats

# -------------------- Robot-target helpers --------------------

def robot_tcp_targets_from_grasp(T_base_cam: np.ndarray, T_tcp_gripper: np.ndarray, grasp_T_cam: np.ndarray,
                                 approach_offset_m: float, retreat_m: float):
    """
    Produce robot Active-TCP targets (pre, contact, retreat) from a grasp in the camera frame.
    Returns poses as [x,y,z,rx,ry,rz] and their 4x4 matrices.
    """
    # 1) Gripper TCP contact pose in BASE
    T_base_gripper_contact = T_base_cam @ grasp_T_cam

    # 2) Pre/retreat along grasp -Z
    z_axis = -T_base_gripper_contact[:3, 2]
    T_base_gripper_pre = T_base_gripper_contact.copy()
    T_base_gripper_pre[:3, 3] = T_base_gripper_pre[:3, 3] + z_axis * approach_offset_m

    T_base_gripper_retreat = T_base_gripper_contact.copy()
    T_base_gripper_retreat[:3, 3] = T_base_gripper_retreat[:3, 3] - z_axis * retreat_m

    # 3) Convert gripperTCP -> active robot TCP
    T_gripper_robotTCP = np.linalg.inv(T_tcp_gripper)
    T_base_robot_pre     = T_base_gripper_pre @ T_gripper_robotTCP
    T_base_robot_contact = T_base_gripper_contact @ T_gripper_robotTCP
    T_base_robot_retreat = T_base_gripper_retreat @ T_gripper_robotTCP

    pre_p     = pose_from_T(T_base_robot_pre)
    contact_p = pose_from_T(T_base_robot_contact)
    retreat_p = pose_from_T(T_base_robot_retreat)
    return pre_p, contact_p, retreat_p, T_base_robot_pre, T_base_robot_contact, T_base_robot_retreat, T_base_gripper_contact


def select_top_conf_box(boxes: List[Dict]) -> List[Dict]:
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda x: x.get("confidence", 0), reverse=True)
    return [boxes[0]]


def xy_to_bboxs(xy_list: List[Dict],W:int,H:int,bbox_size:int = 10) -> List[Dict]:
    fixed = []
    half_box = bbox_size // 2
    for xy in xy_list:
        x, y, = xy  # ignore any model-provided w/h
        x = int(round(x))
        y = int(round(y))
        # Calculate top-left corner of bbox
        x1 = max(0, min(x - half_box, W - 1))
        y1 = max(0, min(y - half_box, H - 1))
        # Calculate width and height, ensuring bbox stays within image bounds
        w = min(bbox_size, W - x1)
        h = min(bbox_size, H - y1)
        nb = dict(b)
        nb["xywh"] = [x1, y1, w, h]
        fixed.append(nb)


def fixed_bbox_from_centre(
    bboxes: List[Dict],
    bbox_size: int = 10,
) -> List[Dict]:
    half = bbox_size // 2

    for b in bboxes:
        cx, cy = b["centre"]
        W,H  = b["image_size"]
        x = int(round(cx))
        y = int(round(cy))
        x1 = x - half
        y1 = y - half

        if W is not None and H is not None:
            x1 = max(0, min(W - bbox_size, x1))
            y1 = max(0, min(H - bbox_size, y1))
        else:
            x1 = max(0, x1)
            y1 = max(0, y1)

        b["xywh"] = [x1, y1, bbox_size, bbox_size]
    
    remove_bboxes_outside_image(bboxes)

    return bboxes


def remove_bboxes_outside_image(
    bboxes: List[Dict],
    include_edge: bool = True
) -> List[Dict]:

    def centre_in_bounds(c,size):
        W,H = size
        if not isinstance(c, (list, tuple)) or len(c) < 2:
            return False
        cx, cy = c[0], c[1]
        if not (isinstance(cx, (int, float)) and isinstance(cy, (int, float))):
            return False
        if not (math.isfinite(cx) and math.isfinite(cy)):
            return False

        if include_edge:
            return (0 <= cx < W) and (0 <= cy < H)
        else:
            return (0 < cx < W - 1) and (0 < cy < H - 1)

    return [b for b in bboxes if centre_in_bounds(b.get("centre"),b.get("image_size"))]

# ------------------------------ paths ------------------------------

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

# ------------------------------ small helpers ------------------------------
def norm(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + eps)

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def deep_update(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in (patch or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = deep_update(base[k], v)
        else:
            base[k] = v
    return base

def as_np_bounds(xy_bounds):
    if xy_bounds is None:
        return None
    a = np.asarray(xy_bounds, dtype=float)
    if a.shape != (2, 2):
        raise ValueError("xy_bounds must be [[xmin,xmax],[ymin,ymax]] or null")
    return a

def as_tuple2(z_bounds):
    if z_bounds is None:
        return None
    a = tuple(float(v) for v in z_bounds)
    if len(a) != 2:
        raise ValueError("z_bounds must be [zmin, zmax] or null")
    return a

# ---------- orientation / SO(3) ----------
def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    return np.array([[0, -z, y],[z, 0, -x],[-y, x, 0]], dtype=float)

def exp_so3(w: np.ndarray) -> np.ndarray:
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3) + skew(w)
    k = w / th
    K = skew(k)
    c, s = math.cos(th), math.sin(th)
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)

def log_so3(R: np.ndarray) -> np.ndarray:
    tr = float(np.trace(R))
    cos_th = max(-1.0, min(1.0, (tr - 1.0) * 0.5))
    th = math.acos(cos_th)
    if th < 1e-12:
        return np.zeros(3, dtype=float)
    denom = 2.0 * math.sin(th)
    wx = (R[2, 1] - R[1, 2]) / denom
    wy = (R[0, 2] - R[2, 0]) / denom
    wz = (R[1, 0] - R[0, 1]) / denom
    return th * np.array([wx, wy, wz], dtype=float)

def slerp_R(Ra: np.ndarray, Rb: np.ndarray, a: float) -> np.ndarray:
    a = float(max(0.0, min(1.0, a)))
    Rrel = Ra.T @ Rb
    w = log_so3(Rrel)
    return Ra @ exp_so3(a * w)

def look_at_relative(
    from_p: np.ndarray,
    to_p: np.ndarray,
    R_ref: np.ndarray,
    up_hint: np.ndarray = np.array([0.0, 0.0, 1.0]),
    preserve_roll: bool = True,
) -> np.ndarray:
    z = norm(to_p - from_p)
    if not preserve_roll:
        x = norm(np.cross(up_hint, z))
        if np.linalg.norm(x) < 1e-6:
            x = norm(np.cross(R_ref[:, 1], z))
        y = np.cross(z, x)
        return np.stack([x, y, z], axis=1)
    x_ref = R_ref[:, 0]
    x_proj = x_ref - z * float(np.dot(z, x_ref))
    if np.linalg.norm(x_proj) < 1e-6:
        x_proj = np.cross(up_hint, z)
        if np.linalg.norm(x_proj) < 1e-6:
            x_proj = np.cross(R_ref[:, 1], z)
    x = norm(x_proj)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)

def get_T_to_RPY(T,use_deg=False):

    R = T[:3, :3]

    # Protect against numerical issues
    sy = -R[2, 0]
    sy = np.clip(sy, -1.0, 1.0)

    pitch = math.asin(sy)
    roll  = math.atan2(R[2, 1], R[2, 2])
    yaw   = math.atan2(R[1, 0], R[0, 0])

    # Convert to degrees
    if use_deg:
        roll  = math.degrees(roll)
        pitch = math.degrees(pitch)
        yaw   = math.degrees(yaw)

    return [roll, pitch, yaw]
    
def has_any(x):
    if x is None:
        return False
    if isinstance(x, dict):
        return len(x) > 0
    try:
        # works for lists/tuples/ndarrays
        return np.size(x) > 0
    except Exception:
        return False