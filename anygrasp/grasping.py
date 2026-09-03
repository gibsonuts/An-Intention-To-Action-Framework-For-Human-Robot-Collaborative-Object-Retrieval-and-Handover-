"""
Grasping calculations (cloud build, AnyGrasp inference, grasp selection, and motion target generation).

Drop this next to commons/grasp_utils.py as commons/grasping_calcs.py
"""
import sys
sys.path.append('anygrasp')

from typing import Dict, List, Tuple, Optional, Any
import math
import numpy as np
import os
import yaml
from typing import Dict
from anygrasp import grasping
# from tracker import AnyGraspTracker
# External deps from your repo
from graspnetAPI import GraspGroup
from anygrasp.gsnet import AnyGrasp
import commons.grasp_utils as utils
import commons.utils as common_utils
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import copy

import open3d as o3d

class GraspComputationError(RuntimeError):
    """Raised when a grasp cannot be computed/selected."""



class AnyGraspEngine:
    """
    Thin wrapper around AnyGrasp to keep your main script tidy.
    """

    def __init__(
        self,
        camera_intrinsics = None,  # dict with fx,fy,cx,cy,depth_scale
        debug: bool = False,
    ):
        class Dot:  # simple cfg container expected by AnyGrasp
            pass

        # Load YAML config
        #get grasping_config.yaml path
        config_path = os.path.join(os.path.dirname(__file__), "config/grasping_config.yaml")
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)

        flow  = cfg['workflow']

        grasp = flow['grasp']

        cfgs = Dot()
        cfgs.checkpoint_path = cfg['checkpoint_path']
        cfgs.max_gripper_width = grasp['max_gripper_width']
        cfgs.gripper_height = grasp['gripper_height']
        cfgs.top_down_grasp = grasp['top_down_only']
        cfgs.debug = debug

        tracker_cfgs = Dot()
        tracker_cfgs.checkpoint_path = cfg['tracker_checkpoint_path']
        tracker_cfgs.max_gripper_width = grasp['max_gripper_width']
        tracker_cfgs.gripper_height = grasp['gripper_height']
        tracker_cfgs.top_down_grasp = grasp['top_down_only']
        tracker_cfgs.filter = grasp['filter']
        tracker_cfgs.debug = debug

        self.apply_object_mask=grasp['apply_object_mask']
        self.dense_grasp=grasp['dense_grasp'],
        self.collision_detection=grasp['collision_detection'],
        
        self.debug = debug
        self.flow  = flow

        #speeds
        # assume: flow = cfg['workflow']
        approach = flow['approach']
        motion   = flow['motion']

        self.approach_offset_m=float(approach['offset_m']),
        self.retreat_m=float(approach['retreat_m']),

        self.descend_speed = float(approach['descend_speed'])
        self.descend_accel = float(approach['descend_accel'])  # was retreat_m by mistake
        self.retreat_m     = float(approach['retreat_m'])

        self.move_speed = float(motion['move_speed'])
        self.move_accel = float(motion['move_accel'])

        self.fx = camera_intrinsics['fx'] if camera_intrinsics else None
        self.fy = camera_intrinsics['fy'] if camera_intrinsics else None
        self.cx = camera_intrinsics['cx'] if camera_intrinsics else None
        self.cy = camera_intrinsics['cy'] if camera_intrinsics else None
        self.scale = camera_intrinsics['depth_scale'] if camera_intrinsics else None
        self._ag = AnyGrasp(cfgs)
        self._ag.load_net()

        # self._ag_tracker = AnyGraspTracker(cfgs)
        # self._ag_tracker.load_net()
        self.grasp_ids = [0]

        self.cloud = None
        self.gg = None

    def check_reachability_for_grasps(
        self,
        gg: GraspGroup,
        arm ,
        T_base_tcp,
        T_tcp_cam,
        T_tcp_gripper,
        *,
        boxes: Optional[List[Dict]] = None,
        margins: int = 20,
        poses: Tuple[str, ...] = ("pre", "contact"),  # check BOTH by default
        require_all: bool = True,                    # True => all poses must be reachable
        filter_to_reachable: bool = True,
        debug: bool = False,
    ):
        """
        Compute grasp targets and test reachability for each grasp at multiple poses.
        By default checks both ("pre","contact") and requires both to be reachable.

        Returns dict with:
          - gg_eval: evaluated grasps (after box filtering if any)
          - kept_indices: indices into ORIGINAL gg for gg_eval
          - reachable: list of per-grasp dicts (includes reachability per pose)
          - unreachable: list of per-grasp dicts
          - gg_reachable: GraspGroup containing only reachable grasps (if filter_to_reachable)
          - reachable_indices: indices into ORIGINAL gg for reachable grasps
        """
        if gg is None or len(gg) == 0:
            return {
                "gg_eval": None,
                "kept_indices": [],
                "reachable": [],
                "unreachable": [],
                "gg_reachable": None,
                "reachable_indices": [],
            }

        valid_poses = {"pre", "contact", "retreat"}
        for p in poses:
            if p not in valid_poses:
                raise ValueError(f"Invalid pose '{p}'. Must be one of {sorted(valid_poses)}")

        # Optional box filter first
        gg_eval = gg
        kept_indices = list(range(len(gg)))
        if boxes:
            gg_eval, kept_indices = self.filter_grasps_to_boxes(gg, boxes, margins=margins)
            if gg_eval is None or len(gg_eval) == 0:
                return {
                    "gg_eval": None,
                    "kept_indices": kept_indices,
                    "reachable": [],
                    "unreachable": [],
                    "gg_reachable": None,
                    "reachable_indices": [],
                }

        reachable = []
        unreachable = []
        reachable_local_idx: List[int] = []
        reachable_global_idx: List[int] = []

        def _T_for_pose(targets: Dict[str, Any], pose_name: str) -> np.ndarray:
            if pose_name == "pre":
                return targets["T_base_robot_pre"]
            if pose_name == "contact":
                return targets["T_base_robot_contact"]
            return targets["T_base_robot_retreat"]

        for local_i, g in enumerate(gg_eval):
            global_i = kept_indices[local_i]

            try:
                targets = self.compute_grasp_targets(
                    chosen_grasp=g,
                    T_base_tcp=T_base_tcp,
                    T_tcp_cam=T_tcp_cam,
                    T_tcp_gripper=T_tcp_gripper,
                )

                per_pose = {}
                ok_list = []

                for pose_name in poses:
                    T_check = _T_for_pose(targets, pose_name)
                    pose_ur = common_utils.matrix_to_ur_pose(T_check)
                    ok, analysis = arm.check_pose_reachable(pose_ur)

                    per_pose[pose_name] = {
                        "reachable": bool(ok),
                        "pose_ur": pose_ur,
                        "analysis": analysis,
                    }
                    ok_list.append(bool(ok))

                overall_ok = all(ok_list) if require_all else any(ok_list)

                rec = {
                    "global_index": int(global_i),
                    "local_index": int(local_i),
                    "overall_reachable": bool(overall_ok),
                    "per_pose": per_pose,     # <-- reachability for "pre"/"contact"
                    "targets": targets,
                    "score": float(utils._get_grasp_score(g)) if hasattr(utils, "_get_grasp_score") else None,
                    "width": float(utils._get_grasp_width(g)) if hasattr(utils, "_get_grasp_width") else None,
                }

                if overall_ok:
                    reachable.append(rec)
                    reachable_local_idx.append(local_i)
                    reachable_global_idx.append(global_i)
                    if debug:
                        print(f"[REACHABLE] idx={global_i} (local={local_i}) "
                              f"pre={per_pose.get('pre',{}).get('reachable')} "
                              f"contact={per_pose.get('contact',{}).get('reachable')}")
                else:
                    unreachable.append(rec)
                    if debug:
                        print(f"[UNREACHABLE] idx={global_i} (local={local_i}) "
                              f"pre={per_pose.get('pre',{}).get('reachable')} "
                              f"contact={per_pose.get('contact',{}).get('reachable')}")

            except Exception as e:
                rec = {
                    "global_index": int(global_i),
                    "local_index": int(local_i),
                    "overall_reachable": False,
                    "per_pose": {},
                    "targets": None,
                    "analysis": {"exception": repr(e)},
                }
                unreachable.append(rec)
                if debug:
                    print(f"[ERROR] idx={global_i} failed: {e}")

        gg_reachable = None
        if filter_to_reachable and reachable_local_idx:
            # slice if supported
            try:
                gg_reachable = gg_eval[reachable_local_idx]
                if not isinstance(gg_reachable, GraspGroup):
                    raise TypeError
            except Exception:
                gg_reachable = GraspGroup()
                for li in reachable_local_idx:
                    gg_reachable.add(gg_eval[li])

        return {
            "gg_eval": gg_eval,
            "kept_indices": kept_indices,
            "reachable": reachable,
            "unreachable": unreachable,
            "gg_reachable": gg_reachable,
            "reachable_indices": reachable_global_idx,
        }
    
        
    def generate_grasps(
        self,
        pts: np.ndarray,
        cols: np.ndarray,
        debug: bool = False,
        *,
        repeats: int = 10,          # run AnyGrasp this many times and merge results
        do_nms: bool = True,       # keep your original behavior by default
        correct_grasp_rotation: bool = True,
    ) -> Tuple[GraspGroup, np.ndarray]:
        """
        Run AnyGrasp 'repeats' times and return a merged, optionally NMS'ed + sorted GraspGroup
        and the (optional) open3d cloud from the first successful pass.

        Returns:
            (GraspGroup, cloud) or (None, None) if no grasps found across all passes.
        """
        if 'workspace_cam' not in self.flow:
            raise ValueError("workflow.workspace_cam not found in grasping_config.yaml")

        ws = self.flow['workspace_cam']

        # ---------- Prep ----------
        pts = pts.astype(np.float32, copy=False)
        cols = cols.astype(np.float32, copy=False)

        merged = GraspGroup()
        first_cloud = None

        # ---------- Repeated AnyGrasp inference ----------
        num_passes = max(1, int(repeats))
        for _ in range(num_passes):
            gg_k, cloud_k = self._ag.get_grasp(
                pts, cols,
                lims=[ws['xmin'], ws['xmax'], ws['ymin'], ws['ymax'], ws['zmin'], ws['zmax']],
                apply_object_mask=self.apply_object_mask,
                dense_grasp=self.dense_grasp,
                collision_detection=self.collision_detection,
            )

            if gg_k is None or len(gg_k) == 0:
                continue

            if first_cloud is None:
                first_cloud = cloud_k

            # Append results into merged
            try:
                # Fast path if GraspGroup supports += concatenation in your version
                for g in gg_k:
                    merged.add(g)
            except Exception:
                # Fallback: explicit add loop already used above; kept for clarity
                for g in gg_k:
                    merged.add(g)

        # ---------- Post-process ----------
        if len(merged) == 0:
            # nothing detected across all passes
            return None, None

        if do_nms:
            merged = merged.nms()

        # sort by score (descending)
        merged = merged.sort_by_score()
        if len(merged) == 0:
            raise GraspComputationError("No grasp left after NMS/score sorting")

        # store for later visualization helpers
        self.cloud = first_cloud
        self.gg = merged
        
        if debug:
            grippers = merged.to_open3d_geometry_list()
            size=0.1
            f = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
            o3d.visualization.draw_geometries([f,first_cloud, *grippers])

        #rotate grasps to match coordinate system
        R_y90 = np.array([
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [-1, 0, 0, 0],
            [0, 0, 0, 1]
        ], dtype=float)
        R_z90 = np.array([
            [0, -1, 0, 0],
            [1,  0, 0, 0],
            [0,  0, 1, 0],
            [0,  0, 0, 1]
        ], dtype=float)
        correct_rotation = R_y90 @ R_z90
        if correct_grasp_rotation:
            for i,g in enumerate(self.gg):
                T = utils.grasp_T_from_grasp_obj(g)
                T=T@correct_rotation
                T_new = T.copy()
                utils.set_T_on_grasp_group(self.gg,i,T_new)


        return merged, first_cloud
    
    def visualise_grasps(self,gg,cloud):
        
        grippers = gg.to_open3d_geometry_list()
        size=0.1
        f = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        o3d.visualization.draw_geometries([f,cloud, *grippers])

      

    def generate_grasps_track(
        self,
        pts: np.ndarray,
        cols: np.ndarray,
        grasp_ids_to_track: List=None,
        debug: bool = False
    ) -> Tuple[GraspGroup, np.ndarray]:
  
        # ---------- AnyGrasp inference on fused cloud ----------
        pts = pts.astype(np.float32, copy=False)
        cols = cols.astype(np.float32, copy=False)

        if grasp_ids_to_track is None:
            grasp_ids_to_track = [0]

        target_gg, curr_gg, target_grasp_ids, corres_preds = self._ag_tracker.update(
            pts, 
            cols,
            grasp_ids_to_track)

        if debug:
            vis = o3d.visualization.Visualizer()
            vis.create_window(height=720, width=1280)
            trans_mat = np.array([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(pts)
            cloud.colors = o3d.utility.Vector3dVector(c)
            cloud.transform(trans_mat)
            grippers = target_gg.to_open3d_geometry_list()
            for gripper in grippers:
                gripper.transform(trans_mat)
            vis.add_geometry(cloud)
            for gripper in grippers:
                vis.add_geometry(gripper)
            vis.poll_events()
            vis.remove_geometry(cloud)
            for gripper in grippers:
                vis.remove_geometry(gripper)


        return target_gg,target_grasp_ids
    

    # ---------- Monkey-patch: build_workspace_from_cloud (base-frame) ----------
    def build_workspace_from_cloud(self, pts_base: np.ndarray, cols: np.ndarray):
        """
        Filters a base-frame point cloud using workflow.workspace_base bounds from self.flow.
        Returns (pts, cols, mask). 'mask' indexes the input pts_base.
        """
        if "workspace_cam" not in self.flow:
            raise ValueError("workflow.workspace_base not found in grasping_config.yaml")
        ws = self.flow["workspace_cam"]

        assert pts_base.ndim == 2 and pts_base.shape[1] == 3, "pts_base must be Nx3"
        assert cols.ndim == 2 and cols.shape[1] == 3 and len(cols) == len(
            pts_base
        ), "cols must be Nx3 and same length as pts_base"

        x, y, z = pts_base[:, 0], pts_base[:, 1], pts_base[:, 2]
        mask = (
            (x > ws["xmin"])
            & (x < ws["xmax"])
            & (y > ws["ymin"])
            & (y < ws["ymax"])
            & (z > ws["zmin"])
            & (z < ws["zmax"])
        )
        return pts_base[mask], cols[mask], mask


    def build_workspace_cloud(
        self,
        depth: np.ndarray,
        color: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        if 'workspace_cam' not in self.flow:
            raise ValueError("workflow.workspace_cam not found in grasping_config.yaml")

        ws = self.flow['workspace_cam']

        assert depth.ndim == 2 and color.ndim == 3, "Expected depth HxW and color HxWx3"
        colors = color.astype(np.float32) / 255.0
        depths = depth.astype(np.float32)

        H, W = depths.shape
        xmap, ymap = np.meshgrid(np.arange(W), np.arange(H))

        points_z = depths * self.scale
        points_x = (xmap - self.cx) / self.fx * points_z
        points_y = (ymap - self.cy) / self.fy * points_z

        mask = (
            (points_z > ws['zmin']) & (points_z < ws['zmax']) &
            (points_x > ws['xmin']) & (points_x < ws['xmax']) &
            (points_y > ws['ymin']) & (points_y < ws['ymax'])
        )

        pts = np.stack([points_x, points_y, points_z], axis=-1)
        pts = pts[mask].astype(np.float32).reshape(-1, 3)
        cols = colors[mask].astype(np.float32).reshape(-1, 3)
        return pts, cols, mask

    def visualize_grasp_box_filter(self,color_rgb: np.ndarray, gg, object_bboxes: List[Dict]):
        """
        Quick overlay to show grasp centres vs. LLM boxes.
        """
        # Precompute projected centres
        proj = []
        for i, g in enumerate(gg):
            T = utils.grasp_T_from_grasp_obj(g)
            t = T[:3, 3]
            pv = utils.project_cam_to_pixel(t, self.fx, self.fy, self.cx, self.cy)
            if pv is None:
                proj.append((None, None, None))
            else:
                u, v = pv
                proj.append((u, v, float(t[2])))

        canvas = color_rgb.copy()

        # Draw boxes
        for b in object_bboxes or []:
            x, y, w, h = b["xywh"]
            label = b.get("label", "")
            conf  = b.get("confidence", None)
            cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 165, 255), 2)
            if label or conf is not None:
                txt = label
                if conf is not None: txt += f" ({conf:.2f})"
                cv2.putText(canvas, txt, (x, max(0, y - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2, cv2.LINE_AA)

        keep_idx = utils.indices_of_grasps_in_boxes(gg, self.fx, self.fy, self.cx, self.cy, object_bboxes, margin=10)
        inside = set(keep_idx)

        for i, (u, v, Z) in enumerate(proj):
            if u is None:
                continue
            pt = (int(round(u)), int(round(v)))
            color = (0, 220, 0) if i in inside else (0, 0, 220)
            cv2.circle(canvas, pt, 4, color, -1)

        fig, ax = plt.subplots()
        ax.imshow(canvas)
        ax.set_axis_off()
        plt.tight_layout()
        plt.show()

    def to_o3d_pointcloud(self,pts_base: np.ndarray,
                        cols: np.ndarray | None = None) -> o3d.geometry.PointCloud:
        """
        Convert Nx3 points (meters, base frame) and optional Nx3 uint8 colors into an Open3D PointCloud.
        """
        assert pts_base.ndim == 2 and pts_base.shape[1] == 3, "pts_base must be Nx3"

        # Remove invalid rows
        mask = np.isfinite(pts_base).all(axis=1)
        if cols is not None:
            mask &= np.isfinite(cols).all(axis=1)
        pts = pts_base[mask].astype(np.float64, copy=False)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)

        if cols is not None:
            # Expect uint8 0..255 -> float 0..1
            if cols.dtype != np.float32 and cols.dtype != np.float64:
                colors = (cols[mask].astype(np.float32) / 255.0)
            else:
                colors = cols[mask]
                # If colors look like 0..255 but float, scale down
                if colors.max() > 1.0:
                    colors = colors / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64, copy=False))

        return pcd

    def visualise3dpointcloud(self,cloud,T_base_cam=None):
     

        def ensure_pointcloud(pc):
            if isinstance(pc, o3d.geometry.PointCloud):
                return pc
            pc_o3d = o3d.geometry.PointCloud()
            pc_o3d.points = o3d.utility.Vector3dVector(pc)
            return pc_o3d

        cloud_viz = ensure_pointcloud(cloud)
        if T_base_cam:
            cloud_viz.transform(T_base_cam)
    
        o3d.visualization.draw_geometries([
                    cloud_viz, 
                ])

    def setgg(self,gg,delta=None,correct_grasp_rotation=True):
        local_gg = copy.deepcopy(gg)

        if delta is not None:
            local_gg = utils.rotate_grasp_objects(local_gg,delta)

        #rotate grasps to match coordinate system
        R_y90 = np.array([
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [-1, 0, 0, 0],
            [0, 0, 0, 1]
        ], dtype=float)
        R_z90 = np.array([
            [0, -1, 0, 0],
            [1,  0, 0, 0],
            [0,  0, 1, 0],
            [0,  0, 0, 1]
        ], dtype=float)
        correct_rotation = R_y90 @ R_z90
        if correct_grasp_rotation:
            for i,g in enumerate(local_gg):
                T = utils.grasp_T_from_grasp_obj(g)
                T=T@np.linalg.inv(correct_rotation)
                T_new = T.copy()
                utils.set_T_on_grasp_group(local_gg,i,T_new)

        self.gg = local_gg

    def visualise3dgrasp(self,T_base_cam,T_base_tcp,T_base_robot_pre,T_base_robot_contact,T_base_gripper_contact):
        cloud = self.cloud
        gg = self.gg

        if cloud is None or gg is None:
            print('Error please first perform grasp detection',cloud,gg)
            return 

        def frame_from_T(T, size=0.06):
            f = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
            f.transform(T)
            return f
        def ensure_pointcloud(pc):
            if isinstance(pc, o3d.geometry.PointCloud):
                return pc
            pc_o3d = o3d.geometry.PointCloud()
            pc_o3d.points = o3d.utility.Vector3dVector(pc)
            return pc_o3d

        cloud_viz = ensure_pointcloud(cloud)
        cloud_viz.transform(T_base_cam)

        grippers = gg.to_open3d_geometry_list()
        for i, gviz in enumerate(grippers):
            gviz.transform(T_base_cam)
            gviz.paint_uniform_color([0, 1, 0] if i == 0 else [1, 0, 0])

        base_frame = frame_from_T(np.eye(4), size=0.08)
        tcp_frame_live = frame_from_T(T_base_tcp, size=0.06)
        cam_frame = frame_from_T(T_base_cam, size=0.09)
        tcp_approach_frame = frame_from_T(T_base_robot_pre, size=0.06)
        tcp_target_frame = frame_from_T(T_base_robot_contact, size=0.06)
        grasp_contact_frame = frame_from_T(T_base_gripper_contact, size=0.10)

        o3d.visualization.draw_geometries([
            cloud_viz, *grippers,
            base_frame, tcp_frame_live, tcp_target_frame, tcp_approach_frame,
            cam_frame, grasp_contact_frame
        ])

    def filter_grasps_to_boxes(
        self,
        gg: GraspGroup,
        boxes: Optional[List[Dict]],
        margins: int = 20,
    ) -> Tuple[GraspGroup, List[int]]:
        """
        Keep only grasps whose projected centres land inside the provided boxes.
        Returns the filtered GraspGroup and the kept indices.
        """
        if not boxes:
            return None, list(range(len(gg)))

        keep_idx = utils.indices_of_grasps_in_boxes(gg, self.fx, self.fy, self.cx, self.cy, boxes,margins)
        if len(keep_idx) == 0:
            print("No grasps fall inside the detected bounding boxes. Try a different query or margin.")
            return None, list(range(len(gg)))

        # GraspGroup supports slicing in newer versions; fall back to manual copy if needed
        try:
            gg_f = gg[keep_idx]
            if isinstance(gg_f, GraspGroup):
                return gg_f, keep_idx
            raise TypeError
        except Exception:
            gg_new = GraspGroup()
            for i in keep_idx:
                gg_new.add(gg[i])
            return gg_new, keep_idx



    # def choose_grasp(
    #     self,
    #     gg: GraspGroup,
    #     boxes: Optional[List[Dict]]
    # ) -> Tuple[object, Dict]:
    #     """
    #     Choose a single grasp. If boxes are given: pick best per-box and take the first available.
    #     Otherwise: take the top-scoring grasp overall.
    #     Returns (chosen_grasp, info_dict).
    #     """
    #     if boxes:
    #         best_idxs, stats = utils.best_grasp_indices_per_box(
    #             gg, self.fx, self.fy, self.cx, self.cy, boxes, return_stats=True
    #         )
    #         idx_for_motion = next((i for i in best_idxs if i is not None), None)
    #         if idx_for_motion is None:
    #             raise GraspComputationError("No per-box best grasps found inside any LLM box.")
    #         return gg[idx_for_motion], {"best_idxs": best_idxs, "stats": stats}

    #     return gg[0], {"best_idxs": [0], "stats": []}

    def _normalize_angle_deg(self,a: float) -> float:
        """Normalize angle to [-180, 180) degrees."""
        return (a + 180.0) % 360.0 - 180.0

    def filter_grasps_by_angle_limits(self,
        gg: GraspGroup,
        roll_limits_deg: Optional[Tuple[float, float]] = None,
        pitch_limits_deg: Optional[Tuple[float, float]] = None,
        yaw_limits_deg: Optional[Tuple[float, float]] = None,
    ) -> Tuple[Optional[GraspGroup], List[int]]:
        """
        Filter grasps based on roll/pitch/yaw limits.

        Angles are interpreted as ZYX RPY (same convention as utils.rpy_from_R_zyx).

        Parameters
        ----------
        gg : GraspGroup
            Input grasps.
        roll_limits_deg : (min_roll, max_roll) in degrees, or None to skip.
        pitch_limits_deg : (min_pitch, max_pitch) in degrees, or None to skip.
        yaw_limits_deg : (min_yaw, max_yaw) in degrees, or None to skip.

        Returns
        -------
        gg_f : GraspGroup or None
            Filtered GraspGroup (None if no grasps remain).
        keep_idx : list[int]
            Indices in the original GraspGroup that were kept.
        """
        if gg is None or len(gg) == 0:
            return None, []

        keep_idx: List[int] = []
        print('roll_limits',roll_limits_deg,'pitch_limits',pitch_limits_deg,'yaw_limits',yaw_limits_deg)

        for i, g in enumerate(gg):
            T = utils.grasp_T_from_grasp_obj(g)
            R = T[0:3,0:3]
            # ---- get rotation matrix for this grasp ----
            # Adjust attribute name if AnyGrasp uses a different one
            # if hasattr(g, "rotation_matrix"):
            #     R = g.rotation_matrix
            # elif hasattr(g, "rotation"):
            #     R = g.rotation
            # else:
            #     raise AttributeError("Grasp object has no rotation matrix attribute")

            # ZYX roll, pitch, yaw (same as you use for T_base_tcp)
            roll, pitch, yaw = utils.rpy_from_R_zyx(R)

            roll = np.rad2deg(roll)#self._normalize_angle_deg(np.rad2deg(roll))
            pitch = np.rad2deg(pitch)#self._normalize_angle_deg(np.rad2deg(pitch))
            yaw = np.rad2deg(yaw)#self._normalize_angle_deg(np.rad2deg(yaw))
           
            
            ok = True
            # print(roll,pitch,yaw)
            if roll_limits_deg is not None:
                r_min, r_max = roll_limits_deg
                if not (r_min <= roll <= r_max):
                    ok = False

            if ok and pitch_limits_deg is not None:
                p_min, p_max = pitch_limits_deg
                if not (p_min <= pitch <= p_max):
                    ok = False

            if ok and yaw_limits_deg is not None:
                y_min, y_max = yaw_limits_deg
                if not (y_min <= yaw <= y_max):
                    ok = False

            if ok:
                print(roll,pitch,yaw)
                keep_idx.append(i)

        if len(keep_idx) == 0:
            return None, []

        # Slice the GraspGroup (with fallback if slicing isn't supported)
        try:
            gg_f = gg[keep_idx]
            if isinstance(gg_f, GraspGroup):
                return gg_f, keep_idx
            raise TypeError
        except Exception:
            gg_new = GraspGroup()
            for i in keep_idx:
                gg_new.add(gg[i])
            return gg_new, keep_idx

        
    def choose_grasp(
        self,
        gg: "GraspGroup",
        boxes: Optional[List[Dict]],
        
        *,
        criterion: str = "both",  # "distance" | "score" |  "width" | "both"
        alpha: List[float] = [0.5,0.5,0.5],  
    ) -> Tuple[object, Dict]:
        """
        Choose a single grasp.
        - If boxes are given:
            Use per-box selection according to 'criterion'.
            Return the first available per-box winner in box order.
        - Otherwise:
            Return top-scoring grasp overall if criterion != "distance",
            else keep previous behavior (gg[0]).
        Returns (chosen_grasp, info_dict).
        """
        if boxes:
            best_idxs, stats = utils.best_grasp_indices_per_box(
                gg, self.fx, self.fy, self.cx, self.cy,
                boxes, return_stats=True, criterion=criterion, alpha=alpha
            )
            idx_for_motion = next((i for i in best_idxs if i is not None), None)
            if idx_for_motion is None:
                raise GraspComputationError("No valid grasps found inside any box.")
            return gg[idx_for_motion], {
                "best_idxs": best_idxs,
                "stats": stats,
                "criterion": criterion,
                "alpha": alpha if criterion == "both" else None,
            }

        # No boxes: decide globally
        if criterion == "distance":
            # Preserve legacy behavior: gg is assumed pre-sorted by distance/quality
            return gg[0], {"best_idxs": [0], "stats": [], "criterion": criterion}
        
        elif criterion == "score":
            # Pick global max score
            scores = [utils._get_grasp_score(g) for g in gg]
            best_i = max(range(len(gg)), key=lambda i: scores[i]) if gg else 0
            return gg[best_i], {
                "best_idxs": [best_i],
                "stats": [{"chosen_score": float(scores[best_i]) if gg else None}],
                "criterion": criterion,

            }
        
        elif criterion == "width":
            # Pick global max mwidt
            print('gedt width')
            width = [utils._get_grasp_width(g) for g in gg]
            print(width)
            best_i = min(range(len(gg)), key=lambda i: width[i]) if gg else 0
            return gg[best_i], {
                "best_idxs": [best_i],
                "stats": [{"chosen_score": float(width[best_i]) if gg else None}],
                "criterion": criterion,
            }
        else:
            
            return None

    def compute_grasp_targets(
        self,
        chosen_grasp: object,
        T_base_tcp: np.ndarray,
        T_tcp_cam: np.ndarray,
        T_tcp_gripper: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Convert a chosen grasp (in camera frame) into pre/contact/retreat poses for the *active TCP*.
        Also resolves the 180° rotational symmetries to stay close to the current TCP rotation.
        """
        # Build grasp 4x4 in camera frame; apply local +90° about Y if your gripper requires it
        grasp_T_cam = utils.grasp_T_from_grasp_obj(chosen_grasp)
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
        # grasp_T_cam = grasp_T_cam @ R_y90 @ R_z90

        T_base_cam = T_base_tcp @ T_tcp_cam

        # Symmetry resolution: find orientation closest to current TCP
        Rz_pi = utils.rot_axis_angle([0, 0, 1], math.pi)
        Rx_pi = utils.rot_axis_angle([1, 0, 0], math.pi)

        def apply_local_R(T_cam: np.ndarray, R_local: np.ndarray) -> np.ndarray:
            T2 = T_cam.copy()
            T2[:3, :3] = T2[:3, :3] @ R_local
            return T2

        T_gripper_robotTCP = np.linalg.inv(T_tcp_gripper)  # gripper→robotTCP
        best = None
        for Rloc in (np.eye(3), Rz_pi, Rx_pi, Rx_pi @ Rz_pi):
            cand_cam = apply_local_R(grasp_T_cam, Rloc)
            T_base_gripper_contact_cand = T_base_cam @ cand_cam
            T_base_robot_contact_cand = T_base_gripper_contact_cand @ T_gripper_robotTCP
            R_contact = T_base_robot_contact_cand[:3, :3]
            ang = utils.so3_distance(R_contact, T_base_tcp[:3, :3])  # closeness to current TCP
            if (best is None) or (ang < best[0]):
                best = (ang, cand_cam)

        grasp_T_cam_opt = best[1]
        symmetry_delta_deg = best[0] * 180.0 / math.pi

        # Build pre/contact/retreat active-TCP targets
        (pre_p, contact_p, retreat_p,
        T_base_robot_pre, T_base_robot_contact, T_base_robot_retreat,
        T_base_gripper_contact) = utils.robot_tcp_targets_from_grasp(
            T_base_cam,
            T_tcp_gripper,
            grasp_T_cam_opt,
            self.approach_offset_m,
            self.retreat_m
        )

        if self.debug:
            self.visualise3dgrasp(T_base_cam,T_base_tcp,T_base_robot_pre,T_base_robot_contact,T_base_gripper_contact)

        return {
            "pre_p": pre_p,
            "contact_p": contact_p,
            "retreat_p": retreat_p,
            "T_base_robot_pre": T_base_robot_pre,
            "T_base_robot_contact": T_base_robot_contact,
            "T_base_robot_retreat": T_base_robot_retreat,
            "T_base_gripper_contact": T_base_gripper_contact,
            "T_base_cam": T_base_cam,
            "grasp_T_cam": grasp_T_cam_opt,
            "symmetry_delta_deg": symmetry_delta_deg,
        }


    # ---------- Depth/Color to base-frame cloud + voxel ----------
    def depth_color_to_cloud_in_base(
        self,
        depth: np.ndarray,
        color: np.ndarray,
        T_base_cam: np.ndarray,
        max_depth_mm: float = 2.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert one RGB-D frame into a base-frame point cloud.
        Returns (pts_base Nx3 float32, cols Nx3 float32 in [0,1]).
        """
        assert depth.ndim == 2 and color.ndim == 3
        H, W = depth.shape
        xmap, ymap = np.meshgrid(np.arange(W), np.arange(H))

        depths = depth.astype(np.float32)
        colors = (color.astype(np.float32) / 255.0).reshape(-1, 3)
      
        
        points_z = depths * self.scale
        points_x = (xmap - self.cx) / self.fx * points_z
        points_y = (ymap - self.cy) / self.fy * points_z

        pts_cam = np.stack([points_x, points_y, points_z], axis=-1).reshape(-1, 3)
       

        # Valid Z
        valid = pts_cam[:, 2] < max_depth_mm
        pts_cam = pts_cam[valid]
        cols = colors[valid]

        # Cam->Base
        pts_cam_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1), dtype=pts_cam.dtype)], axis=1)
        pts_base_h = (T_base_cam @ pts_cam_h.T).T
        pts_base = pts_base_h[:, :3].astype(np.float32)

        return pts_base, cols.astype(np.float32)

