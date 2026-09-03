#!/usr/bin/env python3


from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
import time
import numpy as np
from PIL import Image
import traceback
import cv2
# Local modules
from commons.logger import LoggedFrame, Scanner
# from commons.logger_old import exploratory_scan_and_log
import commons.grasp_utils as utils
from detectors.sam3_object_detection import Sam3Detector

import anygrasp.grasping as grasping
from commons.grasp_utils import check_path_exists
import yaml

# Types shared from initializer
from hardware.hardware_init import HardwareHandles,HardwareInitializer

CFG_PATH = 'config/grasp.yaml'
# ---------- Exceptions ----------
class PipelineStopped(RuntimeError):
    """Raised internally when a stop() request has been received."""


# ---------- Helpers ----------
class DetectThread(threading.Thread):
    def __init__(self, detector ,image: str,query_list: List[str], conf_list: List[float],debug: bool):
        super().__init__(daemon=True)
        self.image = image
        self.query_list = query_list
        self.conf_list = conf_list
        self.debug = debug
        self.object_bboxes: Optional[List[Dict[str, Any]]] = None
        self.error: Optional[BaseException] = None
        self.detector = detector

    def run(self):
        try:
            bbox = self.detector.detect_bbox(
                    color = self.image, prompts = self.query_list, conf_list=self.conf_list, debug=self.debug,
            )
            self.object_bboxes = utils.select_top_conf_box(bbox)
        
        except Exception as e:  # bubble up via field
            self.error = e
            self.object_bboxes = None
            self.object_bboxes_debug_image = None


@dataclass
class PipelineResult:
    boxes: Optional[List[Dict[str, Any]]] = None
    grasps_all: Optional[Any] = None
    grasps_filtered: Optional[Any] = None
    chosen_grasp: Optional[Any] = None
    cloud: Optional[Any] = None
    aborted: Optional[str] = None
    stopped: bool = False


class QbotGrasp:
    """
    Active-scan grasp pipeline as a reusable, stoppable class that *assumes*
    all hardware is already initialized and passed in via HardwareHandles.

    Public API:
        run(det_query: str, blocking: bool = True) -> dict
        stop()
        is_running() -> bool
        wait(timeout: Optional[float])
        shutdown()
        set_on_done(handler)
    """

    def __init__(
        self,
        hw: HardwareHandles,
        on_progress: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        on_done: Optional[Callable[[str, str, Dict[str, Any]], None]] = None,
        interactive = False,
        move_to_start: bool = False,
        ignore_scanning: bool = False,
        detector: Optional[Sam3Detector] = None,
        debug: bool = False,
    ):
        self.hw = hw
        self.interactive = interactive
       # config
        cfg = {}
        cfg_file = check_path_exists(CFG_PATH,__file__)
        if cfg_file:
            with cfg_file.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        else:
            print('ERROR no cfg file', cfg_file)
            raise SystemExit(1)
        self.cfg = cfg
        self.bbox_cfg = cfg['bbox']
        print('setup anygrasp')
        self.grasping_engine = grasping.AnyGraspEngine(
            camera_intrinsics=hw.cam_arm_cfg.get("intrinsics"),
            debug=debug
        )
        print('done')
        self.debug = debug
        self.move_to_start = move_to_start
        if not ignore_scanning:
            self.ignore_scanning = self.cfg['ignore_scanning']
        else:
            self.ignore_scanning = True

        self.margins = self.bbox_cfg['margins_px']
            
        self.start_location = self.cfg['start_location']
        if detector is None:
            self.detector =  Sam3Detector(device="cuda")
        else:
            self.detector = detector
        # Shorthands so the original logic stays readable
        
        # Threading / lifecycle
        self._stop_event = threading.Event()
        self._run_impl_thread: Optional[threading.Thread] = None

        # Results & callbacks
        self.last_result: Dict[str, Any] = {}
        self.on_progress = on_progress
        self.on_done = on_done

    # ----- Lifecycle helpers -----
    def _emit(self, event: str, **info):
        if self.on_progress:
            try:
                self.on_progress(event, info)
            except Exception:
                pass  # progress callbacks must never break the pipeline

    def _check_stop(self):
        if self._stop_event.is_set():
            raise PipelineStopped("Stop requested")

    # --- public convenience setter
    def set_on_done(self, handler: Callable[[str, str, Dict[str, Any]], None]) -> None:
        """handler(status: 'complete'|'stopped'|'error', details: str, result: dict)"""
        self.on_done = handler

    # --- internal notifier
    def _notify_done(self, status: str, details: str, result: Dict[str, Any]):
        if self.on_done:
            try:
                self.on_done(status, details, result)
            except Exception:
                pass

    def grab_object(self, query:str,grasp, handles, detector, conf:float=0.2 ,rotate_grasp:bool=True,debug:bool=False):
        det_query_list = [query]
        conf_list = [conf]
        depth, color = handles.cam_arm.get_rgbd()
        object_bbox = detector.detect_bbox(prompts= det_query_list,color=color,conf_list=conf_list,debug=debug)    
        grasp.run(det_query_list=det_query_list, rotate_grasp=rotate_grasp, blocking=True, color_image=color, depth_image=depth, object_bboxes=object_bbox)
            

    # ----- Public API -----
    def run(self,det_query_list: List[str],conf_list: List[float], blocking: bool = True, rotate_grasp=True,color_image=None, depth_image=None, object_bboxes=None) -> Dict[str, Any]:
        """Run the whole pipeline. If blocking=False, returns immediately and
        the result will be stored in `self.last_result` after completion.
        """
        if self._run_impl_thread and self._run_impl_thread.is_alive():
            raise RuntimeError("Pipeline already running")

        # Reset stop flag and results
        self._stop_event.clear()
        self.last_result = {}

        if blocking:
            return self._run_impl(det_query_list,conf_list=conf_list,rotate_grasp=rotate_grasp, color_image = color_image,  depth_image = depth_image, object_bboxes = object_bboxes)
        else:
            self._run_impl_thread = threading.Thread(
                target=self._run_impl, args=(det_query_list,conf_list,rotate_grasp,color_image, depth_image, object_bboxes), daemon=True
            )
            self._run_impl_thread.start()
            return {}

    def stop(self):
        """Request the pipeline to stop and attempt to halt hardware immediately."""
        self._stop_event.set()
        try:
            if self.hw.arm is not None:
                self.hw.arm.stop()  # should interrupt current motion
        except Exception:
            pass
        try:
            if self.hw.cam_arm is not None:
                self.hw.cam_arm.stop()
            if self.hw.cam_fixed is not None:
                self.hw.cam_fixed.stop()
        except Exception:
            pass

    def is_running(self) -> bool:
        return self._run_impl_thread is not None and self._run_impl_thread.is_alive()

    def wait(self, timeout: Optional[float] = None):
        if self._run_impl_thread:
            self._run_impl_thread.join(timeout=timeout)

    def shutdown(self):
        """Best-effort cleanup for devices (safe even if already stopped)."""
        try:
            if self.hw.cam_arm is not None:
                self.hw.cam_arm.stop()
            if self.hw.cam_fixed is not None:
                self.hw.cam_fixed.stop()
        except Exception:
            pass
        try:
            if self.hw.arm is not None:
                self.hw.arm.stop()
        except Exception:
            pass



    def generate_grasps(self,depth_image,color_image,rotate_tf = None,scan=True):
            # Active scan while detection runs
            logged_frames: List[LoggedFrame] = []
            pc_cfg = self.cfg['pointcloud']
            voxel_downsample_size = pc_cfg.get("voxel_downsample_size", 0.005)
            outlier_removal_std = pc_cfg.get("outlier_removal_std", 0.1)

            if self.hw.arm is not None and not self.hw.ignore_arm and not self.ignore_scanning and scan:
                scan_cfg = self.cfg['scanning']
                pitch_min_deg = scan_cfg.get("pitch_min_deg", -20)
                pitch_max_deg = scan_cfg.get("pitch_max_deg", 20)
                min_offset_m = scan_cfg.get("min_offset_m", -0.2)
                max_offset_m = scan_cfg.get("max_offset_m", 0.1)
                log_interval_s = float(scan_cfg.get("log_interval_s", 0.1))
                scan_type = scan_cfg.get("scan_type", "line")
                axis = scan_cfg.get("axis", "y")

                self._emit(
                    "scan_start"
                )

                scanner = Scanner(self.hw.arm, self.hw.cam_arm, self.hw.T_tcp_cam, debug=self.debug)
            
                if scan_type == 'line':
                    logged_frames, T_start = scanner.scan_line(
                        min_offset_m=min_offset_m,   # left
                        max_offset_m=max_offset_m,   # right
                        n_points=30,
                        axis=axis,
                        speed=0.1,
                    )
                if scan_type == 'hemisphere':
                    logged_frames, T_base_tcp_start =scanner.scan_hemisphere(
                        pitch_min_deg=pitch_min_deg,
                        pitch_max_deg=pitch_max_deg,
                        yaw_min_deg=-1.0,
                        yaw_max_deg=1.0,
                        n_pitch=10,
                        n_yaw=4,
                        speed=.2,
                        log_interval_s=log_interval_s,
                    )
                if scan_type == 'xy_arch':
                    logged_frames, T_start = scanner.scan_arch_xy(
                        # radius_m = 0.40,
                        # angle_min_deg = -45.0,
                        # angle_max_deg = 45.0,

                        radius_m = max_offset_m,
                        angle_min_deg = pitch_min_deg,
                        angle_max_deg = pitch_max_deg,

                        n_points =25,
                        speed = 0.10,
                        accel = 1.00,
                    )
                self._emit("scan_done", frames=len(logged_frames))
            else:
                print('ignoring scanning')
                # Fallback: single frame with current/identity pose
                if self.hw.arm is not None:
                    T_base_tcp_now = self.hw.arm.get_T_base_tcp()
                else:
                    T_base_tcp_now = np.eye(4)
                T_base_cam_now = T_base_tcp_now @ self.hw.T_tcp_cam
                logged_frames = [
                    LoggedFrame(depth=depth_image, color=color_image, T_base_cam=T_base_cam_now)
                ]
                self._emit("scan_skipped", frames=1)

            self._check_stop()
            self._emit("cloud_build_start")

            # Build fused cloud in base frame
            all_pts_base = []
            all_cols = []
            for fr in logged_frames:
                pts_b, cols = self.grasping_engine.depth_color_to_cloud_in_base(fr.depth, fr.color, fr.T_base_cam,max_depth_mm=700)
                all_pts_base.append(pts_b)
                all_cols.append(cols)

            if len(all_pts_base) == 0:
                self._emit("cloud_empty")
                return {}

            pts_base = np.concatenate(all_pts_base, axis=0)
            cols = np.concatenate(all_cols, axis=0)

            #plot final pointcloud
            # cloud = self.grasping_engine.to_o3d_pointcloud(pts_base,cols)
            # self.grasping_engine.visualise3dpointcloud(cloud)

    
            # Transform to reference camera frame (first logged frame)
            T_base_cam_ref = logged_frames[0].T_base_cam
            T_cam_ref_base = np.linalg.inv(T_base_cam_ref)
            pts_base_h = np.hstack([pts_base, np.ones((pts_base.shape[0], 1), dtype=pts_base.dtype)])
            pts_cam = (T_cam_ref_base @ pts_base_h.T).T[:, :3]
            self._emit("pts_cam", points=pts_cam.shape[0])

            # Workspace filtering
            pts_cam, cols, _ = self.grasping_engine.build_workspace_from_cloud(pts_cam, cols)
            self._emit("workspace_filtered", points=pts_cam.shape[0])

            # Voxel downsample
            voxel_size = self.grasping_engine.flow.get("voxel", {}).get("size_m", voxel_downsample_size)
            pts_cam, cols = utils.voxel_downsample(pts_cam, cols, voxel_size)
            self._emit("cloud_voxelized", points=pts_cam.shape[0], voxel_size=voxel_size)
            pts_cam, cols = utils.statistical_outlier_removal_kdtree(pts_cam, cols=cols,k=16, std_ratio=outlier_removal_std)
            self._emit("statistical_outlier_removal", points=pts_cam.shape[0], std_ratio=outlier_removal_std)
      

            if rotate_tf is not None:
                pts_cam_h = np.hstack([pts_cam, np.ones((pts_cam.shape[0], 1), dtype=pts_cam.dtype)])
                pts_cam = (rotate_tf @ pts_cam_h.T).T[:, :3]

            # Generate grasps
            gg, cloud = self.grasping_engine.generate_grasps(pts_cam, cols,debug=self.debug)
            if gg:
                self._emit("grasps_generated", count=len(gg))

            return gg,cloud

    def detect_generic(
        self,
        prompt: str,
        color: np.ndarray,
        conf: float = 0.2,
        debug: bool = False
    ) -> List[Dict[str, Any]]:
        
        if debug:     
            cv2.imshow("SAM3 Raw", color)
            cv2.waitKey(0)
            
        masks = self.detector.segment(
            color_bgr=color,
            text_prompt=prompt,
            confidence_threshold=conf,
            category=prompt,
        )
        
        if debug and masks:
            output_dir = self.config.get('debug', 'output_dir', default='data/image_samples')
            output_file = self.config.get('debug', 'images', 'generic_detect', default='generic_detect.png')
            colors = self.config.get('debug', 'colors', default={})
   
            vis = draw_mask_debug(
                color, masks,
                output_path=f"{output_dir}/{output_file}",
                category_colors={prompt: tuple(colors.get(prompt, [0, 0, 255]))}
            )
            cv2.imshow("SAM3 Debug Visualization", vis)
            cv2.waitKey(0)

        return masks
    
    def rotate_cloud(self,cloud,rotate_tf ):
        cloud_h = np.hstack([cloud, np.ones((cloud.shape[0], 1), dtype=cloud.dtype)])
        cloud = (rotate_tf @ cloud_h.T).T[:, :3]
        return cloud

    def generate_targets(self,depth_image,color_image,scan=True,det_thread=None,rotate_grasp=False,object_bboxes=None, margins = 20):
            
        #ROTATE POINTCLOUD TO POINT DOWN, So it only uses top down grasps
        T_base_tcp = self.hw.arm.get_T_base_tcp()
        T_base_tcp_down = self.hw.arm.get_T_base_tcp_point_down()
        delta_tcp_to_down = np.linalg.inv(T_base_tcp)@T_base_tcp_down
        print('delta_tcp_to_down',delta_tcp_to_down)

        if rotate_grasp:
            gg,cloud = self.generate_grasps(depth_image,color_image,rotate_tf=delta_tcp_to_down,scan=scan)
            print('delta_tcp_to_down',delta_tcp_to_down)
        else:
            gg,cloud = self.generate_grasps(depth_image,color_image,scan=scan)        

        if not gg:
            self._notify_done(status="failed", details="no grasps detected", result=None)
            print("ERROR no grasps detected")
            return None

        self._emit("grasps_generated", count=len(gg))

        # Case A: detection thread path
        if object_bboxes is None and det_thread is not None:
            det_thread.join()
            if det_thread.error:
                self._emit("object_detection_failed", error=str(det_thread.error))
            else:
                object_bboxes = det_thread.object_bboxes
                if object_bboxes is None:
                    print('no object found!')
                    self._emit("no object found ")
                    self._notify_done(status="failed", details="can not find the object", result=None)
                    return None        
                if self.debug:
                    print('object_bboxes',object_bboxes)
                    debug_image = color_image.copy()
                    for b in object_bboxes:
                        x1, y1, w, h = b["xywh"]
                        x1 -= self.margins
                        y1 -= self.margins
                        x2 = x1 + w + self.margins*2
                        y2 = y1 + h + self.margins*2
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
                    cv2.imshow('object_bboxes_debug_image',debug_image)
                    cv2.waitKey(0)
                    

        print('total ggs',len(gg))

        # Rotate the Grasp Back
        if rotate_grasp:
            max_angle = 15.0
            gg = utils.filter_tilted_grasps(gg,max_angle)
            print('filter_rotate ggs',len(gg))

            #rotate grasp objects back to original angle
            gg_f = utils.rotate_grasp_objects(gg,np.linalg.inv(delta_tcp_to_down))
        else:
            gg_f = gg
        
        if object_bboxes is not None:
            gg_f, keep_idx = self.grasping_engine.filter_grasps_to_boxes(gg_f, object_bboxes,margins=margins)

        if not gg_f:
            self._emit("no grasp availible")
            self._notify_done(status="failed", details="can see object but cannot find a way to grab this object", result=None)
            return None

        print('filter_grasps_to_boxes ggs',len(gg_f))
        
      
        res = self.grasping_engine.check_reachability_for_grasps(
            gg=gg_f,
            arm=self.hw.arm, 
            T_base_tcp=T_base_tcp,
            T_tcp_cam=self.hw.T_tcp_cam,
            T_tcp_gripper=self.hw.T_tcp_to_gripper.copy(),
            poses=("pre", "contact"),
            require_all=True,          # both must be reachable
            filter_to_reachable=True,
            debug=True,
        )


        gg_f = res["gg_reachable"]   # <-- UPDATED HERE



         #revisualize new grasp
        if rotate_grasp:
            self.grasping_engine.setgg(gg_f,delta=delta_tcp_to_down,correct_grasp_rotation=True) 
        else:
            self.grasping_engine.setgg(gg_f)

        self._emit("grasps_filtered", before=len(gg), after=len(gg_f))

        # Choose best grasp
        chosen_grasp, choose_info = self.grasping_engine.choose_grasp(gg_f, object_bboxes,criterion='both',alpha = [0.5,0.0,0.5])
        grasp_width = getattr(chosen_grasp, "width") 
        open_adjustment_width = 0.03
        chosen_grasp_width = grasp_width + open_adjustment_width
        self._emit("grasp_chosen", score=getattr(chosen_grasp, "score", None))
        min_width = 0.02
        max_width = 0.14
        max_stroke = 0.0425
        stroke_adjustment = max_stroke*((grasp_width-max_width)/(min_width-max_width))
        print('stroke_adjustment',stroke_adjustment,'max_width',max_width,'grasp_width',grasp_width)
        adjusted_gripper = self.hw.T_tcp_to_gripper.copy()
        adjusted_gripper[2][3] += stroke_adjustment
        print('adjusted_gripper',adjusted_gripper)

        #Adjust approch with grasp width


        # if self.debug:
            # self.grasping_engine.visualize_grasp_box_filter(color_image, gg_f, object_bboxes)

        if self.hw.gripper is not None and not self.hw.ignore_gripper:
            self.hw.gripper.go_to_position_metres(chosen_grasp_width,block=True)                


        # Motion execution
        if self.hw.arm is not None and not self.hw.ignore_arm:
            # self._check_stop()

            roll, pitch, yaw = utils.rpy_from_R_zyx(T_base_tcp[:3, :3])
            self._emit("tcp_pose_read", roll=roll, pitch=pitch, yaw=yaw)
    
            targets = self.grasping_engine.compute_grasp_targets(
                chosen_grasp=chosen_grasp,
                T_base_tcp=T_base_tcp,
                T_tcp_cam=self.hw.T_tcp_cam,
                T_tcp_gripper=adjusted_gripper
            )
            return targets
        
        return None

    # ----- Core implementation -----
    def _run_impl(self,det_query_list: List[str],conf_list: List[float], rotate_grasp=False,color_image=None, depth_image=None, object_bboxes=None) -> Dict[str, Any]:
 
        try:
            # Move to observation
            if self.hw.arm is not None and not self.hw.ignore_arm and self.move_to_start:
                self.hw.arm.moveJ(self.start_location)
                self.hw.gripper.open(True)#open gripper
                self._emit("move_to_observation", joints=self.start_location)
                time.sleep(2.0)
            self._check_stop()
            self._emit("frame_acquire", stage="initial")
            
            if color_image is None and depth_image is None:
                print('getting rgbd from arm cam')
                depth_image, color_image = self.hw.cam_arm.get_rgbd()  # depth uint16, color RGB uint8
                self._check_stop()
            print('image acquired, starting detection')
            # if self.debug:
            #     cv2.imshow("image raw", color_image)
            #     cv2.waitKey(0)

            # Start detection thread if no boxes provided
            if object_bboxes is None:
                det_thread = DetectThread(self.detector, color_image, det_query_list, conf_list,self.debug)
                det_thread.start()
                self._emit("object_detection_started", query=det_query_list)
            else:
                det_thread = None

            #generate targets
            fail = False
            scan = True
            if self.ignore_scanning:
                scan = False

            targets = self.generate_targets(depth_image,color_image,scan=scan,det_thread=det_thread,rotate_grasp=rotate_grasp,object_bboxes=object_bboxes,margins=self.margins)
            if not targets:
                fail = True

            # Motion execution
            if self.hw.arm is not None and not self.hw.ignore_arm and not fail:
                self._check_stop()

                pre_p = targets["pre_p"]
                contact_p = targets["contact_p"]
                retreat_p = targets["retreat_p"]

                # Motion params
                move_speed = getattr(self.grasping_engine, "move_speed", 0.25)
                move_accel = getattr(self.grasping_engine, "move_accel", 0.8)
                descend_speed = getattr(self.grasping_engine, "descend_speed", 0.08)
                descend_accel = getattr(self.grasping_engine, "descend_accel", 0.3)

                def _confirm(step: str) -> bool:
                    if not self.interactive:
                        return True
                    try:
                        ans = input(f"Execute {step}? [y/N]: ")
                        return ans.strip().lower() == "y"
                    except EOFError:
                        return False

                # Approach to pre-grasp
                self._check_stop()
                if _confirm("approach to pre-grasp"):
                    self.hw.arm.moveL(pose=pre_p, speed=move_speed, accel=move_accel)
                    self._emit("moved_pre")
                else:
                    self._emit("aborted", stage="pre")
                    return {"aborted": "pre"}
                
                ###REFINE_GRASP HERE#####  
                # depth_image, color_image = self.hw.cam_arm.get_rgbd()  # depth uint16, color RGB uint8
                # new_targets = self.generate_targets(depth_image,color_image,scan=True,det_thread=None,filter_rotate=True)
                # self.hw.arm.moveL(pose=new_targets["pre_p"], speed=move_speed, accel=move_accel)
                # contact_p = new_targets["contact_p"]

                # Grasp to contact

                self._check_stop()
                if _confirm("grasp to contact"):
                    self.hw.arm.moveL(contact_p, descend_speed, descend_accel)
                    self._emit("moved_contact")
                else:
                    self._emit("aborted", stage="contact")
                    return {"aborted": "contact"}

                # Close gripper
                if self.hw.gripper is not None and not self.hw.ignore_gripper:
                    self._check_stop()
                    if _confirm("close gripper"):
                        self.hw.gripper.close(block=True)
                        self._emit("gripper_closed")
                        time.sleep(1)
                        if not self.hw.gripper.is_holding_object():
                            #try again
                            self.hw.gripper.open(block=True)
                            self.hw.gripper.close(block=True)
                            time.sleep(1)
                            if not self.hw.gripper.is_holding_object():
                                self._notify_done(status="failed", details="the grab was unsuccessful", result=None)
                                fail = True
                        # if self.hw.gripper.is_completely_closed():
                        #     self._notify_done(status="failed", details="grab was unsuccessful", result=None)

                    else:
                        self._emit("aborted", stage="gripper_close")
                        return {"aborted": "gripper_close"}
                else:
                    self._emit("no_gripper")
                    fail = True

                # Retreat
                self._check_stop()
                if _confirm("retreat"):
                    self.hw.arm.moveL(retreat_p, move_speed, move_accel)
                    self._emit("moved_retreat")
                    if not self.hw.gripper.is_holding_object():
                        fail = True
                        self._notify_done(status="failed", details="grab was unsuccessful", result=None)
                else:
                    self._emit("aborted", stage="retreat")
                    return {"aborted": "retreat"}

                self._emit("done")

            # Package results
            self.last_result = targets
            if not fail:
                self._notify_done(status="complete", details="pick up was successful", result=self.last_result)
    
            self.last_result

        except PipelineStopped:
            self._emit("stopped")
            try:
                if self.hw.arm is not None:
                    self.hw.arm.stop()
            except Exception:
                pass
            res = {"stopped": True}
            # self._notify_done(status="stopped", details="pipeline interrupted", result=res)
            return res

        except Exception as e:
            self._emit("error", message=str(e))
            # self._notify_done(status="error", details=str(e), result={"traceback": traceback.format_exc()})
            raise

        finally:
            if self._stop_event.is_set():
                # Best-effort cleanup if stop was requested mid-run
                try:
                    if self.hw.cam_arm is not None:
                        self.hw.cam_arm.stop()
                    if self.hw.cam_fixed is not None:
                        self.hw.cam_fixed.stop()
                except Exception:
                    pass
                try:
                    if self.hw.arm is not None:
                        self.hw.arm.stop()
                except Exception:
                    pass

def _print_progress(event: str, info: Dict[str, Any]) -> None:
    if info:
        print(f"[progress] {event}: {info}")
    else:
        print(f"[progress] {event}")


def _on_done(status: str, details: str, result: Dict[str, Any]) -> None:
    print(f"[done] status={status} details={details} result_keys={list(result.keys())}")


def _run_once(det_query_list: List[str] ,conf_list: List[float] ,  **init_kwargs):
    initializer = HardwareInitializer(**init_kwargs)
    hw = initializer.initialize()

    grasp = QbotGrasp(
        hw=hw,
        on_progress=_print_progress,
        on_done=_on_done,
        debug=init_kwargs.get("debug", False),
    )


    try:
        grasp.run(det_query_list,conf_list=conf_list, filter_rotation=False,blocking=True)
    finally:
        grasp.shutdown()
        initializer.shutdown()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a single Qbot grasp attempt")
    parser.add_argument("--query", nargs="+", help="detection query (can be multiple items: cup mug block)")
    parser.add_argument("--ignore-arm", action="store_true", dest="ignore_arm")
    parser.add_argument("--ignore-gripper", action="store_true", dest="ignore_gripper")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    _run_once(
        det_query_list=args.query,
        conf_list = None,
        ignore_arm=args.ignore_arm,
        ignore_gripper=args.ignore_gripper,
        debug=args.debug,
    )
