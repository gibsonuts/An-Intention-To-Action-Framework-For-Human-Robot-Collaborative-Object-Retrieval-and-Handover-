import os
import time
import math
import yaml
import argparse
import numpy as np
import pyrealsense2 as rs

import open3d as o3d
import torch

from gsnet import AnyGrasp
from graspnetAPI import GraspGroup

# UR RTDE
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive

from PIL import Image, ImageDraw, ImageFont

# -------------------- Utils --------------------
def axis_angle_from_rot(R):
    # Converts 3x3 rotation matrix to UR axis-angle (rx, ry, rz)
    theta = math.acos(max(min((np.trace(R) - 1) / 2.0, 1.0), -1.0))
    if abs(theta) < 1e-9:
        return np.zeros(3)
    rx = (R[2,1] - R[1,2]) / (2*math.sin(theta))
    ry = (R[0,2] - R[2,0]) / (2*math.sin(theta))
    rz = (R[1,0] - R[0,1]) / (2*math.sin(theta))
    axis = np.array([rx, ry, rz])
    return axis * theta

def pose_from_T(T):
    R = T[:3,:3]
    p = T[:3,3]
    rxyz = axis_angle_from_rot(R)
    return [float(p[0]), float(p[1]), float(p[2]), float(rxyz[0]), float(rxyz[1]), float(rxyz[2])]

def make_T(R, t):
    T = np.eye(4)
    T[:3,:3] = R
    T[:3, 3] = t
    return T

def grasp_T_from_grasp_obj(g):
    """
    Build a 4x4 grasp pose (camera frame) from a Grasp object that prints
    'rotation:' and 'translation:' fields.
    """
    R = None
    t = None
    for key in ("rotation", "R", "rot", "rotation_matrix"):
        if hasattr(g, key):
            R = np.array(getattr(g, key), dtype=float)
            break
    if R is None and hasattr(g, "rotationMatrix"):
        R = np.array(g.rotationMatrix(), dtype=float)

    for key in ("translation", "t", "center", "translation_vector"):
        if hasattr(g, key):
            t = np.array(getattr(g, key), dtype=float).reshape(3)
            break
    if t is None and hasattr(g, "translationVector"):
        t = np.array(g.translationVector(), dtype=float).reshape(3)

    if R is None or t is None:
        raise ValueError("Could not find rotation/translation on the Grasp object")

    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3,  3] = t
    return T

# Grasp frame convention (AnyGrasp):
# Assume gripper closes along +X of grasp frame, approach is -Z

def robot_tcp_targets_from_grasp(T_base_cam, T_tcp_gripper, grasp_T_cam, approach_offset_m, retreat_m):
    """
    Produce robot Active-TCP targets (pre, contact, retreat) from a grasp in the camera frame.

    Inputs:
      - T_base_cam:    base→camera
      - T_tcp_gripper: robotTCP→gripperTCP (calibration)
      - grasp_T_cam:   camera→gripperTCP (from AnyGrasp; grasp pose)
      - approach_offset_m, retreat_m: distances along grasp -Z

    Returns:
      - pre, contact, retreat poses as [x,y,z,rx,ry,rz] for the *active robot TCP*
      - and their 4x4 matrices
      - plus T_base_gripper_contact for visualization
    """
    # 1) Compute contact frame for the *gripper TCP* in base
    T_base_gripper_contact = T_base_cam @ grasp_T_cam

    # 2) Pre/retreat in the gripper frame, along grasp -Z
    z_axis = -T_base_gripper_contact[:3, 2]  # approach along -Z of grasp
    T_base_gripper_pre = T_base_gripper_contact.copy()
    T_base_gripper_pre[:3, 3] = T_base_gripper_pre[:3, 3] + z_axis * approach_offset_m

    T_base_gripper_retreat = T_base_gripper_contact.copy()
    T_base_gripper_retreat[:3, 3] = T_base_gripper_retreat[:3, 3] - z_axis * retreat_m

    # 3) Convert gripperTCP targets → active robot TCP targets
    T_gripper_robotTCP = np.linalg.inv(T_tcp_gripper)  # gripper→robotTCP
    T_base_robot_pre     = T_base_gripper_pre @ T_gripper_robotTCP
    T_base_robot_contact = T_base_gripper_contact @ T_gripper_robotTCP
    T_base_robot_retreat = T_base_gripper_retreat @ T_gripper_robotTCP

    return (
        pose_from_T(T_base_robot_pre),
        pose_from_T(T_base_robot_contact),
        pose_from_T(T_base_robot_retreat),
        T_base_robot_pre,
        T_base_robot_contact,
        T_base_robot_retreat,
        T_base_gripper_contact
    )

def text_pointcloud(text, T_world, size=0.02, offset=(0.03, 0.0, 0.0), thickness=1, font_size=28):
    """
    Make a small point cloud label for `text` placed near the origin of pose T_world.
    - size: overall text height (meters)
    - offset: where to place the label relative to T_world origin (meters, in that frame)
    - thickness: pixel dilation for legibility (int>=1)
    """
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    dummy_img = Image.new("L", (1, 1), 0)
    d = ImageDraw.Draw(dummy_img)
    w, h = d.textbbox((0,0), text, font=font)[2:]
    pad = 2
    img = Image.new("L", (w + 2*pad, h + 2*pad), 0)
    d = ImageDraw.Draw(img)
    d.text((pad, pad), text, 255, font=font)

    if thickness > 1:
        from PIL import ImageFilter
        img = img.filter(ImageFilter.MaxFilter(size=thickness))

    arr = np.array(img)
    ys, xs = np.where(arr > 0)
    if xs.size == 0:
        return o3d.geometry.PointCloud()

    H, W = arr.shape
    xs = (xs - W/2.0) / max(H, 1)
    ys = -(ys - H/2.0) / max(H, 1)
    zs = np.zeros_like(xs, dtype=np.float32)

    pts_local = np.stack([xs, ys, zs], axis=1).astype(np.float32) * size
    pts_local += np.array(offset, dtype=np.float32).reshape(1,3)

    pts_h = np.concatenate([pts_local, np.ones((pts_local.shape[0],1), dtype=np.float32)], axis=1)
    pts_world_h = (T_world @ pts_h.T).T
    pts_world = pts_world_h[:, :3]

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts_world)
    pc.colors = o3d.utility.Vector3dVector(np.tile(np.array([[0.05, 0.05, 0.05]]), (pts_world.shape[0],1)))
    return pc

# -------------------- Main pipeline --------------------
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='../calibration/calibration.yaml')
parser.add_argument('--checkpoint_path', required=True)
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

with open(args.config, 'r') as f:
    cfg = yaml.safe_load(f)

cam_cfg = cfg['camera']
frames = cfg['frames']
flow = cfg['workflow']
rob = cfg['robot']

# === NEW: TCP-based calibrations ===
T_tcp_cam = np.array(frames['T_tcp_camera'], dtype=float)      # robotTCP→camera
T_tcp_gripper = np.array(frames['T_tcp_gripper'], dtype=float) # robotTCP→gripperTCP

# RealSense setup
pipe = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, cam_cfg['intrinsics']['width'], cam_cfg['intrinsics']['height'], rs.format.z16, 30)
config.enable_stream(rs.stream.color, cam_cfg['intrinsics']['width'], cam_cfg['intrinsics']['height'], rs.format.bgr8, 30)
profile = pipe.start(config)

align = rs.align(rs.stream.color) if cam_cfg.get('align_depth_to_color', False) else None

# If you prefer to use live intrinsics, you can read them here
s = profile.get_stream(rs.stream.depth)
intr = s.as_video_stream_profile().get_intrinsics()
fx, fy, cx, cy = cam_cfg['intrinsics']['fx'], cam_cfg['intrinsics']['fy'], cam_cfg['intrinsics']['cx'], cam_cfg['intrinsics']['cy']
scale = cam_cfg['intrinsics']['depth_scale']

# AnyGrasp
class Dot: pass
cfgs = Dot()
cfgs.checkpoint_path = args.checkpoint_path
cfgs.max_gripper_width = flow['grasp']['max_gripper_width']
cfgs.gripper_height = flow['grasp']['gripper_height']
cfgs.top_down_grasp = flow['grasp']['top_down_only']
cfgs.debug = args.debug

anygrasp = AnyGrasp(cfgs)
anygrasp.load_net()

# UR setup
rtde_c = RTDEControl(rob['ur_ip'])
rtde_r = RTDEReceive(rob['ur_ip'])

# Optional: set payload and TCP in controller from YAML
if 'payload' in rob:
    p = rob['payload']
    rtde_c.setPayload(p['mass'], p['cog'])

if 'tcp_offset_rpy_xyz' in rob:
    rx, ry, rz, x, y, z = rob['tcp_offset_rpy_xyz']
    # This sets the *active robot TCP* in the controller.
    # Our transforms below always interpret getActualTCPPose() as base→activeTCP.
    rtde_c.setTcp([x, y, z, rx, ry, rz])

try:
    print("Waiting for a synchronized RGB-D frame...")
    for _ in range(60):  # grab a couple of seconds of frames for auto-exposure
        pipe.wait_for_frames()

    frameset = pipe.wait_for_frames()
    if align:
        frameset = align.process(frameset)

    depth = np.asanyarray(frameset.get_depth_frame().get_data())
    color = np.asanyarray(frameset.get_color_frame().get_data())[:, :, ::-1]  # BGR→RGB

    colors = color.astype(np.float32) / 255.0
    depths = depth.astype(np.float32)

    H, W = depths.shape
    xmap, ymap = np.meshgrid(np.arange(W), np.arange(H))

    points_z = depths * scale
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z

    # workspace cull in camera frame
    ws = flow['workspace_cam']
    mask = (
        (points_z > ws['zmin']) & (points_z < ws['zmax']) &
        ((points_x > ws['xmin']) & (points_x < ws['xmax'])) &
        ((points_y > ws['ymin']) & (points_y < ws['ymax']))
    )

    pts = np.stack([points_x, points_y, points_z], axis=-1)
    pts = pts[mask].astype(np.float32)
    cols = colors[mask].astype(np.float32)

    print(f"Using {len(pts)} points for grasp detection")

    gg, cloud = anygrasp.get_grasp(
        pts, cols,
        lims=[ws['xmin'], ws['xmax'], ws['ymin'], ws['ymax'], ws['zmin'], ws['zmax']],
        apply_object_mask=True, dense_grasp=False, collision_detection=True
    )

    if len(gg) == 0:
        raise RuntimeError("No grasp detected after collision detection")

    gg = gg.nms().sort_by_score()

    chosen = gg[0]
    print("Chosen grasp score:", chosen.score)

    # Build grasp 4x4 in camera frame
    grasp_T_cam = grasp_T_from_grasp_obj(chosen)

    # === Live active TCP pose from the robot (base→activeTCP) ===
    tcp_pose_base = rtde_r.getActualTCPPose()  # [x,y,z,rx,ry,rz]
    rx, ry, rz = tcp_pose_base[3:6]
    angle = math.sqrt(rx*rx + ry*ry + rz*rz) + 1e-12
    ax = np.array([rx, ry, rz]) / angle
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    R_tcp = np.eye(3) + math.sin(angle)*K + (1-math.cos(angle))*(K@K)
    T_base_tcp = make_T(R_tcp, np.array(tcp_pose_base[:3]))
    print('Live base→TCP:\n', T_base_tcp)

    # === NEW: base→camera from TCP-based hand–eye ===
    T_base_cam = T_base_tcp @ T_tcp_cam

    # Pregrasp/contact/retreat targets for the *active robot TCP*
    (pre_p, contact_p, retreat_p,
     T_base_robot_pre, T_base_robot_contact, T_base_robot_retreat,
     T_base_gripper_contact) = robot_tcp_targets_from_grasp(
        T_base_cam,
        T_tcp_gripper,
        grasp_T_cam,
        flow['approach']['offset_m'],
        flow['approach']['retreat_m']
    )

    # Motion params
    spd = flow['motion']['move_speed']
    acc = flow['motion']['move_accel']

    # ---- Visualization (no flange) ----
    if cfgs.debug:
        def frame_from_T(T, size=0.05):
            f = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
            f.transform(T)
            return f

        def line_along_dir(origin, direction, length=0.18):
            d = direction / (np.linalg.norm(direction) + 1e-12)
            pts = o3d.utility.Vector3dVector([origin, origin + d * length])
            lines = o3d.utility.Vector2iVector([[0, 1]])
            ls = o3d.geometry.LineSet(points=pts, lines=lines)
            ls.colors = o3d.utility.Vector3dVector([[0.2, 0.2, 0.2]])
            return ls

        def ensure_pointcloud(pc):
            if isinstance(pc, o3d.geometry.PointCloud):
                return pc
            pc_o3d = o3d.geometry.PointCloud()
            pc_o3d.points = o3d.utility.Vector3dVector(pc)
            return pc_o3d

        # 1) Put cloud and grasp visuals into BASE
        cloud_viz = ensure_pointcloud(cloud)
        cloud_viz.transform(T_base_cam)

        grippers = gg.to_open3d_geometry_list()
        for i, g in enumerate(grippers):
            g.transform(T_base_cam)      # grippers now in BASE
            if i == 0:
                g.paint_uniform_color([0, 1, 0])   # best (green)
            else:
                g.paint_uniform_color([1, 0, 0])   # others (red)

        # 2) Viewer flip for nicer screen coords (optional)
        trans_mat = np.array([[1, 0, 0, 0],
                              [0,-1, 0, 0],
                              [0, 0,-1, 0],
                              [0, 0, 0, 1]])
        cloud_viz.transform(trans_mat)
        for g in grippers:
            g.transform(trans_mat)

        # 3) Frames (all in BASE, then flipped for view)
        base_frame = frame_from_T(trans_mat @ np.eye(4), size=0.07)
        tcp_frame_live = frame_from_T(trans_mat @ T_base_tcp, size=0.06)
        cam_frame = frame_from_T(trans_mat @ (T_base_cam), size=0.09)
        grasp_contact_frame = frame_from_T(trans_mat @ T_base_gripper_contact, size=0.10)

        # Camera +Z axis for sanity
        cam_dir_base = T_base_cam[:3, 2]
        cam_origin   = T_base_cam[:3, 3]
        cam_axis_line = line_along_dir(
            (trans_mat @ np.r_[cam_origin, 1.0])[:3],
            (trans_mat[:3,:3] @ cam_dir_base),
            length=0.18
        )

        o3d.visualization.draw_geometries([
            cloud_viz, *grippers,
            base_frame, tcp_frame_live, cam_frame, grasp_contact_frame, cam_axis_line
        ])

    user_in = input("Execute this grasp? [y/N]: ")
    if user_in.lower() != 'y':
        print("Aborting grasp execution.")
        exit()

    print("Moving to pre-grasp (active TCP target):", pre_p)
    rtde_c.moveL(pre_p, spd, acc, True)

    print("Approach to contact:", contact_p)
    rtde_c.moveL(contact_p, flow['approach']['descend_speed'], flow['approach']['descend_accel'], True)

    # TODO: trigger gripper close command here (vendor-specific)
    # close_gripper()
    time.sleep(0.4)

    print("Retreat:", retreat_p)
    rtde_c.moveL(retreat_p, spd, acc, True)

    print("Done.")

finally:
    pipe.stop()
    try:
        rtde_c.stopScript()
    except Exception:
        pass
