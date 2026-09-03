import argparse
from pathlib import Path

import yaml

from hardware.hardware_init import HardwareInitializer
from qbot.qbot_cycles import GenericCycleManager
from detectors.sam3_subprocess import Sam3SubprocessDetector
import time

cycles_config_path = Path(__file__).resolve().parent / "qbot" / "config" / "cycles.yaml"
with cycles_config_path.open("r", encoding="utf-8") as f:
    cycle_config = yaml.safe_load(f) or {}

START_J = cycle_config.get("grasp_joint")
if START_J is None:
    raise ValueError("grasp_joint missing from qbot/config/cycles.yaml")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # Testing generic object detection
    parser.add_argument("--prompt", type=str, default = None, help="Prompt for generic object detection")
    parser.add_argument("--prompt_conf", type=float, default=0.2, help="Confidence for generic object detection")
    parser.add_argument("--prompt_z_offset", type=float, default=0.6, help="Z offset for generic object detection")
    parser.add_argument("--prompt_camera_type", type=str, default="arm", help="Camera type, 'fixed' or 'arm'")
    parser.add_argument("--prompt_orientation_alignment", type=str, default="long", help="Orientation to align to [short or long]")
    parser.add_argument("--pick_up", action="store_true", help="pcik up",)
    parser.add_argument("--test_image", type=str, default = None, help="Debug a image file for generic object detection")
    parser.add_argument("--sam3_env", type=str, default="quendabot_demo", help="Conda environment used for SAM3 inference")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug prints / visualizations where supported.",
    )
    return parser.parse_args()

def move_to_start_position(handles):
        handles.arm.moveJ(START_J)
         
def detect_and_grab_object(query:str,grasp, handles, detector, conf:float=0.2 ,rotate_grasp:bool=True,debug:bool=False):
    det_query_list = [query]
    conf_list = [conf]
    depth, color = handles.cam_arm.get_rgbd()
    object_bbox = detector.detect_bbox(prompts= det_query_list,color=color,conf_list=conf_list,debug=debug)    
    grasp.run(det_query_list=det_query_list, conf_list=conf_list, rotate_grasp=rotate_grasp, blocking=True, color_image=color, depth_image=depth, object_bboxes=object_bbox)

def done(status: str, details: str, result: dict):
     print(f"[on_done] status={status}")
     last_status = status

def main():
    last_status = None
    last_detail = None
    def on_progress(event: str, info: dict):
        print(f"[progress] {event}: {info}")
    
    def done(status: str, details: str, result: dict):
        print('done callback',status,details,result)
        nonlocal last_status, last_detail
        last_status = status
        last_detail = details

    args = parse_args()

    detector = Sam3SubprocessDetector(env_name=args.sam3_env)

    # Initialize hardware``
    hw = HardwareInitializer(
        camera_arm_name="camera_gripper",
        camera_fixed_name="camera_fixed",
        ignore_gripper=False,
        tool_name="tcp_gripper",
        debug=args.debug,
    )
    handles = hw.initialize()

    cycle_manager = GenericCycleManager(
        handles=handles,
        detector=detector,
        voice_client=None,
        move_to_start=False,
        debug=False,
    )

    grasp = None
    if args.pick_up:
        from qbot.qbot_grasp import QbotGrasp

        grasp = QbotGrasp(
            hw=handles,
            on_progress=on_progress,
            on_done=done,
            move_to_start=False,
            ignore_scanning=True,
            detector=detector,
            debug=args.debug,
        )

      
    move_to_start_position(handles)

    #move according to fixed camera first
    success, status = cycle_manager.move_to_generic_prompt(args.prompt,conf=0.2, z_offset=args.prompt_z_offset, camera_type='fixed',orientation_align=args.prompt_orientation_alignment,move=True)

    #move according to arm camera first
    success, status = cycle_manager.move_to_generic_prompt(args.prompt,conf=0.2, z_offset=args.prompt_z_offset, camera_type=args.prompt_camera_type,orientation_align=args.prompt_orientation_alignment,move=True)

    # success, status = cycle_manager.run_servo_prompt(args.prompt, z_offset=args.prompt_z_offset, camera_type=args.prompt_camera_type,orientation_align=args.prompt_orientation_alignment)

    if success:
        time.sleep(1)
        if args.pick_up:
            retry = True
            while retry:
                depth, color = handles.cam_arm.get_rgbd()
                boxes = detector.detect_bbox(
                    prompts=[args.prompt],
                    color=color,
                    conf_list=[0.1],
                    debug=args.debug,
                )
                if not boxes:
                    print(f"No object found for prompt: {args.prompt}")
                    break
                result = grasp.run(
                    det_query_list=[args.prompt],
                    conf_list=[0.1],
                    rotate_grasp=True,
                    blocking=True,
                    color_image=color,
                    depth_image=depth,
                    object_bboxes=boxes,
                )
                print('grasp result',result)
                if last_detail and 'failed' in last_status:
                    print('retrying')
                    handles.gripper.open()
                    retry = True
                else:
                    print(last_detail)
                    retry = False
    else:
        print("Failed to reach target for prompt: status ",status)     
    
        
if __name__ == "__main__":
    main()
