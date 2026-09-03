import time
from llm.openai_realtime_client import OpenAIRealtimeClient,is_client_realtime_running
from hardware.hardware_init import  HardwareInitializer
from llm.vlm_watcher import VisionPickUpWatcher
from qbot.qbot_behavior import BehaviorManager
import sys
from typing import Optional, Dict, Any, Tuple
import threading
from pathlib import Path
import yaml





def main(args):
    cycles_config_path = Path(__file__).resolve().parent / "qbot" / "config" / "cycles.yaml"
    with cycles_config_path.open("r", encoding="utf-8") as f:
        cycle_config = yaml.safe_load(f) or {}
    handover_joint = cycle_config.get("handover_joint")
    if handover_joint is None:
        raise ValueError("handover_joint missing from qbot/config/cycles.yaml")

    def on_progress(event: str, info: dict):
        print(f"[progress] {event}: {info}")


    # === Tool handlers that capture `qbot_runtime` (and `args`) ===
    # -------------------- Tool handlers -------------------- #
    def camera_snapshot_handler(request: str, reason: str) -> Dict[str, Any]:
        """
        Takes a snapshot and sends it to the model as an image message.
        The prompt_text reminds the model to call get_object.
        """
        try:
            frame_jpeg = hw.cam_arm.get_rgb_jpeg()
        except Exception as e:
            return {"status": "error", "error": f"snapshot_failed: {e}"}

        prompt = (f'image width: {hw.cam_arm.width} height: {hw.cam_arm.height} request: ' + request)
        try:
            client.send_image(image_bytes=frame_jpeg, prompt_text=prompt)
        except Exception as e:
            return {"status": "error", "error": f"send_image_failed: {e}"}

        return {"status": "ok", "sent_bytes": len(frame_jpeg)}


    def get_hand_over_handler(description: str):
        print('reason for hand over',description)
        try:
            # qbotaction.stop_all_actions() #stops all qbot actions
            qbotaction.start_handover_to_hand()
            res = {}
            return {"status": "ok", "action": "in progress", "result": res}
        except Exception as e:
            print("[TOOL:get_object] error:", e)
            return {"status": "error", "error": str(e), "result": res}

    def get_object_handler(description: str):
        """
        Run the Qbot to fetch something described by `description`.
        Uses `description` as the query; 
        """   
        query = (description or "").strip()
        if object_pipeline_error is not None:
            return {
                "status": "error",
                "error": f"object pickup pipeline unavailable: {object_pipeline_error}",
                "det_query": query,
            }

        if vlm_watcher:  
            vlm_watcher.pause_watcher() 

        if qbotgrasp and qbotgrasp.is_running():
            print("grasping pipline is busy cannot complete task")
            return {"status": "ok", "action": "grasping busy", "det_query": query}

        print(f"[TOOL:get_object] description={description!r}")
        if query == "":
            print("no object requested!!! ")
            return
        try:
            qbotaction.stop_all_actions() #stops all qbot actions

            print('query',query)     
            if vlm_watcher:
                vlm_watcher.pause_watcher()


            ####DETECT AND GRAB OBJECT   
            # det_query_list = [query] 
            # status = qbotgrasp.run(det_query_list=det_query_list, blocking=False)

            ####MOVE TO OBJECT FIRST THEN DETECT AND GRAB

            j = cycle_manager.config.get('grasp_joint')
            success, status = cycle_manager.move_to_generic_prompt(prompt=query,camera_type='arm',x_offset=-0.4,z_offset=0.5,move=True,start_joint=j,ignore_rotation=True)
            if success:
                det_query_list = [query]
                confidence_list = [0.2]
                depth, color = hw.cam_arm.get_rgbd()
                boxes = detector.detect_bbox(
                    prompts=det_query_list,
                    color=color,
                    conf_list=confidence_list,
                    debug=args.debug,
                )
                if not boxes:
                    return {
                        "status": "error",
                        "error": f"no object found for: {query}",
                        "det_query": query,
                    }
                qbotgrasp.run(
                    det_query_list=det_query_list,
                    conf_list=confidence_list,
                    rotate_grasp=True,
                    blocking=False,
                    color_image=color,
                    depth_image=depth,
                    object_bboxes=boxes,
                )
            else:
                client.send_text('hand over failed. task is [completed], move onto next task')
                print("Failed to reach target for prompt: status ",status)     


            return {"status": "ok", "action": "running", "det_query": query, "result": status}
        except Exception as e:
            print("[TOOL:get_object] error:", e)
            return {"status": "error", "error": str(e), "det_query": query}


    def vlm_get_object_handler(object: str, reason: str):
        """
        Simple test callback for VisionPickUpWatcher.

        In your real system, this is where you'd trigger the robot grasp behavior.
        """
        print("\n[VLM CALLBACK] pick_up_object called!")
        print(f"  object_description: {object}")
        print(f"  reason:     {reason}\n")

        # client.speak_openai(reason)

        #say reason for hand over
        # client.speak_openai(f"  i will go get the  {object} for you now\n")
        client.send_text(f"go get the {object}. reason: {reason}, say to user insight on the reasonget b")
        if vlm_watcher:
            vlm_watcher.pause_watcher() 
        

    def stop_action_handler(reason: str = ""):
        """
        Stop any ongoing Qbot action.
        """
        print(f"[TOOL:stop_action] reason={reason!r}")
        try:
            if qbotgrasp:
                qbotgrasp.stop()
            # qbotaction.start_point_at_person(distance_m=0.7, target="chest", rate_hz=10)
            #go to handover pose
            hw.arm.moveJ(handover_joint)#start at handover pose
            if vlm_watcher:
                vlm_watcher.resume_watcher()    
           
            return {"status": "ok", "action": "stopped"}
        except Exception as e:
            print("[TOOL:stop_action] error:", e)
            return {"status": "error", "error": str(e)}

    def hand_over_complete(payload):
        status = payload['status']
        info = payload['action']
        print('hand over complete',status,info)
        reason = info
        print(f"[on_done] status={status}")
        if 'completed' in status:
            client.send_text('hand over success. task is [completed], move onto next task')
            #go to handover pose
            hw.arm.moveJ(handover_joint)#start at handover pose
            if vlm_watcher:
                vlm_watcher.resume_watcher()    
           
        else:    
            #start  hand over routine
            if status in ("failed"):
                print("Error:", status + ", details: " +reason)
                # print(info.get("traceback"))  # if you enabled traceback above
                client.send_text(status + ', reason:' +reason+' [task is failed]')

            elif status in ("error"):
                print("Error:", status + ", details: " +reason)
                # print(info.get("traceback"))  # if you enabled traceback above
                client.send_text(status + ', reason:' +reason+' [task is failed]')

            elif status in ("stopped"):
                print("Error:", status + ", details: " +reason)
                # print(info.get("traceback"))  # if you enabled traceback above
                client.send_text(status + ', you stopped me:' +reason+' [task is failed]')

            else:
                print('Status Unknown' , status)

            print('DONT KNOW WHAT TO DO NEXT HAND OVER FAILED')
            #SHOULD DO SOMETHING HERE, UNDECIDED YET
            # qbotaction.start_idle_lookaround()


    #GRASPING CALLBACK
    def grapsing_action_complete(status: str, details: str, result: dict):
        print(f"[on_done] status={status}")

        if status in ("complete"):
            print("Result:", details)
            client.send_text('grap complete, ask user to hold out their hand [completed]')
            
            #start  hand over routine
            qbotaction.start_handover_to_hand(hand='either',on_complete=hand_over_complete)

        else:    
            #start  hand over routine
            if status in ("failed"):
                print("Error:", status + ", details: " +details)
                # print(info.get("traceback"))  # if you enabled traceback above
                client.send_text(status + ', reason: ' +details+' [task is failed]')

            elif status in ("error"):
                print("Error:", status + ", details: " +details)
                # print(info.get("traceback"))  # if you enabled traceback above
                client.send_text(status + ', reason:' +details+' [task is failed]')

            elif status in ("stopped"):
                print("Error:", status + ", details: " +details)
                # print(info.get("traceback"))  # if you enabled traceback above
                client.send_text(status + ', you stopped me:' +details+' [task is failed]')
            else:
                print('Status Unknown' , status)
            
            #go to handover pose
            hw.arm.moveJ(handover_joint)#start at handover pose

            # qbotaction.start_idle_lookaround()

    initializer = HardwareInitializer(
        camera_arm_name="camera_gripper",
        camera_fixed_name="camera_fixed",
        ignore_gripper=False,
        tool_name="tcp_gripper",
        debug=args.debug,
    )
    hw = initializer.initialize()
    
    hw.arm.moveJ(handover_joint)#start at handover pose
    
    qbotgrasp = None
    cycle_manager = None
    object_pipeline_error = None
    try:
        from detectors.sam3_subprocess import Sam3SubprocessDetector
        from qbot.qbot_grasp import QbotGrasp
        from qbot.qbot_cycles import GenericCycleManager

        detector = Sam3SubprocessDetector(env_name=args.sam3_env)
        qbotgrasp = QbotGrasp(
            hw=hw,
            on_progress=on_progress,
            on_done=grapsing_action_complete,
            debug=args.debug,
            interactive=args.interactive,
            move_to_start=False,
            detector=detector,
            ignore_scanning=False,
        )

        cycle_manager = GenericCycleManager(
            handles=hw,
            detector=detector,
            voice_client=None,
            move_to_start=False,
            debug=args.debug,
        )
    except Exception as exc:
        object_pipeline_error = exc
        print(f"[WARN] Object pickup pipeline disabled: {exc}")

    # OpenAI Voice Client
    client = OpenAIRealtimeClient.load()
    client.on_error = lambda e: print("[App] error:", e)
    client.on_text_delta = lambda s: None
    # Register handlers


    client.register_tool_handler("hand_over", get_hand_over_handler)
    client.register_tool_handler("camera_snapshot", camera_snapshot_handler)
    client.register_tool_handler("get_object", get_object_handler)
    client.register_tool_handler("stop_action", stop_action_handler)
    client.start()

    vlm_watcher = None
    if args.watcher_enabled:
        vlm_watcher = VisionPickUpWatcher(
            camera=hw.cam_fixed,
            interval_sec=2.0,
            show_debug_window=False,  # set False to disable the OpenCV window
            on_suggestion=vlm_get_object_handler,
            temperature=0.2,
            max_tokens=-1,
        )

    #start vlm watcher
    if vlm_watcher:
        vlm_watcher.start()

    # Qbot Actions
    qbotaction = BehaviorManager(hw, debug=args.debug)
    # if args.debug:        
    #     hub = DisplayHub([qbotaction.fix_cam_tracker,qbotaction.arm_cam_tracker])

    print("🎙️ Realtime client started. Type to chat. Commands: /snap <request>, /wake, /quit")

    time.sleep(1)
    client.send_text('say quendabot online', role="user", speak=True)

    def input_loop():
        try:
            while True:


                line = sys.stdin.readline()
                if not line:
                    time.sleep(0.05)
                    continue
                msg = line.strip()
                if not msg:
                    continue
                if msg.lower() in ("/q", "/quit", "quit", "exit"):
                    break
                if msg.startswith("/wake"):
                    print("waking up client")
                    client.wake()
                if msg.startswith("/snap"):
                    request = msg[len("/snap"):].strip() or "the requested item"
                    # Reuse the same path as the tool
                    res = camera_snapshot_handler(request=request, reason="manual keyboard snapshot")
                    if res.get("status") != "ok":
                        print("[/snap ERROR]", res)
                    continue

                # normal text message
                try:
                    client.send_text(msg, role="user", speak=True)
                except Exception as e:
                    print("[send_text ERROR]", e)
        except KeyboardInterrupt:
            pass

    t = threading.Thread(target=input_loop, name="keyboard_loop", daemon=True)
    t.start()


    # Wait until user quits
    try:
        while t.is_alive():
            # if args.debug:  
                # hub.tick(poll_ms=1)
            t.join(timeout=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        print("\nShutting down...")
        try:
            client.stop()
            if vlm_watcher:
                vlm_watcher.stop()
            hw.shutdown()
        except Exception:
            pass


# Optional quick CLI shim for ad-hoc runs
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/calibration.yaml")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--ignore_arm", action="store_true")
    ap.add_argument("--ignore_gripper", action="store_true")
    ap.add_argument("--interactive", action="store_true", help="skip motion confirmations")
    ap.add_argument("--watcher_enabled", action="store_true", help="enable vision pickup watcher")
    ap.add_argument("--sam3_env", default="quendabot_demo", help="Conda environment used for SAM3 inference")
    args = ap.parse_args()
    main(args)
