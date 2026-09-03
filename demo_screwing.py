import argparse
from hardware.hardware_init import HardwareInitializer
from qbot.qbot_cycles import ScrewCycleManager, ManualScrewCycleManager
from detectors.sam3_object_detection import Sam3Detector
from hardware.screwdriver_client import ScrewdriverClient
from llm.openai_realtime_client import OpenAIRealtimeClient
import time
import signal
import sys
import select

ACTIVE_SCREWDRIVER_CLIENT = None

# Handle Ctrl-C to clean up screwdriver client
def handle_sigint(signum, frame):
    print("\n[CTRL-C] Caught interrupt signal. Cleaning up...")

    global ACTIVE_SCREWDRIVER_CLIENT
    if ACTIVE_SCREWDRIVER_CLIENT is not None:
        try:
            print("[INFO] Stopping screwdriver server...")
            ACTIVE_SCREWDRIVER_CLIENT.stop()   # <-- You must have a .stop() method
        except Exception as e:
            print(f"[WARN] Failed to stop screwdriver server: {e}")

    print("[EXIT] Exiting program now.")
    sys.exit(0)
signal.signal(signal.SIGINT, handle_sigint)


def wait_tool_button_event(
    handles,
    long_press_sec: float = 2.5,   # <-- 2.5 seconds = long press
    poll: float = 0.02,
) -> str:
    """
    Waits for a tool button interaction.

    Returns:
        "short"  - button pressed and released before long_press_sec
        "long"   - button held for at least long_press_sec
    """
    print("[INFO] Waiting for tool button (short=auto, long=manual)...")

    pressed_start = None

    while True:
        try:
            pressed = handles.arm.get_tool_io()
        except Exception as e:
            raise RuntimeError(f"Failed to read tool button state: {e}")

        now = time.time()

        if pressed:
            if pressed_start is None:
                pressed_start = now
            elif now - pressed_start >= long_press_sec:
                print(f"[EVENT] Long press detected (>{long_press_sec:.1f}s) -> manual mode.")
                return "long"
        else:
            if pressed_start is not None:
                # Released before long_press_sec => short press
                held = now - pressed_start
                if held < long_press_sec:
                    print(f"[EVENT] Short press detected ({held:.2f}s) -> auto mode.")
                    return "short"
                # (In practice we should never hit this else, because long is already returned)

            pressed_start = None  # ensure we reset if it was never really pressed


        # Non-blocking check for keyboard ENTER
        try:
            rlist, _, _ = select.select([sys.stdin], [], [], 0)
        except Exception:
            rlist = []

        if rlist:
            # Read the line (user hit Enter or typed something)
            _ = sys.stdin.readline()
            print("[STEP] Enter key detected on keyboard.")
            return "short"


        time.sleep(poll)

def wait_stop_button_released(handle,voice_client , poll: float = 0.05):

    pressed = handle.arm.get_stop_io()
    if pressed and voice_client is not None:
        voice_client.speak_openai("Please release the stop button to start operation.")
    
    while True:
        try:
            pressed = handle.arm.get_stop_io()
        except Exception as e:
            msg = f"Failed to read stop button state before start: {e}"
            raise RuntimeError(msg)
        
        if not pressed:
            print("[STEP] Stop button released – starting operation.")
            return
        time.sleep(poll)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--use_servo", action="store_true")
    parser.add_argument("--enable_screwdriver", action="store_true")

    parser.add_argument("--mode", choices=["auto", "manual","none"], default="none", help="Initial mode (auto/manual)")

    return parser.parse_args()


def main():
    global ACTIVE_SCREWDRIVER_CLIENT

    args = parse_args()
    

    pickup_screw = True
    move_to_cross = True

    # Initialize hardware``
    hw = HardwareInitializer(
        camera_arm_name="camera_drill",
        camera_fixed_name="camera_fixed",
        ignore_gripper=True,
        tool_name="tcp_drill",
        debug=args.debug,
    )
    handles = hw.initialize()
    detector = Sam3Detector()
    screwdriver_client = (
        ScrewdriverClient() if args.enable_screwdriver else None
    )
    ACTIVE_SCREWDRIVER_CLIENT = screwdriver_client
    
    voice_client = OpenAIRealtimeClient.load()
    voice_client.speak_openai('Quendabot Online')
    
    # Create cycle manager
    cycle_manager = ScrewCycleManager(
        handles, detector, screwdriver_client, voice_client
    )
    
    # Create manual cycle manager
    manual_cycle_manager = ManualScrewCycleManager(
        handles,screwdriver_client,voice_client
    )
    
    print(f"[STEP] Checking stop button before starting...")
    wait_stop_button_released(handles,voice_client)
        
    try:
        while True:

            if  args.mode == "none":
                if voice_client is not None:
                    voice_client.speak_openai(
                        "Press trigger for automatic mode, or hold the trigger for manual mode."
                    )
                event = wait_tool_button_event(handles)  # <--- new
            else:
                if args.mode == "auto":
                    event = "short"
                else:
                    event = "long"

            if event == "short":

                # Automatic screw cycle (your existing behavior)
                success = cycle_manager.run_cycle(use_servo = args.use_servo,enable_screwdriver=args.enable_screwdriver,pickup_screw=pickup_screw,move_to_cross=move_to_cross,debug=args.debug)
                if not success:
                    print("[WARN] Screw cycle ended with errors. Continuing to next cycle.\n")
                else:
                    print("[INFO] Screw cycle completed successfully.\n")

            elif event == "long":
                # New manual mode
                manual_cycle_manager.run_cycle(use_servo = args.use_servo,debug=args.debug)

            if args.mode != "none":
                # Only run one cycle if mode was specified via command line
                break

    except KeyboardInterrupt:
        if ACTIVE_SCREWDRIVER_CLIENT is not None:
            try:
                print("[INFO] Stopping screwdriver server...")
                ACTIVE_SCREWDRIVER_CLIENT.stop()
            except Exception as e:
                print(f"[WARN] Failed to stop screwdriver server: {e}")


if __name__ == "__main__":
    main()