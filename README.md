# Quendabot human–robot collaboration demos

This repository contains the Quendabot object retrieval, handover, grasping, and screw-installation demos used with a UR10e, RealSense cameras, a Robotiq gripper, and a network-controlled screwdriver.

The system is split across two computers:

- **Robot computer** — runs the demos and AI pipelines in this repository.
- **Hardware gateway** — connects directly to the UR controller, RealSense cameras, gripper, and screwdriver, then exposes them to the robot computer over the network.

> [!CAUTION]
> These programs command a physical robot. Before enabling motion, clear the workspace, verify the tool/TCP and payload, reduce the initial speed, test the stop input, and keep an operator at the emergency stop. Never use example transforms or joint positions on a robot that has not been calibrated for them.

## Available systems

| System | Entry point | Purpose |
| --- | --- | --- |
| Object approach/pickup | `demo_goto_object.py` | Find a text-prompted object, move above it, and optionally grasp it. |
| Object retrieval and handover | `demo_handover.py` | Voice-controlled pickup followed by a person-aware handover. |
| Basic screw cycle | `demo_screwing.py` | Trigger-controlled automatic or manual screw installation. |
| LLM screw installation | `demo_screwing_llm.py` | Voice/text screw selection, pickup, target detection, and installation. |
| LLM gripper + screw workflow | `demo_screwing_gripper_llm.py` | Load objects with the gripper and perform screw workflows with optional VLM suggestions. |
| Individual behaviours | `qbot/qbot_behavior.py` | Test pointing, looking, and handover behaviours. |
| Individual grasp | `qbot/qbot_grasp.py` | Run one AnyGrasp pickup attempt. |
| Cycle/debug harness | `qbot/qbot_cycles.py` | Test screw cycles and generic prompt motion. |

Run every command in this README from the repository root unless stated otherwise.

## 1. Clone the repository

SAM3 is stored as a Git submodule:

```bash
git clone --recurse-submodules git@github.com:gibsonuts/An-Intention-To-Action-Framework-For-Human-Robot-Collaborative-Object-Retrieval-and-Handover-.git
cd An-Intention-To-Action-Framework-For-Human-Robot-Collaborative-Object-Retrieval-and-Handover-
```

If the repository was cloned without submodules:

```bash
git submodule update --init --recursive
```

## 2. Install the controller environment

The checked-in AnyGrasp extension names target CPython 3.10, so Python 3.10 is recommended for the main controller environment. CUDA, PyTorch, MinkowskiEngine, and the AnyGrasp extensions must be mutually compatible.

On Ubuntu, install the common native dependencies:

```bash
sudo apt update
sudo apt install -y build-essential git libgl1 libglib2.0-0 portaudio19-dev
```

Create the environment:

```bash
conda create -n quendabot_anygrasp python=3.10 -y
conda activate quendabot_anygrasp
python -m pip install --upgrade pip setuptools wheel
```

Install the Python packages used by the controller, cameras, vision, and speech clients:

```bash
python -m pip install \
  numpy scipy Pillow tqdm PyYAML requests httpx pyzmq \
  opencv-python open3d matplotlib mediapipe \
  openai sounddevice soundfile websocket-client azure-cognitiveservices-speech
```

Install the vendored grasp packages after installing a CUDA-compatible PyTorch build:

```bash
python -m pip install -e anygrasp/graspnetAPI
python -m pip install -e anygrasp/pointnet2
python -m pip install -e anygrasp/MinkowskiEngine
```

AnyGrasp is licensed software. Obtain the following from the authorised AnyGrasp distribution and place them locally in the corresponding paths:

```text
anygrasp/checkpoint_detection.tar
anygrasp/checkpoint_tracking.tar
anygrasp/gsnet.cpython-310-x86_64-linux-gnu.so
anygrasp/tracker.cpython-310-x86_64-linux-gnu.so
anygrasp/lib_cxx.so
anygrasp/license/
```

These files are intentionally excluded from Git. Do not commit user-bound licence or activation files.

## 3. Configure object segmentation

There are two supported SAM3 modes.

### Option A: Roboflow-hosted SAM3

This is the backend currently selected in `qbot/config/cycles.yaml` and does not require a local SAM3 GPU environment:

```bash
export ROBOFLOW_API_KEY="your-roboflow-key"
export SAM3_BACKEND=roboflow
```

The environment variable is used by direct `Sam3Detector()` calls; the YAML setting is used by workflows that construct the detector from `cycles.yaml`.

### Option B: local SAM3

Local SAM3 requires Python 3.12+, PyTorch 2.7+, and a compatible CUDA GPU. Create the environment expected by the subprocess demos:

```bash
conda create -n quendabot_demo python=3.12 -y
conda activate quendabot_demo

# Install a PyTorch build appropriate for this machine first, then:
python -m pip install -e dependencies/sam3
python -m pip install numpy Pillow opencv-python requests
```

Request checkpoint access and authenticate with Hugging Face as described in `dependencies/sam3/README.md`. The SAM3 model builder downloads/loads the authorised checkpoint using the upstream package.

If the environment is installed somewhere unusual, tell the subprocess detector which interpreter to use:

```bash
export SAM3_PYTHON="$(conda run -n quendabot_demo which python)"
export SAM3_BACKEND=local
```

## 4. Configure credentials

Keep credentials in the shell environment; do not paste them into YAML files:

```bash
export OPENAI_API_KEY="your-openai-key"
export ROBOFLOW_API_KEY="your-roboflow-key"          # hosted SAM3 only
export AZURE_SPEECH_KEY="your-azure-speech-key"      # wake-word mode only
export AZURE_SPEECH_REGION="australiaeast"            # wake-word mode only
```

The LLM and speech settings are in `llm/config/config.yaml`. To run without Azure wake-word support, set `speechsdk.enable_wake_word: false`; text input and the configured speech pipeline can still be used.

## 5. Configure the hardware

Edit these files for the actual installation:

- `hardware/config/config.yaml` — camera serials/intrinsics, gateway addresses, robot address, and camera transforms.
- `hardware/config/arm_control.yaml` — UR tool TCPs, motion limits, impedance settings, and payload calibration.
- `qbot/config/cycles.yaml` — calibrated joints, motion offsets, stop/trigger polarity, detection, verification, and LLM settings.
- `qbot/config/grasp.yaml` — grasp scan volume, point-cloud filtering, and grasp angle limits.
- `qbot/config/behaviour.yaml` — handover and person-tracking behaviours.

The current network defaults are:

| Service | Default address | Protocol |
| --- | --- | --- |
| RealSense cameras | `192.168.10.100:5552` | ZMQ RGB/depth stream |
| UR robot gateway | `http://192.168.10.100:5553` | HTTP |
| Robotiq gripper gateway | `192.168.10.100:5554` | newline-delimited JSON/TCP |
| Screwdriver gateway | `http://192.168.10.100:5560` | HTTP |
| UR10e controller | `192.168.10.112` | accessed by the gateway |

The main demos always create network camera clients. Start the required gateway services before running a demo. This repository includes `utilties/realsense_server.py` and direct hardware drivers, but the combined UR, gripper, and screwdriver server deployment is installation-specific and must implement the protocols expected by the corresponding client modules.

For the bundled RealSense publisher, create a gateway-side YAML file with a `realsense.listen_port` value and one or more camera blocks, then run:

```bash
python utilties/realsense_server.py --config hardware.yaml --bind 0.0.0.0
```

The serial numbers in the gateway configuration must match the `serial` fields in `hardware/config/config.yaml`, because the clients subscribe by camera ID.

## 6. Calibrate before enabling motion

Perform calibration with low robot speeds and save the resulting values in the configuration files:

1. Confirm each UR TCP in `hardware/config/arm_control.yaml` for the physically attached tool.
2. Configure the tool mass, centre of gravity, and UR payload.
3. Calibrate `T_tcp_drill_camera` and `T_tcp_gripper_camera`.
4. Calibrate `T_base_fixed_camera`.
5. Verify RealSense intrinsics and depth scale for every camera.
6. Teach every joint pose in `qbot/config/cycles.yaml` on the actual cell.
7. Verify `hardware.stop_io.active_high` and the tool trigger behaviour.
8. Start with conservative `j_speed`, `j_accel`, `l_speed`, and `l_accel` values.

Do not copy transforms, TCPs, or joint values from another robot cell.

## 7. Bring up and test components

Activate the controller environment:

```bash
conda activate quendabot_anygrasp
```

Check the network paths first:

```bash
ping 192.168.10.100
ping 192.168.10.112
```

Preview the three configured camera IDs; press Esc to close the windows:

```bash
python hardware/camera_rs_client.py \
  --server-ip 192.168.10.100 --port 5552 \
  --cams 023422070739 102122070147 950122070492
```

Read the robot TCP and stop/tool inputs without commanding a move:

```bash
python utilties/check_robot_position.py --tool-name tcp_drill
```

Read the gripper status:

```bash
python hardware/robotik_gripper_client.py \
  --server_host 192.168.10.100 --server-port 5554 status
```

Check the screwdriver HTTP service without starting the tool:

```bash
curl http://192.168.10.100:5560/health
```

Only after these checks pass should motion be tested. Use focused debug stages and keep the real screwdriver disabled initially.

## 8. Run the demos

### Find and optionally pick up an object

```bash
python demo_goto_object.py --prompt "red cup" --debug
python demo_goto_object.py --prompt "red cup" --pick_up --debug
```

Useful options include `--prompt_camera_type fixed|arm`, `--prompt_z_offset`, `--prompt_orientation_alignment short|long`, and `--sam3_env`.

### Voice-controlled retrieval and handover

```bash
python demo_handover.py --debug --interactive
```

Add `--watcher_enabled` to enable fixed-camera VLM suggestions. The `--interactive` option skips some motion confirmations; use it only after the cell has been validated.

### Basic screw cycle

Run a single automatic cycle without starting the screwdriver client:

```bash
python demo_screwing.py --mode auto --debug
```

Run manual mode:

```bash
python demo_screwing.py --mode manual --debug
```

After validating pickup, target detection, stop behaviour, and tool alignment, add `--enable_screwdriver` to enable the real screwdriver service. With no `--mode`, a short tool-trigger press selects automatic mode and a long press selects manual mode.

### LLM screw installation

Start the voice/text interface without the real screwdriver:

```bash
python demo_screwing_llm.py --debug
```

Focused stage tests:

```bash
python demo_screwing_llm.py --debug-step pickup --debug-request "a screw" --debug
python demo_screwing_llm.py --debug-step target --debug
python demo_screwing_llm.py --debug-step screw --enable_screwdriver --debug
```

For the full physical workflow:

```bash
python demo_screwing_llm.py --enable_screwdriver --debug
```

At the prompt, use `/listen`, `/say <request>`, or `/quit`.

### Combined gripper and screw workflow

```bash
python demo_screwing_gripper_llm.py --debug
```

It supports focused `--debug-step` values: `load`, `pickup`, `pickup_target`, `pickup_target_screwdriver`, `target`, `screw`, and `watcher`. For example:

```bash
python demo_screwing_gripper_llm.py \
  --debug-step load --debug-request spanner --load-debug-no-program --debug
```

Enable the physical screwdriver only after the preceding stages are safe:

```bash
python demo_screwing_gripper_llm.py --enable_screwdriver --debug
```

Keyboard commands include `/listen`, `/say <text>`, `/stop`, and `/quit`.

### Individual behaviours and grasping

```bash
python qbot/qbot_behavior.py --behavior point --debug
python qbot/qbot_behavior.py --behavior look --debug
python qbot/qbot_behavior.py --behavior handover --hand either --debug
python qbot/qbot_grasp.py --query "red cup" --debug
```

## 9. Troubleshooting

- **`SAM3 Python not found`** — set `SAM3_PYTHON` to the Python executable inside the SAM3 environment.
- **SAM3 import/checkpoint failure** — initialise the submodule, install `dependencies/sam3`, authenticate with Hugging Face, and verify the CUDA/PyTorch versions.
- **`Roboflow SAM3 ... no API key`** — export `ROBOFLOW_API_KEY` or switch the backend to `local`.
- **AnyGrasp import failure** — use Python 3.10 and install the authorised AnyGrasp binaries, checkpoint, licence, PointNet2, and MinkowskiEngine builds for the same CUDA/PyTorch stack.
- **No camera frames** — check the gateway process, firewall/port 5552, camera serials, topic prefix, and that color/depth resolutions match the configured intrinsics.
- **Robot gateway unavailable** — check port 5553 and confirm the gateway can reach the UR controller at its configured IP.
- **Unexpected trigger/stop behaviour** — correct `hardware.stop_io.active_high` in `qbot/config/cycles.yaml` before proceeding.
- **OpenAI or speech authentication failure** — verify the exported variables in the same shell used to start the demo.
- **Audio device errors** — list PortAudio devices and update `input_device`/`output_device` in `llm/config/config.yaml`.

## Shutdown

Use `/quit` where supported or press Ctrl+C once. Confirm the robot and tool have stopped, then stop the gateway processes. If a client exits abnormally, use the physical emergency stop and the gateway's stop controls before restarting software.
