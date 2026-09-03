#!/usr/bin/env python3
import argparse
import json
import socket
import sys
import time
from pathlib import Path

import yaml
import requests


DEFAULT_HOST = "192.168.10.100"
DEFAULT_PORT = 5556
DEFAULT_SCREWDRIVER_URL = "http://192.168.10.100:5560"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "hardware.yaml"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hardware.screwdriver_client import ScrewdriverClient


def load_default_port():
    try:
        with open(CONFIG_PATH, "r") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("drill", {}).get("listen_port", DEFAULT_PORT)
    except Exception:
        return DEFAULT_PORT


def read_line(reader):
    line = reader.readline()
    if not line:
        raise RuntimeError("Drill server closed the connection")
    return line.rstrip("\r\n")


def send_command(writer, reader, cmd):
    print(f">>> {cmd}")
    writer.write(f"{cmd}\n")
    writer.flush()
    reply = read_line(reader)
    print(f"<<< {reply}")
    if not reply.startswith("OK"):
        raise RuntimeError(f"{cmd} failed: {reply}")
    return reply


def print_http_response(label, response):
    print(f"{label} status: {response.status_code}")
    try:
        body = response.json()
        print(f"{label} body:", json.dumps(body, indent=2))
    except Exception:
        print(f"{label} body: {response.text}")


def screwdriver_health_ready(health_body):
    if isinstance(health_body, dict) and isinstance(health_body.get("detail"), dict):
        health_body = health_body["detail"]
    if not isinstance(health_body, dict):
        return False, "health response is not JSON object"
    if health_body.get("status") != "ok":
        return False, f"status={health_body.get('status')!r}"
    if not bool(health_body.get("ur_online", False)):
        return False, "UR robot is not online"
    if not bool(health_body.get("drill_online", False)):
        return False, "drill is not online"
    return True, ""


def test_screwing_operation(
    server_url=DEFAULT_SCREWDRIVER_URL,
    *,
    retract_first=False,
    disengage_first=False,
    debug=False,
    poll_interval=0.5,
    check_health=True,
    require_health=False,
):
    """
    Run the full screwdriver/screwing operation through the screwdriver server.

    This is different from the raw drill TCP commands below: it calls the
    Screwdriver Operation Server's /run_screw endpoint and waits until the
    operation finishes or reports an error.
    """
    client = ScrewdriverClient(base_url=server_url)

    if check_health:
        health_response = client.session.get(
            f"{client.base_url}/health",
            timeout=client.timeout,
        )
        print_http_response("Health", health_response)
        try:
            health_body = health_response.json()
        except Exception:
            health_body = None
        ready, not_ready_reason = screwdriver_health_ready(health_body)
        if health_response.status_code >= 400 or not ready:
            message = "Screwdriver server is not ready"
            if not_ready_reason:
                message = f"{message}: {not_ready_reason}"
            if require_health:
                raise RuntimeError(message)
            print(f"[WARN] {message} Continuing because --require-health was not set.")

    print("[STEP] Starting screwing operation...")
    payload = {
        "retract_first": retract_first,
        "disengage_first": disengage_first,
        "debug": debug,
    }
    try:
        response = client.session.post(
            f"{client.base_url}/run_screw",
            json=payload,
        )
    except KeyboardInterrupt:
        print("\n[CTRL-C] Stopping screwing operation...")
        try:
            print("Stop:", json.dumps(client.stop(), indent=2))
        except Exception as exc:
            print(f"[WARN] Failed to stop screwdriver server: {exc}")
        raise

    print_http_response("Run screw", response)
    try:
        response_body = response.json()
    except Exception:
        response_body = {"raw_response": response.text}

    ok = response.status_code < 400
    return ok, response_body


def main():
    parser = argparse.ArgumentParser(
        description="Test drill start/stop through the running drill TCP server."
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="Drill server host")
    parser.add_argument(
        "--port",
        type=int,
        default=load_default_port(),
        help="Drill server TCP port",
    )
    parser.add_argument(
        "--speed",
        choices=("slow", "medium", "fast"),
        default="slow",
        help="Speed command to test before stopping",
    )
    parser.add_argument(
        "--direction",
        choices=("forward", "reverse", "skip"),
        default="forward",
        help="Optional direction command before starting the drill",
    )
    parser.add_argument(
        "--run-seconds",
        type=float,
        default=1.0,
        help="How long to wait between start and stop",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=3.0,
        help="Socket timeout in seconds",
    )
    parser.add_argument(
        "--screw-operation",
        action="store_true",
        help="Run the full screwdriver/screwing operation instead of raw drill TCP commands.",
    )
    parser.add_argument(
        "--server-url",
        default=DEFAULT_SCREWDRIVER_URL,
        help="Screwdriver operation server URL for --screw-operation.",
    )
    parser.add_argument(
        "--retract",
        action="store_true",
        help="Ask the screwing operation to retract first.",
    )
    parser.add_argument(
        "--disengage",
        action="store_true",
        help="Ask the screwing operation to disengage first.",
    )
    parser.add_argument(
        "--debug-operation",
        action="store_true",
        help="Enable debug mode for the screwing operation.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.5,
        help="Polling interval while waiting for --screw-operation to finish.",
    )
    parser.add_argument(
        "--skip-health",
        action="store_true",
        help="Skip screwdriver server health check before --screw-operation.",
    )
    parser.add_argument(
        "--require-health",
        action="store_true",
        help="Fail before running if screwdriver server health returns an error.",
    )
    args = parser.parse_args()

    if args.screw_operation:
        try:
            ok, _status = test_screwing_operation(
                server_url=args.server_url,
                retract_first=args.retract,
                disengage_first=args.disengage,
                debug=args.debug_operation,
                poll_interval=args.poll_interval,
                check_health=not args.skip_health,
                require_health=args.require_health,
            )
            return 0 if ok else 1
        except KeyboardInterrupt:
            return 130
        except Exception as exc:
            print(f"Screwing operation test failed: {exc}", file=sys.stderr)
            return 1

    speed_cmd = args.speed.upper()
    direction_cmd = None if args.direction == "skip" else args.direction.upper()
    started = False

    try:
        with socket.create_connection((args.host, args.port), timeout=args.timeout) as sock:
            sock.settimeout(args.timeout)
            reader = sock.makefile("r", encoding="utf-8", newline="\n")
            writer = sock.makefile("w", encoding="utf-8", newline="\n")

            banner = read_line(reader)
            print(f"<<< {banner}")

            send_command(writer, reader, "STATUS")

            if direction_cmd:
                send_command(writer, reader, direction_cmd)

            send_command(writer, reader, speed_cmd)
            started = True

            if args.run_seconds > 0:
                print(f"... waiting {args.run_seconds:.2f}s ...")
                time.sleep(args.run_seconds)

            send_command(writer, reader, "STOP")
            started = False
            send_command(writer, reader, "STATUS")
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130
    except Exception as exc:
        print(f"Test failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if started:
            print("Note: the client disconnected after starting the drill.")
            print("The server should still send a best-effort STOP on disconnect.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
