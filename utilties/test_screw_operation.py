#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
import threading
import time
from typing import Optional

import requests
import yaml


DEFAULT_CONFIG_FILE = Path(__file__).resolve().parent / "screw_operation.yaml"


def resolve_config_path(config_path: str) -> Path:
    candidate = Path(config_path)
    if candidate.is_file():
        return candidate

    script_dir_candidate = Path(__file__).resolve().parent / config_path
    if script_dir_candidate.is_file():
        return script_dir_candidate

    raise FileNotFoundError(f"Could not find config file: {config_path}")


def load_config(config_path: str) -> dict:
    with resolve_config_path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def build_base_url(cfg: dict, host_override: Optional[str], port_override: Optional[int]) -> str:
    host = host_override or cfg.get("listen_host", "127.0.0.1")
    port = port_override if port_override is not None else int(cfg.get("listen_port", 5560))
    return f"http://{host}:{port}"


def response_payload(response: requests.Response):
    try:
        return response.json()
    except ValueError:
        return response.text.strip()


def format_api_error(response: requests.Response) -> str:
    payload = response_payload(response)
    if isinstance(payload, dict) and "detail" in payload:
        return f"HTTP {response.status_code}: {payload['detail']}"
    if payload:
        return f"HTTP {response.status_code}: {payload}"
    return f"HTTP {response.status_code}"


def wait_for_server(session: requests.Session, base_url: str, wait_timeout: float, request_timeout: float):
    deadline = time.monotonic() + wait_timeout if wait_timeout is not None and wait_timeout >= 0 else None

    while True:
        try:
            response = session.get(f"{base_url}/op_status", timeout=request_timeout)
            if response.ok:
                return response.json()
        except requests.RequestException:
            pass

        if deadline is not None and time.monotonic() >= deadline:
            raise RuntimeError(
                f"Screw operation server did not respond at {base_url} within {wait_timeout:.1f} seconds."
            )

        time.sleep(0.5)


def fetch_status(session: requests.Session, base_url: str, timeout: float) -> dict:
    response = session.get(f"{base_url}/op_status", timeout=timeout)
    response.raise_for_status()
    return response.json()


def print_status(prefix: str, status: dict):
    state = status.get("state", "unknown")
    phase = status.get("phase", "unknown")
    message = status.get("message") or ""
    last_error = status.get("last_error")
    telemetry = status.get("telemetry") or {}

    parts = [f"state={state}", f"phase={phase}"]
    if message:
        parts.append(f"message={message}")
    if last_error:
        parts.append(f"last_error={last_error}")
    elapsed = telemetry.get("elapsed_s")
    if elapsed is not None:
        parts.append(f"t={elapsed:.2f}s")
    z = telemetry.get("z")
    if z is not None:
        parts.append(f"z={z:.4f}m")
    dz_from_start = telemetry.get("dz_from_start")
    if dz_from_start is not None:
        parts.append(f"dz_start={dz_from_start:.4f}m")
    dz_from_contact = telemetry.get("dz_from_contact")
    if dz_from_contact is not None:
        parts.append(f"dz_contact={dz_from_contact:.4f}m")
    fz = telemetry.get("fz")
    if fz is not None:
        parts.append(f"Fz={fz:.2f}N")
    fz_spike = telemetry.get("fz_spike")
    if fz_spike is not None:
        parts.append(f"-Zspike={fz_spike:.2f}N")
    fnorm = telemetry.get("force_norm")
    if fnorm is not None:
        parts.append(f"|F|={fnorm:.2f}N")
    drill_mode = telemetry.get("drill_mode")
    if drill_mode:
        parts.append(f"drill={drill_mode}")
    moving = telemetry.get("moving")
    if moving is not None:
        parts.append(f"moving={moving}")
    contact_detected = telemetry.get("contact_detected")
    if contact_detected is not None:
        parts.append(f"contact={contact_detected}")

    print(f"{prefix} {' | '.join(parts)}")


def run_stop(session: requests.Session, base_url: str, timeout: float) -> int:
    try:
        response = session.post(f"{base_url}/stop", timeout=timeout)
    except requests.RequestException as exc:
        print(f"[ERROR] Failed to stop operation: {exc}", file=sys.stderr)
        return 1

    payload = response_payload(response)

    if not response.ok:
        print(f"[ERROR] Failed to stop operation: {format_api_error(response)}", file=sys.stderr)
        return 1

    print("[STOP] Stop request accepted.")
    if isinstance(payload, dict) and "operation" in payload:
        print_status("[STOP]", payload["operation"])
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="Test client for the screw driving operation server")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_FILE),
        help="Path to screw_operation.yaml",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Override the screw operation server host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Override the screw operation server port from the config",
    )
    parser.add_argument(
        "--wait-timeout",
        type=float,
        default=10.0,
        help="Seconds to wait for the screw operation server to come online",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=3.0,
        help="Timeout for short HTTP requests like status polling",
    )
    parser.add_argument(
        "--status-interval",
        type=float,
        default=1.0,
        help="Seconds between /op_status polls while the operation is running",
    )
    parser.add_argument(
        "--run-timeout",
        type=float,
        default=None,
        help="Optional read timeout for /run_screw; default waits indefinitely",
    )
    parser.add_argument(
        "--retract-first",
        action="store_true",
        help="Request the retract URP before starting the screw cycle",
    )
    parser.add_argument(
        "--disengage-first",
        action="store_true",
        help="Request the disengage URP before starting the screw cycle",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for this screw operation request",
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Send /stop instead of starting a new screw operation",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    base_url = build_base_url(cfg, args.host, args.port)

    session = requests.Session()

    try:
        initial_status = wait_for_server(session, base_url, args.wait_timeout, args.request_timeout)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print(f"[INFO] Screw operation server: {base_url}")
    print_status("[INFO]", initial_status)

    if args.stop:
        return run_stop(session, base_url, args.request_timeout)

    try:
        health_response = session.get(f"{base_url}/health", timeout=args.request_timeout)
        if health_response.ok:
            print("[INFO] Health check passed.")
        else:
            print(f"[WARN] Health check reported an issue: {format_api_error(health_response)}")
    except requests.RequestException as exc:
        print(f"[WARN] Health check could not be completed: {exc}")

    request_body = {
        "retract_first": args.retract_first,
        "disengage_first": args.disengage_first,
        "debug": args.debug,
    }

    result: dict = {}
    done = threading.Event()

    def call_run_screw():
        try:
            with requests.Session() as run_session:
                response = run_session.post(
                    f"{base_url}/run_screw",
                    json=request_body,
                    timeout=args.run_timeout,
                )
            result["status_code"] = response.status_code
            result["payload"] = response_payload(response)
            if response.ok:
                result["ok"] = True
            else:
                result["ok"] = False
                result["error"] = format_api_error(response)
        except requests.RequestException as exc:
            result["ok"] = False
            result["error"] = str(exc)
        finally:
            done.set()

    worker = threading.Thread(target=call_run_screw, daemon=True)
    worker.start()

    print("[RUN] Screw operation request sent.")

    last_snapshot = None
    try:
        while not done.wait(args.status_interval):
            try:
                status = fetch_status(session, base_url, args.request_timeout)
            except requests.RequestException as exc:
                print(f"[WARN] Status poll failed: {exc}")
                continue

            snapshot = (
                status.get("state"),
                status.get("phase"),
                status.get("message"),
                status.get("last_error"),
            )
            if status.get("state") == "running" or snapshot != last_snapshot:
                print_status("[STATUS]", status)
                last_snapshot = snapshot
    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C received. Sending stop request...")
        stop_rc = run_stop(session, base_url, args.request_timeout)
        done.wait(5.0)
        return 130 if stop_rc == 0 else 1

    worker.join(timeout=0.1)

    try:
        final_status = fetch_status(session, base_url, args.request_timeout)
        print_status("[FINAL]", final_status)
    except requests.RequestException as exc:
        print(f"[WARN] Could not fetch final operation status: {exc}")

    if result.get("ok"):
        print("[DONE] Screw operation completed successfully.")
        payload = result.get("payload")
        if isinstance(payload, dict):
            print(payload)
        return 0

    print(f"[ERROR] Screw operation failed: {result.get('error', 'unknown error')}", file=sys.stderr)
    payload = result.get("payload")
    if payload:
        print(payload, file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
