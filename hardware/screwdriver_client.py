#!/usr/bin/env python3
"""
Simple client for the Screwdriver Operation Server.

Usage from another file:

    from screwdriver_client import ScrewdriverClient

    client = ScrewdriverClient("http://127.0.0.1:8100")

    # Check health
    print(client.health())

    # Start operation asynchronously
    client.run_screw_async(retract_first=True, disengage_first=False, debug=True)

    # Poll status
    while True:
        status = client.get_status()
        print("Status:", status)
        if status.state in ("completed", "error"):
            break
        time.sleep(0.5)

    if status.state == "error":
        print("Operation failed:", status.error)
"""

from dataclasses import dataclass
from threading import Thread, Lock
from typing import Optional, Dict, Any
import time

import requests
import signal
import sys
import argparse
import json

@dataclass
class OperationStatus:
    """
    Simple status snapshot for the last / current screw operation.
    """
    state: str = "idle"               # "idle", "running", "completed", "error"
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
        }


class ScrewdriverClient:
    """
    Client for the Screwdriver Operation Server.

    Expects the server to expose:
        GET  /health
        POST /run_screw   (body: {"retract_first": bool, "disengage_first": bool, "debug": bool})
        POST /stop
    """

    def __init__(self, base_url: str = 'http://192.168.10.100:5560', timeout: float = 5.0):
        """
        :param base_url: Base URL of the server, e.g. "http://127.0.0.1:8100"
        :param timeout:  Default timeout (seconds) for short HTTP calls (health, stop).
                         The long-running /run_screw call is done without a timeout.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

        self._status = OperationStatus()
        self._lock = Lock()
        self._thread: Optional[Thread] = None

    # ---------- Internal helpers ----------

    def _set_status(self, **kwargs):
        with self._lock:
            for k, v in kwargs.items():
                setattr(self._status, k, v)

    # ---------- Public API ----------

    def health(self) -> Dict[str, Any]:
        """
        Call /health on the server.
        Raises requests.exceptions.RequestException on failure.
        """
        url = f"{self.base_url}/health"
        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def stop(self) -> Dict[str, Any]:
        """
        Request an emergency stop (/stop).
        """
        url = f"{self.base_url}/stop"
        resp = self.session.post(url, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def get_status(self) -> OperationStatus:
        """
        Get a snapshot of the current/last operation status.
        """
        with self._lock:
            # Return a shallow copy so callers can't mutate internal state
            return OperationStatus(
                state=self._status.state,
                started_at=self._status.started_at,
                finished_at=self._status.finished_at,
                error=self._status.error,
            )

    def is_running(self) -> bool:
        """
        Convenience helper: True when an operation is running.
        """
        return self.get_status().state == "running"

    def run_screw_async(
        self,
        retract_first: bool = False,
        disengage_first: bool = False,
        debug: bool = False,
    ) -> None:
        """
        Start a screw operation in a background thread and immediately return.

        Raises RuntimeError if an operation is already running.
        """

        # Prevent concurrent operations
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("An operation is already running")

        payload = {
            "retract_first": retract_first,
            "disengage_first": disengage_first,
            "debug": debug,
        }

        def worker():
            self._set_status(
                state="running",
                started_at=time.time(),
                finished_at=None,
                error=None,
            )
            url = f"{self.base_url}/run_screw"
            try:
                # Long-running request: no timeout so the operation can complete
                resp = self.session.post(url, json=payload)
                resp.raise_for_status()
                # You can parse resp.json() here if you want to store more info
                self._set_status(
                    state="completed",
                    finished_at=time.time(),
                    error=None,
                )
            except Exception as e:  # broad on purpose so we catch network + HTTP errors
                self._set_status(
                    state="error",
                    finished_at=time.time(),
                    error=str(e),
                )

        self._thread = Thread(target=worker, daemon=True)
        self._thread.start()

    def run_screw_blocking(
        self,
        retract_first: bool = False,
        disengage_first: bool = False,
        debug: bool = False,
        poll_interval: float = 0.5,
    ) -> OperationStatus:
        """
        Start a screw operation and block until it completes or errors.

        :returns: Final OperationStatus.
        """
        self.run_screw_async(
            retract_first=retract_first,
            disengage_first=disengage_first,
            debug=debug,
        )

        while True:
            status = self.get_status()
            if status.state in ("completed", "error"):
                return status
            time.sleep(poll_interval)



ACTIVE_SCREWDRIVER_CLIENT = None

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
    

def main():
    global ACTIVE_SCREWDRIVER_CLIENT
    ap = argparse.ArgumentParser(description="Simple client for Screwdriver Operation Server")
    ap.add_argument("--server-url", default="http://192.168.10.100:5560", help="Base URL of server")
    ap.add_argument("--retract", action="store_true")
    ap.add_argument("--disengage", action="store_true")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    client = ScrewdriverClient(args.server_url)
    ACTIVE_SCREWDRIVER_CLIENT = client

    print("Health:", json.dumps(client.health(), indent=2))

    print("Starting screw operation (blocking)...")
    try:
        final_status = client.run_screw_blocking(
            retract_first=args.retract,
            disengage_first=args.disengage,
            debug=args.debug,
        )
        print("Final status:", final_status.as_dict())
    except KeyboardInterrupt:
        try:
            print("[INFO] Stopping screwdriver server...")
            ACTIVE_SCREWDRIVER_CLIENT.stop()   # <-- You must have a .stop() method
        except Exception as e:
            print(f"[WARN] Failed to stop screwdriver server: {e}")    

    

# Optional quick CLI/demo usage
if __name__ == "__main__":
   main()
   