#!/usr/bin/env python3
"""
Client-side drop-in that mimics your `RobotiqOnUR` API but talks to the
remote gripper server over JSON/TCP.

Usage (same feel as your local class):

    g = RobotiqOnUR(
        server_host="192.168.0.10",  # IP/host where your server_yaml.py is running
        server_port=55555,
        auth_token="SUPERSECRET",
    )
    g.activate()
    g.open(block=True)
    print(g.get_width_m())
    g.close(block=True)

Notes:
- Methods map to server commands: activate/open/close/goto_frac/goto_width/
  get_width/get_status/is_closed.
- get_position_raw() is implemented by asking for status and extracting POS.
- Motor current (COU) is returned in get_status() if the server/URCap provides it.
"""

import json
import socket
from typing import Optional, Tuple, Any, Dict
import sys 


class RobotiqOnURClient:
    def __init__(self,
                 server_host: str,
                 server_port: int = 55555,
                 # keep these for drop-in compatibility; they are not used client-side
                 max_width: float = 0.085,  # optional hint; server is source of truth
                 timeout: float = 5.0):
        self.server_host = server_host
        self.server_port = server_port
        self.timeout = timeout
        self.max_open_m = max_width
        self._rid = 0
        self._sock: Optional[socket.socket] = None
        self._file = None  # type: Optional[Any]

        # Establish a persistent TCP connection to the server.
        try:
            self._connect()
        except (ConnectionRefusedError, socket.timeout, OSError) as e:
            # ConnectionRefusedError is a subclass of OSError, but kept for clarity
            print(
                f"[ERROR] Could not connect to gripper server at "
                f"{self.server_host}:{self.server_port}: {e} , is it running?"
            )
            # Exit the program if the server is not reachable
            sys.exit(1)

        # Establish a persistent TCP connection to the server.
        self._connect()

    # ----------- transport -----------
    def _connect(self) -> None:
        if self._sock:
            return
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(self.timeout)
        s.connect((self.server_host, self.server_port))
        self._sock = s
        self._file = s.makefile("rwb")

    def _rpc(self, cmd: str, args: Optional[Dict[str, Any]] = None):
        if args is None:
            args = {}
        if not self._sock:
            self._connect()
        self._rid += 1
        req = {"id": self._rid, "cmd": cmd, "args": args}
        data = (json.dumps(req) + "\n").encode("utf-8")
        try:
            assert self._file is not None
            self._file.write(data)
            self._file.flush()
            line = self._file.readline()
            if not line:
                raise RuntimeError("server closed connection")
            resp = json.loads(line.decode("utf-8").strip())
        except (BrokenPipeError, ConnectionResetError):
            # reconnect once
            self._close()
            self._connect()
            assert self._file is not None
            self._file.write(data)
            self._file.flush()
            line = self._file.readline()
            if not line:
                raise RuntimeError("server closed connection after reconnect")
            resp = json.loads(line.decode("utf-8").strip())

        if not resp.get("ok", False):
            raise RuntimeError(resp.get("error", "unknown error"))
        return resp.get("result")

    def _close(self):
        try:
            if self._file:
                self._file.close()
        finally:
            self._file = None
            if self._sock:
                try:
                    self._sock.close()
                finally:
                    self._sock = None

    def __enter__(self):
        self._connect()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._close()

    # ----------- high-level API (mirrors your class) -----------
    def activate(self, wait: bool = True, timeout: float = 5.0) -> None:
        self._rpc("activate", {"wait": bool(wait), "timeout": float(timeout)})

    def open(self, block: bool = True, timeout: float = 5.0) -> None:
        self._rpc("open", {"block": bool(block), "timeout": float(timeout)})

    def close(self, block: bool = True, timeout: float = 5.0) -> None:
        self._rpc("close", {"block": bool(block), "timeout": float(timeout)})

    def is_completely_closed(self, tol_pos: int = 30, require_stopped: bool = True,
                             min_current: Optional[int] = None) -> bool:
        res = self._rpc("is_closed", {
            "tol_pos": int(tol_pos),
            "require_stopped": bool(require_stopped),
            "min_current": (int(min_current) if min_current is not None else None),
        })
        return bool(res.get("closed"))

    def go_to_fraction(self, frac: float, block: bool = True, timeout: float = 5.0) -> int:
        res = self._rpc("goto_frac", {"frac": float(frac), "block": bool(block), "timeout": float(timeout)})
        return int(res.get("pos"))

    def go_to_position_metres(self, width_m: float, block: bool = True, timeout: float = 5.0) -> int:
        res = self._rpc("goto_width", {"width_m": float(width_m), "block": bool(block), "timeout": float(timeout)})
        return int(res.get("pos"))

    def get_width_m(self) -> float:
        res = self._rpc("get_width")
        # Keep local hint up-to-date if server shares it in config (optional)
        try:
            self.max_open_m = float(res.get("max_width", self.max_open_m))
        except Exception:
            pass
        return float(res["width_m"])  # server guarantees this key

    def get_status(self) -> dict:
        return dict(self._rpc("get_status"))

    # ----------- compatibility helpers -----------
    def get_position_raw(self) -> int:
        status = self.get_status()
        pos = status.get("POS")
        if pos is None:
            # Server should return POS; if not, degrade gracefully
            raise RuntimeError("Server did not return POS in status")
        return int(pos)

    def get_motor_current(self) -> Optional[int]:
        status = self.get_status()
        v = status.get("COU")
        return int(v) if v is not None else None

    # These are no-ops on the client; the server blocks until ready/motion done
    def _wait_until_ready(self, timeout: float = 5.0) -> None:
        return None

    def _wait_motion_done(self, timeout: float = 5.0, eps: int = 1):
        return None

    def is_holding_object(self,
                          tol_pos_closed: int = 30,
                          tol_pos_open: int = 10,
                          require_stopped: bool = True,
                          min_current: Optional[int] = None) -> bool:
        """
        Check if the gripper is likely holding an object (client-side proxy).

        Arguments mirror the server-side heuristic:
          tol_pos_closed: how close to 255 counts we consider "fully closed"
          tol_pos_open:   how close to 0 counts we consider "fully open"
          require_stopped: if True, require the gripper to be in a stopped OBJ state
          min_current:    if given, require motor current >= this threshold
        """
        res = self._rpc("is_holding", {
            "tol_pos_closed": int(tol_pos_closed),
            "tol_pos_open": int(tol_pos_open),
            "require_stopped": bool(require_stopped),
            "min_current": (int(min_current) if min_current is not None else None),
        })
        return bool(res.get("holding"))
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Robotiq client proxy (talks to remote gripper server and can run tests)"
    )
    parser.add_argument(
        "--server_host",
        default="192.168.10.100",
        help="Host/IP where the gripper server is running",
    )
    parser.add_argument(
        "--server-port",
        dest="server_port",
        type=int,
        default=5554,
        help="TCP port of the gripper server",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Socket timeout in seconds",
    )

    # What action to perform:
    parser.add_argument(
        "action",
        choices=[
            "test_basic",   # activate → open → report width → close → status
            "cycle",        # run open/close cycles
            "activate",
            "open",
            "close",
            "status",
            "width",
            "goto_frac",
            "goto_width",
            "is_closed",
            "is_holding",  
        ],
        help="Which action/test to run against the gripper server",
    )

    # Extra parameters for some actions:
    parser.add_argument(
        "--cycles",
        type=int,
        default=3,
        help="Number of open/close cycles for 'cycle' action",
    )
    parser.add_argument(
        "--wait",
        type=float,
        default=1.0,
        help="Seconds to wait between operations in 'cycle' test",
    )
    parser.add_argument(
        "--frac",
        type=float,
        default=0.5,
        help="Target fraction (0..1) for 'goto_frac'",
    )
    parser.add_argument(
        "--width_m",
        type=float,
        default=0.05,
        help="Target opening width in metres for 'goto_width'",
    )
    parser.add_argument(
        "--tol_pos",
        type=int,
        default=30,
        help="Tolerance (counts) for 'is_closed'",
    )

    args = parser.parse_args()

    with RobotiqOnURClient(
        server_host=args.server_host,
        server_port=args.server_port,
        timeout=args.timeout,
    ) as g:
        if args.action == "test_basic":
            print("[test_basic] Activating…")
            g.activate()
            print("[test_basic] Opening…")
            g.open()
            w = g.get_width_m()
            print(f"[test_basic] Width (m): {w}")
            print("[test_basic] Closing…")
            g.close()
            print("[test_basic] Status:", g.get_status())

        elif args.action == "cycle":
            # Uses the helper we added above
            g.test_cycle(cycles=args.cycles, wait=args.wait)

        elif args.action == "activate":
            print("[activate] Activating gripper…")
            g.activate()

        elif args.action == "open":
            print("[open] Opening gripper…")
            g.open()

        elif args.action == "close":
            print("[close] Closing gripper…")
            g.close()

        elif args.action == "status":
            print("[status] Gripper status:")
            print(g.get_status())

        elif args.action == "width":
            print("[width] Current opening (m):", g.get_width_m())

        elif args.action == "goto_frac":
            print(f"[goto_frac] Moving to fraction {args.frac}…")
            pos = g.go_to_fraction(args.frac)
            print(f"[goto_frac] Targeted POS={pos}")

        elif args.action == "goto_width":
            print(f"[goto_width] Moving to width {args.width_m} m…")
            pos = g.go_to_position_metres(args.width_m)
            print(f"[goto_width] Targeted POS={pos}")

        elif args.action == "is_closed":
            closed = g.is_completely_closed(tol_pos=args.tol_pos)
            print(f"[is_closed] closed={closed}")

        elif args.action == "is_holding":
            holding = g.is_holding_object(
            )
            print(
                f"[is_holding] holding={holding} "
            )


        else:
            raise SystemExit(f"Unknown action: {args.action}")
