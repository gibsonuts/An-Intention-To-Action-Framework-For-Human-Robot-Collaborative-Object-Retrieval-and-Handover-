#!/usr/bin/env python3
import argparse
import socket
import time
from typing import Dict, Optional, Tuple

class RobotiqOnUR:
    """
    Control a Robotiq gripper mounted on a UR robot via the Robotiq URCap socket
    server running on the UR controller.

    This direct path is appropriate when the gripper is physically connected to
    the UR5e/UR controller and the Robotiq URCap exposes port 63352.
    """

    def __init__(
        self,
        robot_ip: str,
        port: int = 63352,
        speed: int = 64,
        force: int = 125,
        max_width: float = 0.05,
        timeout: float = 2.0,
    ):
        self.ip = robot_ip
        self.port = int(port)
        self.speed = int(max(0, min(255, speed)))
        self.force = int(max(0, min(255, force)))
        self.timeout = float(timeout)
        self.max_open_m = float(max_width)
        self._status_reads_supported: Optional[bool] = None

        self.connect()

    def connect(self) -> None:
        """Probe the URCap socket once so failures happen at construction time."""
        try:
            with socket.create_connection((self.ip, self.port), timeout=self.timeout):
                return
        except OSError as exc:
            raise ConnectionError(
                f"Failed to connect to Robotiq URCap socket at {self.ip}:{self.port}: {exc}"
            ) from exc

    def disconnect(self) -> None:
        return

    def __enter__(self) -> "RobotiqOnUR":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.disconnect()

    # ----------- low-level -----------
    def _send(self, msg: str) -> str:
        """Send a single-line command and return the single-line reply."""
        try:
            with socket.create_connection((self.ip, self.port), timeout=self.timeout) as sock:
                sock.settimeout(self.timeout)
                sock.sendall((msg.strip() + "\n").encode("ascii"))
                return sock.recv(1024).decode("ascii", errors="ignore").strip()
        except OSError as exc:
            raise ConnectionError(
                f"Robotiq URCap socket command failed for {self.ip}:{self.port}: {exc}"
            ) from exc

    def _get(self, key: str) -> Tuple[str, Optional[int]]:
        """Return ``(key, value)`` where value may be int or None if unsupported."""
        resp = self._send(f"GET {key}")
        parts = resp.split()
        if len(parts) >= 2 and parts[0].upper() == key.upper():
            try:
                return parts[0], int(parts[1])
            except ValueError:
                return parts[0], None
        return key, None

    def _set(self, key: str, value: int) -> None:
        self._send(f"SET {key} {int(value)}")

    def get_sid_hint(self) -> Optional[str]:
        resp = self._send("GET SID")
        parts = resp.split(None, 1)
        if len(parts) < 2 or parts[0].upper() != "SID":
            return None
        return parts[1]

    def _detect_status_read_support(self) -> bool:
        if self._status_reads_supported is not None:
            return self._status_reads_supported
        try:
            _, sta = self._get("STA")
            _, pos = self._get("POS")
            self._status_reads_supported = any(value is not None for value in (sta, pos))
        except Exception:
            self._status_reads_supported = False
        return self._status_reads_supported

    def _fallback_wait(self, timeout: float, default_wait: float) -> None:
        time.sleep(max(0.0, min(float(timeout), float(default_wait))))

    # ----------- high-level -----------
    def deactivate(self) -> None:
        self._set("GTO", 0)
        self._set("ACT", 0)

    def activate(self, wait: bool = True, timeout: float = 8.0, reset: bool = True) -> None:
        """
        Activate the gripper and apply the configured speed/force.

        ``reset=True`` matches the common Robotiq startup flow and is more robust
        when reconnecting to an already-configured controller.
        """
        if reset:
            self.deactivate()
            time.sleep(0.2)

        self._set("ACT", 1)
        self._set("SPE", self.speed)
        self._set("FOR", self.force)
        self._set("GTO", 1)
        if wait:
            if self._detect_status_read_support():
                self._wait_until_ready(timeout=timeout)
            else:
                self._fallback_wait(timeout=timeout, default_wait=(2.0 if reset else 0.5))

    def open(self, block: bool = True, timeout: float = 8.0) -> None:
        self._set("POS", 0)
        self._set("GTO", 1)
        if block:
            if self._detect_status_read_support():
                self._wait_motion_done(timeout=timeout)
            else:
                self._fallback_wait(timeout=timeout, default_wait=1.0)

    def close(self, block: bool = True, timeout: float = 8.0) -> None:
        self._set("POS", 255)
        self._set("GTO", 1)
        if block:
            if self._detect_status_read_support():
                self._wait_motion_done(timeout=timeout)
            else:
                self._fallback_wait(timeout=timeout, default_wait=1.0)

    def is_completely_closed(
        self,
        tol_pos: int = 30,
        require_stopped: bool = True,
        min_current: Optional[int] = None,
    ) -> bool:
        pos = self.get_position_raw()
        _, obj = self._get("OBJ")
        current_ok = True
        if min_current is not None:
            cou = self.get_motor_current()
            current_ok = cou is not None and cou >= min_current

        pos_ok = pos >= (255 - int(tol_pos))
        obj_ok = (obj in (2, 3)) if require_stopped else True
        return pos_ok and obj_ok and current_ok

    def go_to_fraction(self, frac: float, block: bool = True, timeout: float = 8.0) -> int:
        f = max(0.0, min(1.0, float(frac)))
        pos = int(round(255 * f))
        self._set("POS", pos)
        self._set("GTO", 1)
        if block:
            if self._detect_status_read_support():
                self._wait_motion_done(timeout=timeout)
            else:
                self._fallback_wait(timeout=timeout, default_wait=1.0)
        return pos

    def go_to_position_metres(self, width_m: float, block: bool = True, timeout: float = 8.0) -> int:
        width_m = max(0.0, min(self.max_open_m, width_m))
        frac = 1.0 - (width_m / self.max_open_m)
        return self.go_to_fraction(frac, block=block, timeout=timeout)

    def get_position_raw(self) -> int:
        _, value = self._get("POS")
        if value is None:
            raise RuntimeError("Failed to read POS from the URCap gripper socket.")
        return value

    def get_width_m(self) -> float:
        pos = self.get_position_raw()
        return self.max_open_m * (1.0 - (pos / 255.0))

    def get_motor_current(self) -> Optional[int]:
        _, value = self._get("COU")
        return value

    def get_status(self) -> Dict[str, Optional[int]]:
        keys = ["ACT", "GTO", "STA", "OBJ", "POS", "PRE", "SPE", "FOR", "FLT", "COU"]
        out: Dict[str, Optional[int]] = {}
        for key in keys:
            _, value = self._get(key)
            out[key] = value
        return out

    def has_readable_status(self) -> bool:
        return self._detect_status_read_support()

    def test_cycle(self, cycles: int = 1, pause_s: float = 1.0, timeout: float = 8.0) -> None:
        self.activate(timeout=timeout)
        for idx in range(cycles):
            cycle_num = idx + 1
            print(f"[cycle {cycle_num}] opening")
            self.open(block=True, timeout=timeout)
            print(f"[cycle {cycle_num}] width_m={self.get_width_m():.4f}")
            time.sleep(pause_s)
            print(f"[cycle {cycle_num}] closing")
            self.close(block=True, timeout=timeout)
            print(f"[cycle {cycle_num}] width_m={self.get_width_m():.4f}")
            time.sleep(pause_s)

    # ----------- wait helpers -----------
    def _wait_until_ready(self, timeout: float = 8.0) -> None:
        t0 = time.time()
        while time.time() - t0 < timeout:
            _, sta = self._get("STA")
            if sta == 3:
                return
            time.sleep(0.05)
        raise TimeoutError("Activation timeout (STA != 3).")

    def _wait_motion_done(self, timeout: float = 8.0, eps: int = 1) -> None:
        """
        Poll POS until it stops changing or OBJ indicates a stopped/contact state.
        """
        t0 = time.time()
        last = None
        stable_count = 0
        while time.time() - t0 < timeout:
            _, obj = self._get("OBJ")
            pos = self.get_position_raw()
            if last is not None and abs(pos - last) <= eps:
                stable_count += 1
            else:
                stable_count = 0
            last = pos
            if obj in (1, 2, 3) or stable_count >= 3:
                return
            time.sleep(0.05)

def _load_config_defaults() -> Dict[str, object]:
    defaults: Dict[str, object] = {
        "robot_ip": "192.168.10.111",
        "port": 63352,
        "speed": 64,
        "force": 125,
        "max_width": 0.05,
        "timeout": 2.0,
    }
    return defaults


def _parse_args() -> argparse.Namespace:
    defaults = _load_config_defaults()
    parser = argparse.ArgumentParser(
        description="Direct Robotiq-on-UR gripper test client (URCap socket on the UR controller)."
    )
    parser.add_argument(
        "action",
        choices=[
            "test_basic",
            "cycle",
            "activate",
            "open",
            "close",
            "status",
            "width",
            "goto_frac",
            "goto_width",
            "is_closed",
        ],
        help="Action to run against the URCap gripper socket.",
    )
    parser.add_argument(
        "--robot-ip",
        default="192.168.10.111",
        help="UR controller IP hosting the Robotiq URCap socket.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=defaults["port"],
        help="URCap gripper socket port.",
    )
    parser.add_argument(
        "--speed",
        type=int,
        default=defaults["speed"],
        help="Robotiq speed command, 0..255.",
    )
    parser.add_argument(
        "--force",
        type=int,
        default=defaults["force"],
        help="Robotiq force command, 0..255.",
    )
    parser.add_argument(
        "--max-width-m",
        dest="max_width_m",
        type=float,
        default=defaults["max_width"],
        help="Maximum open width in metres used for width conversion.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=defaults["timeout"],
        help="Socket timeout in seconds.",
    )
    parser.add_argument(
        "--motion-timeout",
        type=float,
        default=8.0,
        help="Timeout for activation and open/close moves.",
    )
    parser.add_argument(
        "--frac",
        type=float,
        default=0.5,
        help="Target close fraction for 'goto_frac'.",
    )
    parser.add_argument(
        "--width-m",
        dest="width_m",
        type=float,
        default=0.02,
        help="Target opening width for 'goto_width'.",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=1,
        help="Number of open/close cycles for 'cycle'.",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=1.0,
        help="Pause between operations in 'cycle' or 'test_basic'.",
    )
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Skip the ACT/GTO reset before activation.",
    )
    args = parser.parse_args()
    return args


def main() -> None:
    args = _parse_args()
    try:
        with RobotiqOnUR(
            robot_ip=args.robot_ip,
            port=args.port,
            speed=args.speed,
            force=args.force,
            max_width=args.max_width_m,
            timeout=args.timeout,
        ) as gripper:
            if args.action == "test_basic":
                print(f"[connect] UR={args.robot_ip}:{args.port}")
                sid_hint = gripper.get_sid_hint()
                if sid_hint:
                    print(f"[sid] controller reports {sid_hint}")
                print("[activate] activating gripper")
                gripper.activate(timeout=args.motion_timeout, reset=not args.no_reset)
                if gripper.has_readable_status():
                    print("[activate] status:", gripper.get_status())
                else:
                    print("[activate] status unreadable on this URCap socket, continuing with timed waits")

                print("[open] opening gripper")
                gripper.open(block=True, timeout=args.motion_timeout)
                if gripper.has_readable_status():
                    print(f"[open] width_m={gripper.get_width_m():.4f}")
                else:
                    print("[open] command sent")
                time.sleep(args.pause)

                print("[close] closing gripper")
                gripper.close(block=True, timeout=args.motion_timeout)
                if gripper.has_readable_status():
                    print(f"[close] width_m={gripper.get_width_m():.4f}")
                    print("[close] status:", gripper.get_status())
                else:
                    print("[close] command sent")

            elif args.action == "cycle":
                gripper.test_cycle(cycles=args.cycles, pause_s=args.pause, timeout=args.motion_timeout)

            elif args.action == "activate":
                gripper.activate(timeout=args.motion_timeout, reset=not args.no_reset)
                print(gripper.get_status())

            elif args.action == "open":
                gripper.open(block=True, timeout=args.motion_timeout)
                print(gripper.get_status())

            elif args.action == "close":
                gripper.close(block=True, timeout=args.motion_timeout)
                print(gripper.get_status())

            elif args.action == "status":
                status = gripper.get_status()
                print(status)
                if not gripper.has_readable_status():
                    print("[status] URCap accepted the socket connection, but this setup does not expose readable registers.")

            elif args.action == "width":
                print(gripper.get_width_m())

            elif args.action == "goto_frac":
                pos = gripper.go_to_fraction(args.frac, block=True, timeout=args.motion_timeout)
                print({"pos": pos, "status": gripper.get_status()})

            elif args.action == "goto_width":
                pos = gripper.go_to_position_metres(args.width_m, block=True, timeout=args.motion_timeout)
                print({"pos": pos, "status": gripper.get_status()})

            elif args.action == "is_closed":
                print(gripper.is_completely_closed())

            else:
                raise SystemExit(f"Unknown action: {args.action}")
    except (ConnectionError, RuntimeError, TimeoutError) as exc:
        raise SystemExit(f"[ERROR] {exc}") from exc


if __name__ == "__main__":
    main()
