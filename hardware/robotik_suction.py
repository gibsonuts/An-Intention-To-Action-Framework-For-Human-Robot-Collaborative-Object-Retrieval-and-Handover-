#!/usr/bin/env python3
import argparse
import socket
import time
from typing import Dict, Optional, Tuple


class RobotiqEPickOnUR:
    """
    Best-effort EPick helper that mirrors the direct URCap socket style used by
    robotik_gripper.py instead of sending URScript programs.

    Robotiq documents EPick using Modbus registers and URCap helper functions.
    This client maps those EPick registers onto the same ASCII GET/SET socket
    path exposed by some Robotiq URCap installations on port 63352.
    """

    def __init__(
        self,
        robot_ip: str = "192.168.10.111",
        port: int = 63352,
        timeout: float = 2.0,
        gripper_id: str = "2",
    ):
        self.ip = robot_ip
        self.port = int(port)
        self.timeout = float(timeout)
        self.gripper_id = str(gripper_id)
        self._status_reads_supported: Optional[bool] = None
        self._sid_hint: Optional[str] = None

        self.connect()

    def connect(self) -> None:
        try:
            with socket.create_connection((self.ip, self.port), timeout=self.timeout):
                pass
        except OSError as exc:
            raise ConnectionError(
                f"Failed to connect to Robotiq URCap socket at {self.ip}:{self.port}: {exc}"
            ) from exc
        self._sid_hint = self.get_sid_hint()

    def disconnect(self) -> None:
        return

    def __enter__(self) -> "RobotiqEPickOnUR":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.disconnect()

    def _exchange(self, sock: socket.socket, msg: str) -> str:
        sock.sendall((msg.strip() + "\n").encode("ascii"))
        chunks = []
        while True:
            try:
                chunk = sock.recv(1024)
            except socket.timeout:
                break
            if not chunk:
                break
            chunks.append(chunk.decode("ascii", errors="ignore"))
            sock.settimeout(min(self.timeout, 0.05))

        responses = []
        for line in "".join(chunks).splitlines():
            line = line.strip()
            if line:
                responses.append(line)
        return responses[-1] if responses else "".join(chunks).strip()

    def _send(self, msg: str, use_sid: bool = True) -> str:
        try:
            with socket.create_connection((self.ip, self.port), timeout=self.timeout) as sock:
                sock.settimeout(self.timeout)
                if use_sid and self.gripper_id:
                    self._exchange(sock, f"SET SID {self.gripper_id}")
                return self._exchange(sock, msg)
        except OSError as exc:
            raise ConnectionError(
                f"Robotiq URCap socket command failed for {self.ip}:{self.port}: {exc}"
            ) from exc

    def _get(self, key: str) -> Tuple[str, Optional[int]]:
        resp = self._send(f"GET {key}")
        idx = resp.upper().find(f"{key.upper()} ")
        if idx > 0:
            resp = resp[idx:]
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
        resp = self._send("GET SID", use_sid=False)
        idx = resp.upper().find("SID ")
        if idx > 0:
            resp = resp[idx:]
        parts = resp.split(None, 1)
        if len(parts) < 2 or parts[0].upper() != "SID":
            return None
        return parts[1]

    def _detect_status_read_support(self) -> bool:
        if self._status_reads_supported is not None:
            return self._status_reads_supported
        try:
            _, sta = self._get("STA")
            _, obj = self._get("OBJ")
            self._status_reads_supported = any(value is not None for value in (sta, obj))
        except Exception:
            self._status_reads_supported = False
        return self._status_reads_supported

    def _fallback_wait(self, timeout: float, default_wait: float) -> None:
        time.sleep(max(0.0, min(float(timeout), float(default_wait))))

    @staticmethod
    def _clamp_byte(value: int) -> int:
        return max(0, min(255, int(value)))

    @staticmethod
    def _vacuum_percent_to_request(vacuum_percent: int) -> int:
        """
        EPick manual mode uses rPR/rFR = 100 + P where vacuum is negative kPa.
        A vacuum percentage maps to approximately 100 - percentage.
        """
        vacuum_percent = max(1, min(100, int(vacuum_percent)))
        return 100 - vacuum_percent

    @staticmethod
    def _timeout_ms_to_ticks(timeout_ms: int) -> int:
        return max(0, min(255, int(round(float(timeout_ms) / 100.0))))

    def grip(
        self,
        advanced_mode: bool = False,
        minimum_vacuum: int = 40,
        maximum_vacuum: int = 60,
        timeout_ms: int = 3000,
        block: bool = True,
        timeout: float = 8.0,
    ) -> None:
        self._set("ACT", 1)
        self._set("MOD", 1 if advanced_mode else 0)
        self._set("ATR", 0)

        if advanced_mode:
            self._set("FOR", self._vacuum_percent_to_request(minimum_vacuum))
            self._set("SPE", self._timeout_ms_to_ticks(timeout_ms))
            self._set("POS", self._vacuum_percent_to_request(maximum_vacuum))
        else:
            self._set("POS", 0)

        self._set("GTO", 0)
        self._set("GTO", 1)

        if block:
            if self._detect_status_read_support():
                self._wait_action_done(timeout=timeout)
            else:
                wait_s = max(0.5, min(float(timeout), float(timeout_ms) / 1000.0))
                self._fallback_wait(timeout=timeout, default_wait=wait_s)

    def release(
        self,
        advanced_mode: bool = False,
        release_value: int = 101,
        release_delay_ms: int = 0,
        block: bool = True,
        timeout: float = 8.0,
    ) -> None:
        self._set("ACT", 1)
        self._set("MOD", 1 if advanced_mode else 0)
        self._set("ATR", 0)

        if advanced_mode:
            self._set("SPE", self._timeout_ms_to_ticks(release_delay_ms))

        self._set("POS", max(100, self._clamp_byte(release_value)))
        self._set("GTO", 0)
        self._set("GTO", 1)

        if block:
            if self._detect_status_read_support():
                self._wait_action_done(timeout=timeout)
            else:
                wait_s = max(0.3, min(float(timeout), float(release_delay_ms) / 1000.0 + 0.5))
                self._fallback_wait(timeout=timeout, default_wait=wait_s)

    def get_status(self) -> Dict[str, Optional[int]]:
        keys = ["ACT", "MOD", "GTO", "ATR", "STA", "OBJ", "POS", "PRE", "SPE", "FOR", "FLT", "COU", "VAC", "NCU", "MVA", "MIV"]
        out: Dict[str, Optional[int]] = {}
        for key in keys:
            _, value = self._get(key)
            out[key] = value
        return out

    def has_readable_status(self) -> bool:
        return self._detect_status_read_support()

    def test_basic(
        self,
        advanced_mode: bool = False,
        minimum_vacuum: int = 40,
        maximum_vacuum: int = 60,
        timeout_ms: int = 3000,
        release_value: int = 101,
        release_delay_ms: int = 0,
        wait_for_object_detected: bool = True,
        wait_for_object_released: bool = True,
        pause_s: float = 1.0,
        motion_timeout: float = 8.0,
    ) -> None:
        print("[grip] sending grip command")
        self.grip(
            advanced_mode=advanced_mode,
            minimum_vacuum=minimum_vacuum,
            maximum_vacuum=maximum_vacuum,
            timeout_ms=timeout_ms,
            block=wait_for_object_detected,
            timeout=motion_timeout,
        )
        if self.has_readable_status():
            print("[grip] status:", self.get_status())
        else:
            print("[grip] command sent")
        time.sleep(max(0.0, float(pause_s)))

        print("[release] sending release command")
        self.release(
            advanced_mode=advanced_mode,
            release_value=release_value,
            release_delay_ms=release_delay_ms,
            block=wait_for_object_released,
            timeout=motion_timeout,
        )
        if self.has_readable_status():
            print("[release] status:", self.get_status())
        else:
            print("[release] command sent")

    def _wait_action_done(self, timeout: float = 8.0) -> None:
        t0 = time.time()
        while time.time() - t0 < timeout:
            _, obj = self._get("OBJ")
            if obj in (1, 2, 3):
                return
            time.sleep(0.05)
        raise TimeoutError("EPick action timeout (OBJ did not settle).")


def _load_defaults() -> Dict[str, object]:
    defaults: Dict[str, object] = {
        "robot_ip": "192.168.10.111",
        "port": 63352,
        "timeout": 2.0,
        "gripper_id": "2",
        "advanced_mode": False,
        "minimum_vacuum": 40,
        "maximum_vacuum": 60,
        "timeout_ms": 3000,
        "wait_for_object_detected": True,
        "wait_for_object_released": True,
        "pause": 1.0,
        "release_value": 101,
        "release_delay_ms": 0,
        "motion_timeout": 8.0,
    }
    return defaults


def _build_parser() -> argparse.ArgumentParser:
    defaults = _load_defaults()

    parser = argparse.ArgumentParser(
        description="Direct Robotiq EPick test client using the URCap socket on the UR controller."
    )
    parser.add_argument(
        "action",
        choices=["test_basic", "grip", "release", "status"],
        help="Action to run against the URCap EPick socket.",
    )
    parser.add_argument("--robot-ip", default=defaults["robot_ip"], help="UR controller IP.")
    parser.add_argument(
        "--port",
        type=int,
        default=defaults["port"],
        help="URCap socket port, typically 63352.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=defaults["timeout"],
        help="Socket timeout in seconds.",
    )
    parser.add_argument(
        "--gripper-socket",
        "--id",
        default=defaults["gripper_id"],
        dest="gripper_id",
        help="Robotiq EPick id/socket as configured on the teach pendant.",
    )
    parser.add_argument(
        "--advanced-mode",
        action=argparse.BooleanOptionalAction,
        default=defaults["advanced_mode"],
        help="Use EPick advanced mode instead of automatic mode.",
    )
    parser.add_argument(
        "--minimum-vacuum",
        type=int,
        default=defaults["minimum_vacuum"],
        help="Minimum vacuum percentage for advanced mode.",
    )
    parser.add_argument(
        "--maximum-vacuum",
        type=int,
        default=defaults["maximum_vacuum"],
        help="Maximum vacuum percentage for advanced mode.",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=defaults["timeout_ms"],
        help="Grip timeout in milliseconds for advanced mode.",
    )
    parser.add_argument(
        "--release-value",
        type=int,
        default=defaults["release_value"],
        help="Release request byte: 100=passive release, 101..255=active release.",
    )
    parser.add_argument(
        "--release-delay-ms",
        type=int,
        default=defaults["release_delay_ms"],
        help="Release delay in milliseconds for advanced mode.",
    )
    parser.add_argument(
        "--wait-for-object-detected",
        action=argparse.BooleanOptionalAction,
        default=defaults["wait_for_object_detected"],
        help="Block locally after grip instead of fire-and-forget.",
    )
    parser.add_argument(
        "--wait-for-object-released",
        action=argparse.BooleanOptionalAction,
        default=defaults["wait_for_object_released"],
        help="Block locally after release instead of fire-and-forget.",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=defaults["pause"],
        help="Pause between grip and release in test_basic.",
    )
    parser.add_argument(
        "--motion-timeout",
        type=float,
        default=defaults["motion_timeout"],
        help="Timeout for grip/release actions.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if not args.robot_ip:
        parser.error("Missing --robot-ip.")

    try:
        with RobotiqEPickOnUR(
            robot_ip=args.robot_ip,
            port=args.port,
            timeout=args.timeout,
            gripper_id=args.gripper_id,
        ) as epick:
            if args.action == "test_basic":
                print(f"[connect] UR={args.robot_ip}:{args.port} id={args.gripper_id}")
                sid_hint = epick.get_sid_hint()
                if sid_hint:
                    print(f"[sid] controller reports {sid_hint}")
                epick.test_basic(
                    advanced_mode=args.advanced_mode,
                    minimum_vacuum=args.minimum_vacuum,
                    maximum_vacuum=args.maximum_vacuum,
                    timeout_ms=args.timeout_ms,
                    release_value=args.release_value,
                    release_delay_ms=args.release_delay_ms,
                    wait_for_object_detected=args.wait_for_object_detected,
                    wait_for_object_released=args.wait_for_object_released,
                    pause_s=args.pause,
                    motion_timeout=args.motion_timeout,
                )

            elif args.action == "grip":
                print(f"[grip] sending grip command to {args.robot_ip}:{args.port} id={args.gripper_id}")
                epick.grip(
                    advanced_mode=args.advanced_mode,
                    minimum_vacuum=args.minimum_vacuum,
                    maximum_vacuum=args.maximum_vacuum,
                    timeout_ms=args.timeout_ms,
                    block=args.wait_for_object_detected,
                    timeout=args.motion_timeout,
                )
                if epick.has_readable_status():
                    print(epick.get_status())
                else:
                    print("[grip] command sent")

            elif args.action == "release":
                print(f"[release] sending release command to {args.robot_ip}:{args.port} id={args.gripper_id}")
                epick.release(
                    advanced_mode=args.advanced_mode,
                    release_value=args.release_value,
                    release_delay_ms=args.release_delay_ms,
                    block=args.wait_for_object_released,
                    timeout=args.motion_timeout,
                )
                if epick.has_readable_status():
                    print(epick.get_status())
                else:
                    print("[release] command sent")

            elif args.action == "status":
                print(epick.get_status())
                sid_hint = epick.get_sid_hint()
                if sid_hint:
                    print(f"[sid] controller reports {sid_hint}")
                if not epick.has_readable_status():
                    print("[status] URCap accepted the socket connection, but this setup does not expose readable registers.")

            else:
                raise SystemExit(f"Unknown action: {args.action}")
    except (ConnectionError, RuntimeError, TimeoutError, ValueError) as exc:
        raise SystemExit(f"[ERROR] {exc}") from exc


if __name__ == "__main__":
    main()
