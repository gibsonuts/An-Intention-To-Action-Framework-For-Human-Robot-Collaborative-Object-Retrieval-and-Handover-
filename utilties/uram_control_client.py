
#############################################
# ================ ur_arm.py =============== #
#############################################

# --- Begin: ur_arm.py ---
"""Dual‑mode UR RTDE client.

- **Remote mode**: pass `--remote http://server:8000` (or `base_url` in code)
  to call the HTTP server endpoints.
- **Local mode**: pass `--host 192.168.0.2` (or `host` in code) to talk
  directly to the UR controller via `ur_rtde`.

The public methods intentionally mirror a small subset of RTDEControl/Receive
so you can reuse calling code.
"""
import argparse
from typing import List, Optional

# Remote deps
try:
    import httpx  # for remote mode
except Exception:
    httpx = None

# Local deps
try:
    from rtde_control import RTDEControlInterface as RTDEControl
    from rtde_receive import RTDEReceiveInterface as RTDEReceive
except Exception:
    RTDEControl = None  # type: ignore
    RTDEReceive = None  # type: ignore


class URArm:
    def __init__(self, host: Optional[str] = None, base_url: Optional[str] = None, timeout: float = 30.0):
        self.remote = base_url is not None or (host and host.startswith("http"))
        if self.remote:
            if base_url is None:
                base_url = host
            if httpx is None:
                raise RuntimeError("httpx not installed; pip install httpx")
            self.h = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)
        else:
            if RTDEControl is None or RTDEReceive is None:
                raise RuntimeError("ur-rtde not installed; pip install ur-rtde")
            if host is None:
                raise ValueError("Provide UR controller host for local mode")
            self.rtde_c = RTDEControl(host)
            self.rtde_r = RTDEReceive(host)

    # ---- Common helpers ----
    def _post(self, path: str, json=None):
        r = self.h.post(path, json=json)  # type: ignore
        r.raise_for_status()
        return r.json()

    def _get(self, path: str):
        r = self.h.get(path)  # type: ignore
        r.raise_for_status()
        return r.json()

    # ---- Public API (subset) ----
    def moveJ(self, q: List[float], speed: Optional[float] = None, accel: Optional[float] = None, async_: bool = False):
        if self.remote:
            return self._post("/movej", {"q": q, "speed": speed, "accel": accel, "async_": async_})
        return self.rtde_c.moveJ(q, speed or 1.0, accel or 1.0, async_)

    def moveL(self, pose: List[float], speed: Optional[float] = None, accel: Optional[float] = None, async_: bool = False):
        if self.remote:
            return self._post("/movel", {"pose": pose, "speed": speed, "accel": accel, "async_": async_})
        return self.rtde_c.moveL(pose, speed or 0.25, accel or 1.2, async_)

    def servoJ(self, target: List[float], speed=0.25, accel=0.5, time=0.0, lookahead_time=0.1, gain=300.0):
        if self.remote:
            return self._post("/servoj", {"target": target, "speed": speed, "accel": accel, "time": time, "lookahead_time": lookahead_time, "gain": gain})
        return self.rtde_c.servoJ(target, speed, accel, time, lookahead_time, gain)

    def servoL(self, target: List[float], speed=0.25, accel=0.5, time=0.0, lookahead_time=0.1, gain=300.0):
        if self.remote:
            return self._post("/servol", {"target": target, "speed": speed, "accel": accel, "time": time, "lookahead_time": lookahead_time, "gain": gain})
        return self.rtde_c.servoL(target, speed, accel, time, lookahead_time, gain)

    def speedL(self, v: List[float], a: float = 0.5, dt: float = 0.008):
        if self.remote:
            return self._post("/speedl", {"v": v, "a": a, "dt": dt})
        return self.rtde_c.speedL(v, a, dt)

    def speedStop(self):
        if self.remote:
            return self._post("/speed_stop")
        return self.rtde_c.speedStop()

    def stopScript(self):
        if self.remote:
            return self._post("/stop_script")
        return self.rtde_c.stopScript()

    def setTcp(self, tcp: List[float]):
        if self.remote:
            return self._post("/set_tcp", {"tcp": tcp})
        return self.rtde_c.setTcp(tcp)

    def setPayload(self, mass: float, cog: List[float]):
        if self.remote:
            return self._post("/set_payload", {"mass": mass, "cog": cog})
        return self.rtde_c.setPayload(mass, cog)

    def getPayload(self):
        if self.remote:
            return self._get("/get_payload")["mass"]
        return self.rtde_r.getPayload()

    def getPayloadCog(self):
        if self.remote:
            return self._get("/get_payload_cog")["cog"]
        return self.rtde_r.getPayloadCog()

    def getActualTCPPose(self):
        if self.remote:
            return self._get("/get_pose")["pose"]
        return self.rtde_r.getActualTCPPose()

    def getActualTCPForce(self):
        if self.remote:
            return self._get("/get_wrench")["wrench"]
        return self.rtde_r.getActualTCPForce()

    def isSteady(self):
        if self.remote:
            return self._get("/is_steady")["isSteady"]
        return self.rtde_c.isSteady()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Dual‑mode UR RTDE client")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--remote", help="Base URL of server, e.g. http://server:8000")
    g.add_argument("--host", help="UR controller IP for local mode")

    sub = ap.add_subparsers(dest="cmd", required=True)

    j = sub.add_parser("movej")
    j.add_argument("--q", nargs=6, type=float, required=True)
    j.add_argument("--speed", type=float)
    j.add_argument("--accel", type=float)
    j.add_argument("--async", dest="async_", action="store_true")

    l = sub.add_parser("movel")
    l.add_argument("--pose", nargs=6, type=float, required=True)
    l.add_argument("--speed", type=float)
    l.add_argument("--accel", type=float)
    l.add_argument("--async", dest="async_", action="store_true")

    sj = sub.add_parser("servoj")
    sj.add_argument("--target", nargs=6, type=float, required=True)
    sj.add_argument("--speed", type=float, default=0.25)
    sj.add_argument("--accel", type=float, default=0.5)
    sj.add_argument("--time", type=float, default=0.0)
    sj.add_argument("--lookahead-time", type=float, default=0.1)
    sj.add_argument("--gain", type=float, default=300.0)

    sl = sub.add_parser("servol")
    sl.add_argument("--target", nargs=6, type=float, required=True)
    sl.add_argument("--speed", type=float, default=0.25)
    sl.add_argument("--accel", type=float, default=0.5)
    sl.add_argument("--time", type=float, default=0.0)
    sl.add_argument("--lookahead-time", type=float, default=0.1)
    sl.add_argument("--gain", type=float, default=300.0)

    sp = sub.add_parser("speedl")
    sp.add_argument("--v", nargs=6, type=float, required=True)
    sp.add_argument("--a", type=float, default=0.5)
    sp.add_argument("--dt", type=float, default=0.008)

    ss = sub.add_parser("speed_stop")
    st = sub.add_parser("stop_script")

    tp = sub.add_parser("set_tcp")
    tp.add_argument("--tcp", nargs=6, type=float, required=True)

    pl = sub.add_parser("set_payload")
    pl.add_argument("mass", type=float)
    pl.add_argument("cog", nargs=3, type=float)

    gp = sub.add_parser("get_pose")
    gw = sub.add_parser("get_wrench")
    gp2 = sub.add_parser("get_payload")
    gp3 = sub.add_parser("get_payload_cog")
    gs = sub.add_parser("is_steady")
    args = ap.parse_args()

    if args.remote:
        arm = URArm(base_url=args.remote)
    else:
        arm = URArm(host=args.host)

    if args.cmd == "movej":
        print(arm.moveJ(args.q, speed=args.speed, accel=args.accel, async_=args.async_))
    elif args.cmd == "movel":
        print(arm.moveL(args.pose, speed=args.speed, accel=args.accel, async_=args.async_))
    elif args.cmd == "servoj":
        print(arm.servoJ(args.target, args.speed, args.accel, args.time, args.lookahead_time, args.gain))
    elif args.cmd == "servol":
        print(arm.servoL(args.target, args.speed, args.accel, args.time, args.lookahead_time, args.gain))
    elif args.cmd == "speedl":
        print(arm.speedL(args.v, args.a, args.dt))
    elif args.cmd == "speed_stop":
        print(arm.speedStop())
    elif args.cmd == "stop_script":
        print(arm.stopScript())
    elif args.cmd == "set_tcp":
        print(arm.setTcp(args.tcp))
    elif args.cmd == "set_payload":
        print(arm.setPayload(args.mass, args.cog))
    elif args.cmd == "get_pose":
        print(arm.getActualTCPPose())
    elif args.cmd == "get_wrench":
        print(arm.getActualTCPForce())
    elif args.cmd == "get_payload":
        print(arm.getPayload())
    elif args.cmd == "get_payload_cog":
        print(arm.getPayloadCog())
    elif args.cmd == "is_steady":
        print(arm.isSteady())
# --- End: ur_arm.py ---
