#!/usr/bin/env python3
import socket
import argparse
import sys


class DrillClient:
    def __init__(self, host: str, port: int, timeout: float = 2.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock = None

    def connect(self):
        try:
            self.sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
            greeting = self.sock.recv(1024).decode().strip()
            print(f"[SERVER] {greeting}")
        except Exception as e:
            print(f"[ERROR] Failed to connect: {e}")
            sys.exit(1)

    def close(self):
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
            self.sock = None

    def send(self, cmd: str):
        try:
            self.sock.sendall((cmd + "\n").encode())
            resp = self.sock.recv(1024).decode().strip()
            return resp
        except Exception as e:
            return f"[ERROR] {e}"


def main():
    parser = argparse.ArgumentParser(description="Drill Server Test Client")
    parser.add_argument("--host", default="192.168.10.100", help="Server hostname")
    parser.add_argument("--port", type=int, default=5556, help="Server port")

    args = parser.parse_args()

    client = DrillClient(args.host, args.port)
    client.connect()

    print("Connected to drill server.")
    print("Type commands: STATUS, STOP, SLOW, MEDIUM, FAST, FORWARD, REVERSE")
    print("Press Ctrl-C to exit.")

    try:
        while True:
            cmd = input("> ").strip()
            if not cmd:
                continue

            resp = client.send(cmd)
            print(resp)

    except KeyboardInterrupt:
        print("\n[CLIENT] Ctrl-C received → closing connection...")

    finally:
        client.close()
        print("[CLIENT] Disconnected cleanly.")


if __name__ == "__main__":
    main()
