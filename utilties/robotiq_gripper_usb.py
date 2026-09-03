import time
import minimalmodbus
import serial
from serial.tools import list_ports


PORT = "/dev/ttyUSB0"   # change if needed
SLAVE_ID = 9
BAUDRATE = 115200
TIMEOUT = 0.5

# Robotiq Modbus register addresses
REG_WRITE_START = 1000
REG_READ_START = 2000


def pack_bytes(high_byte: int, low_byte: int) -> int:
    """Pack two 8-bit values into one 16-bit Modbus register."""
    return ((high_byte & 0xFF) << 8) | (low_byte & 0xFF)


class RobotiqGripper:
    """
    Robotiq gripper controller over Modbus RTU.

    Position:
        0   = open
        255 = closed
    """

    def __init__(self, port: str, slave_id: int = 9, baudrate: int = 115200):
        self.instrument = minimalmodbus.Instrument(port, slave_id)
        self.instrument.mode = minimalmodbus.MODE_RTU

        # Safer settings for USB-RS485 adapters
        self.instrument.clear_buffers_before_each_transaction = True
        self.instrument.close_port_after_each_call = True
        self.instrument.handle_local_echo = True
        self.instrument.debug = True  # set False after debugging

        self.instrument.serial.baudrate = baudrate
        self.instrument.serial.bytesize = 8
        self.instrument.serial.parity = serial.PARITY_NONE
        self.instrument.serial.stopbits = 1
        self.instrument.serial.timeout = TIMEOUT

    def close(self):
        try:
            if self.instrument.serial and self.instrument.serial.is_open:
                self.instrument.serial.close()
        except Exception:
            pass

    def _write_command(self, action: int, position: int, speed: int, force: int):
        """
        Write 3 registers starting at 1000:
          reg 1000: ACTION REQUEST / OPTIONS
          reg 1001: RESERVED / POSITION REQUEST
          reg 1002: SPEED / FORCE
        """
        values = [
            pack_bytes(action, 0x00),
            pack_bytes(0x00, position),
            pack_bytes(speed, force),
        ]
        self.instrument.write_registers(REG_WRITE_START, values)

    def read_status(self) -> dict:
        """
        Read 3 registers starting at 2000:
          reg 2000: gripper status / reserved
          reg 2001: fault status / pos request echo
          reg 2002: actual position / motor current
        """
        regs = self.instrument.read_registers(REG_READ_START, 3)

        status_byte = (regs[0] >> 8) & 0xFF
        fault_byte = (regs[1] >> 8) & 0xFF
        pos_echo = regs[1] & 0xFF
        position = (regs[2] >> 8) & 0xFF
        current = regs[2] & 0xFF

        gACT = status_byte & 0x01
        gGTO = (status_byte >> 3) & 0x01
        gSTA = (status_byte >> 4) & 0x03
        gOBJ = (status_byte >> 6) & 0x03

        return {
            "gACT": gACT,
            "gGTO": gGTO,
            "gSTA": gSTA,
            "gOBJ": gOBJ,
            "fault": fault_byte,
            "pos_echo": pos_echo,
            "position": position,
            "current": current,
            "raw_registers": regs,
        }

    def reset(self):
        print("Sending reset...")
        self._write_command(action=0x00, position=0x00, speed=0x00, force=0x00)
        time.sleep(0.2)

    def activate(self, timeout: float = 8.0):
        """
        Safer activation flow:
        1. Verify we can read status before writing anything.
        2. Send reset.
        3. Send activation command.
        4. Poll until activation is complete.
        """
        print("Checking communication with gripper before activation...")
        try:
            initial_status = self.read_status()
            print("Initial status:", initial_status)
        except Exception as e:
            raise RuntimeError(
                f"Cannot read gripper status before activation. "
                f"Check slave ID / baudrate / wiring / adapter settings. Error: {e}"
            )

        self.reset()

        print("Sending activation command...")
        self._write_command(action=0x01, position=0x00, speed=0x00, force=0x00)
        time.sleep(0.2)

        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                st = self.read_status()
                print("Activation poll:", st)
            except Exception as e:
                print(f"Activation poll read failed: {e}")
                time.sleep(0.1)
                continue

            if st["fault"] != 0:
                raise RuntimeError(f"Gripper fault during activation: {st}")

            if st["gACT"] == 1 and st["gSTA"] == 3:
                print("Activation complete.")
                return st

            time.sleep(0.1)

        raise TimeoutError("Gripper activation timed out")

    def move(self, position: int, speed: int = 255, force: int = 150, wait: bool = True, timeout: float = 8.0):
        position = max(0, min(255, int(position)))
        speed = max(0, min(255, int(speed)))
        force = max(0, min(255, int(force)))

        # rACT=1, rGTO=1 -> action byte 0x09
        print(f"Sending move: position={position}, speed={speed}, force={force}")
        self._write_command(action=0x09, position=position, speed=speed, force=force)

        if not wait:
            return None

        t0 = time.time()
        while time.time() - t0 < timeout:
            st = self.read_status()
            print("Move poll:", st)

            if st["fault"] != 0:
                raise RuntimeError(f"Gripper fault during move: {st}")

            # motion ended or object detected
            if st["pos_echo"] == position and st["gOBJ"] in (1, 2, 3):
                return st

            time.sleep(0.05)

        raise TimeoutError(f"Move to position {position} timed out")

    def open(self, speed: int = 255, force: int = 150, wait: bool = True):
        return self.move(position=0, speed=speed, force=force, wait=wait)

    def close_grip(self, speed: int = 255, force: int = 150, wait: bool = True):
        return self.move(position=255, speed=speed, force=force, wait=wait)


def print_ports():
    print("Available serial ports:")
    for p in list_ports.comports():
        print(f"  {p.device}  |  {p.description}")


if __name__ == "__main__":
    print_ports()

    gripper = RobotiqGripper(PORT, slave_id=SLAVE_ID, baudrate=BAUDRATE)

    try:
        print("\nTrying status read first...")
        status = gripper.read_status()
        print("Status read OK:", status)

        print("\nActivating...")
        gripper.activate()

        print("\nStatus after activation:")
        print(gripper.read_status())

        print("\nOpening...")
        print(gripper.open())

        time.sleep(1.0)

        print("\nClosing...")
        print(gripper.close_grip(force=180))

        time.sleep(1.0)

        print("\nHalf open...")
        print(gripper.move(position=128, speed=100, force=100))

    finally:
        gripper.close()