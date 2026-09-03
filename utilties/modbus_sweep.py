import time
import minimalmodbus
import serial
from serial.tools import list_ports

PORT = "/dev/ttyUSB0"

# Try likely values first
SLAVE_IDS = [9, 1, 2, 10]
BAUDRATES = [115200, 19200, 38400, 57600, 9600]
PARITIES = [serial.PARITY_NONE, serial.PARITY_EVEN]
STOPBITS = [1, 2]
ECHO_OPTIONS = [False, True]

# Try both 0-based and 1-based style address assumptions
READ_TESTS = [
    ("read_2000_len3", 2000, 3),
    ("read_2001_len3", 2001, 3),
]

def print_ports():
    print("Available serial ports:")
    for p in list_ports.comports():
        print(f"  {p.device} | {p.description} | {p.hwid}")

def try_read(slave_id, baudrate, parity, stopbits, echo, address, count):
    instr = minimalmodbus.Instrument(PORT, slave_id)
    instr.mode = minimalmodbus.MODE_RTU
    instr.clear_buffers_before_each_transaction = True
    instr.close_port_after_each_call = True
    instr.handle_local_echo = echo
    instr.debug = False

    instr.serial.baudrate = baudrate
    instr.serial.bytesize = 8
    instr.serial.parity = parity
    instr.serial.stopbits = stopbits
    instr.serial.timeout = 0.5

    try:
        regs = instr.read_registers(address, count)
        return True, regs
    except Exception as e:
        return False, str(e)

if __name__ == "__main__":
    print_ports()
    print(f"\nTesting port: {PORT}\n")

    found = []

    for slave_id in SLAVE_IDS:
        for baudrate in BAUDRATES:
            for parity in PARITIES:
                for stopbits in STOPBITS:
                    for echo in ECHO_OPTIONS:
                        for label, address, count in READ_TESTS:
                            ok, result = try_read(
                                slave_id, baudrate, parity, stopbits, echo, address, count
                            )
                            print(
                                f"id={slave_id:>2} baud={baudrate:>6} parity={parity} "
                                f"stop={stopbits} echo={echo} test={label:<15} -> {result}"
                            )
                            if ok:
                                found.append({
                                    "slave_id": slave_id,
                                    "baudrate": baudrate,
                                    "parity": parity,
                                    "stopbits": stopbits,
                                    "echo": echo,
                                    "test": label,
                                    "result": result,
                                })
                            time.sleep(0.05)

    print("\n====================")
    print("WORKING COMBINATIONS")
    print("====================")
    if not found:
        print("None found.")
    else:
        for item in found:
            print(item)