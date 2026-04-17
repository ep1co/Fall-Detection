# alerts/sim_a7680c.py
import time

try:
    import serial
except ImportError:
    serial = None

class SimA7680CAlert:
    def __init__(self, port="/dev/serial0", baud=115200, numbers=None, ring_sec=20):
        """
        port: /dev/serial0 (UART) hoặc /dev/ttyUSBx nếu dùng USB-serial
        numbers: list số điện thoại dạng +84...
        """
        self.port = port
        self.baud = int(baud)
        self.numbers = numbers or []
        self.ring_sec = int(ring_sec)

    def _at(self, ser, cmd, wait=0.25):
        ser.write((cmd + "\r\n").encode())
        time.sleep(wait)
        return ser.read_all().decode(errors="ignore")

    def send(self, event: dict):
        if serial is None:
            print("[SIM] pyserial not installed. Skipping SIM call.")
            return
        if not self.numbers:
            print("[SIM] No phone numbers configured. Skipping.")
            return

        with serial.Serial(self.port, self.baud, timeout=1, write_timeout=1) as ser:
            # Basic init
            self._at(ser, "ATE0")        # echo off
            self._at(ser, "AT+CMEE=2")   # verbose errors
            self._at(ser, "AT")          # ping

            # Optional checks (can be removed if too slow)
            self._at(ser, "AT+CPIN?")
            self._at(ser, "AT+CSQ")
            self._at(ser, "AT+CEREG?")

            for num in self.numbers:
                print(f"[SIM] Dialing {num} ...")
                self._at(ser, f"ATD{num};")   # voice call (must end with ;)
                time.sleep(self.ring_sec)
                self._at(ser, "AT+CHUP")
                time.sleep(0.5)