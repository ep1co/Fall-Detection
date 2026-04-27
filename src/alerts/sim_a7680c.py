import time
import threading

try:
    import serial
except ImportError:
    serial = None


class SimA7680CAlarm:
    """
    Loop action (dial -> ring -> hangup -> nghỉ -> dial...) until stop().
    stop() will send AT+CHUP to stop right away, even if the call is in progress.
    """

    def __init__(
        self,
        port="/dev/serial0",
        baud=115200,
        numbers=None,
        ring_sec=20,
        retry_pause_sec=5,
    ):
        self.port = port
        self.baud = int(baud)
        self.numbers = numbers or []
        self.ring_sec = int(ring_sec)
        self.retry_pause_sec = int(retry_pause_sec)

        self._stop_evt = threading.Event()
        self._thread = None
        self._running = False

    def start(self):
        if self._running:
            return
        if serial is None:
            print("[SIM] pyserial not installed. Skip.")
            return
        if not self.numbers:
            print("[SIM] No numbers configured. Skip.")
            return

        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._running = True

    def stop(self):
        if not self._running:
            return
        self._stop_evt.set()

        # best-effort hangup
        try:
            with serial.Serial(self.port, self.baud, timeout=1, write_timeout=1) as ser:
                self._at(ser, "AT")
                self._at(ser, "AT+CHUP", wait=0.2)
        except Exception:
            pass

        if self._thread:
            self._thread.join(timeout=2.0)
        self._running = False

    def _at(self, ser, cmd, wait=0.25):
        ser.write((cmd + "\r\n").encode())
        time.sleep(wait)
        return ser.read_all().decode(errors="ignore")

    def _run(self):
        try:
            with serial.Serial(self.port, self.baud, timeout=1, write_timeout=1) as ser:
                self._at(ser, "ATE0")
                self._at(ser, "AT+CMEE=2")
                self._at(ser, "AT")

                # vòng lặp gọi
                idx = 0
                while not self._stop_evt.is_set():
                    num = self.numbers[idx % len(self.numbers)]
                    idx += 1

                    print(f"[SIM] Dialing {num}")
                    self._at(ser, f"ATD{num};", wait=0.2)

                    # chờ đổ chuông hoặc stop
                    t0 = time.time()
                    while (time.time() - t0) < self.ring_sec and not self._stop_evt.is_set():
                        time.sleep(0.2)

                    # gác
                    self._at(ser, "AT+CHUP", wait=0.2)

                    # nghỉ trước khi gọi lại
                    t1 = time.time()
                    while (time.time() - t1) < self.retry_pause_sec and not self._stop_evt.is_set():
                        time.sleep(0.2)

        except Exception as e:
            print(f"[SIM][WARN] Alarm loop stopped: {e}")