import time
import threading
from datetime import datetime

try:
    import serial
except ImportError:
    serial = None


class SimA7680CAlarm:
    """
    Loop action: dial -> wait ring_sec -> if no answer, send SMS -> next number.
    stop() will hangup immediately via AT+CHUP.
    """

    def __init__(
        self,
        port="/dev/serial0",
        baud=115200,
        numbers=None,
        ring_sec=15,
        retry_pause_sec=5,
        send_sms_after_first_call=True,
        sms_text="Canh bao: Phat hien te nga!",
    ):
        self.port = port
        self.baud = int(baud)
        self.numbers = numbers or []
        self.ring_sec = int(ring_sec)
        self.retry_pause_sec = int(retry_pause_sec)
        self.send_sms_after_call = send_sms_after_first_call
        self.sms_text = sms_text

        self._stop_evt = threading.Event()
        self._thread = None
        self._running = False
        self._serial_lock = threading.Lock()

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

        try:
            with serial.Serial(self.port, self.baud, timeout=1, write_timeout=1) as ser:
                self._at(ser, "AT+CHUP", wait=0.1)
        except Exception:
            pass

        if self._thread:
            self._thread.join(timeout=2.0)
        self._running = False

    def _at(self, ser, cmd, wait=0.1):
        """Send AT command and read response without delays for unnecessary checks."""
        ser.write((cmd + "\r\n").encode())
        time.sleep(wait)
        return ser.read_all().decode(errors="ignore")

    def _send_sms(self, ser, phone):
        """Send SMS with timestamp."""
        now = datetime.now()
        timestamp = now.strftime("%H:%M:%S %d/%m/%Y")
        message = f"{self.sms_text} Thoi gian: {timestamp}"

        try:
            self._at(ser, 'AT+CSCS="GSM"', wait=0.2)
            self._at(ser, "AT+CMGF=1", wait=0.2)
            self._at(ser, f'AT+CMGS="{phone}"', wait=0.3)
            ser.write((message + "\x1A").encode())
            time.sleep(0.5)
            response = ser.read_all().decode(errors="ignore")
            if "OK" in response or "+CMGS" in response:
                print(f"[SIM] SMS sent to {phone}: {message}")
                return True
        except Exception as e:
            print(f"[SIM][WARN] SMS send failed: {e}")
        return False

    def send_sms_to_all_once(self):
        """Send SMS to all numbers once (non-blocking, runs in background thread)."""
        if serial is None:
            print("[SIM] pyserial not installed. Skip SMS.")
            return
        if not self.numbers:
            print("[SIM] No numbers configured. Skip SMS.")
            return

        def _send_thread():
            with self._serial_lock:
                try:
                    with serial.Serial(self.port, self.baud, timeout=1, write_timeout=1) as ser:
                        self._at(ser, "ATE0", wait=0.1)
                        self._at(ser, "AT+CMEE=2", wait=0.1)
                        for phone in self.numbers:
                            self._send_sms(ser, phone)
                            time.sleep(0.5)
                except Exception as e:
                    print(f"[SIM][WARN] SMS sending failed: {e}")

        sms_thread = threading.Thread(target=_send_thread, daemon=True)
        sms_thread.start()

    def _run(self):
        try:
            # Initialize serial once, outside the loop
            with self._serial_lock:
                ser = serial.Serial(self.port, self.baud, timeout=1, write_timeout=1)
                self._at(ser, "ATE0", wait=0.1)
                self._at(ser, "AT+CMEE=2", wait=0.1)
            
            # call loop
            idx = 0
            while not self._stop_evt.is_set():
                num = self.numbers[idx % len(self.numbers)]

                with self._serial_lock:
                    print(f"[SIM] Calling {num}...")
                    self._at(ser, f"ATD{num};", wait=0.1)

                # Wait for ring or stop (without holding lock)
                t0 = time.time()
                while (time.time() - t0) < self.ring_sec and not self._stop_evt.is_set():
                    time.sleep(0.1)

                with self._serial_lock:
                    # Hangup
                    self._at(ser, "AT+CHUP", wait=0.1)

                # Move to next number
                idx += 1

                # Brief pause before next call (without holding lock)
                t1 = time.time()
                while (time.time() - t1) < self.retry_pause_sec and not self._stop_evt.is_set():
                    time.sleep(0.1)
            
            # Cleanup
            with self._serial_lock:
                ser.close()

        except Exception as e:
            print(f"[SIM][WARN] Alarm stopped: {e}")