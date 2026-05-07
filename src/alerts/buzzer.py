# alerts/buzzer.py
import time
import threading

try:
    import RPi.GPIO as GPIO
except ImportError:
    GPIO = None


class ContinuousBuzzer:
    """
    Active buzzer: GPIO HIGH = kêu.
    Beep pattern: ON 0.15s, OFF 0.15s, lặp liên tục cho tới khi stop().
    Supports mute with cooldown.
    """

    def __init__(self, pin=23, on_sec=0.15, off_sec=0.15):
        self.pin = int(pin)
        self.on_sec = float(on_sec)
        self.off_sec = float(off_sec)

        self._stop_evt = threading.Event()
        self._mute_until = None  # Timestamp when mute ends
        self._thread = None
        self._running = False
        self._mute_lock = threading.Lock()

        if GPIO:
            GPIO.setwarnings(False)
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.pin, GPIO.OUT, initial=GPIO.LOW)

    def start(self):
        """Start beeping in background (non-blocking)."""
        if self._running:
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._running = True

    def stop(self):
        """Stop beeping and set GPIO LOW."""
        if not self._running:
            return
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        self._running = False
        if GPIO:
            GPIO.output(self.pin, GPIO.LOW)

    def mute(self, seconds: float):
        """
        Mute buzzer for specified duration.
        
        Args:
            seconds: Duration in seconds to mute (e.g., 30)
        """
        with self._mute_lock:
            self._mute_until = time.time() + float(seconds)
            print(f"[BUZZER] Muted for {seconds}s")

    def _is_muted(self) -> bool:
        """Check if buzzer is currently muted."""
        with self._mute_lock:
            if self._mute_until is None:
                return False
            if time.time() < self._mute_until:
                return True
            # Mute duration expired
            self._mute_until = None
            return False

    def _run(self):
        if not GPIO:
            # Cho phép test trên PC: chỉ log
            while not self._stop_evt.is_set():
                if not self._is_muted():
                    print("[BUZZER] beep")
                time.sleep(self.on_sec + self.off_sec)
            return

        while not self._stop_evt.is_set():
            if not self._is_muted():
                GPIO.output(self.pin, GPIO.HIGH)
                time.sleep(self.on_sec)
                GPIO.output(self.pin, GPIO.LOW)
                time.sleep(self.off_sec)
            else:
                time.sleep(0.1)  # Check mute status frequently

    def cleanup(self):
        """Call when program exits (optional)."""
        self.stop()
        if GPIO:
            GPIO.cleanup(self.pin)