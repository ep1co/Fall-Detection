import time
import threading

try:
    import RPi.GPIO as GPIO
except ImportError:
    GPIO = None


class ContinuousBuzzer:
    """
    Active buzzer: GPIO HIGH = beep.
    Beep pattern: ON 0.15s, OFF 0.15s, until stop().
    """

    def __init__(self, pin=23, on_sec=0.15, off_sec=0.15):
        self.pin = int(pin)
        self.on_sec = float(on_sec)
        self.off_sec = float(off_sec)

        self._stop_evt = threading.Event()
        self._thread = None
        self._running = False

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

    def _run(self):
        if not GPIO:
            # Test on log
            while not self._stop_evt.is_set():
                print("[BUZZER] beep")
                time.sleep(self.on_sec + self.off_sec)
            return

        while not self._stop_evt.is_set():
            GPIO.output(self.pin, GPIO.HIGH)
            time.sleep(self.on_sec)
            GPIO.output(self.pin, GPIO.LOW)
            time.sleep(self.off_sec)

    def cleanup(self):
        """Call when program exits."""
        self.stop()
        if GPIO:
            GPIO.cleanup(self.pin)