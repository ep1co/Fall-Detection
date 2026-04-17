# alerts/buzzer.py
import time

try:
    from gpiozero import Buzzer
except ImportError:
    Buzzer = None

class BuzzerAlert:
    def __init__(self, pin=23, pattern=(0.2, 0.2, 0.2, 0.8)):
        """
        pattern: durations in seconds; even index = ON, odd index = OFF
        """
        self.pin = int(pin)
        self.pattern = pattern
        self._buzzer = Buzzer(self.pin) if Buzzer else None

    def send(self, event: dict):
        if not self._buzzer:
            print("[BUZZER] gpiozero not available (not on Pi?). Skipping buzzer.")
            return

        for i, d in enumerate(self.pattern):
            if i % 2 == 0:
                self._buzzer.on()
            else:
                self._buzzer.off()
            time.sleep(d)
        self._buzzer.off()