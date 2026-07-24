import time
import threading

class AlertManager:
    def __init__(self, alerts, cooldown_sec=60):
        self.alerts = alerts
        self.cooldown_sec = float(cooldown_sec)
        self._last_trigger_ts = 0.0
        self._lock = threading.Lock()

    def trigger(self, event: dict):
        """Fire alerts in background, with cooldown."""
        now = time.time()
        with self._lock:
            if now - self._last_trigger_ts < self.cooldown_sec:
                return False
            self._last_trigger_ts = now

        threading.Thread(target=self._run, args=(event,), daemon=True).start()
        return True

    def _run(self, event: dict):
        for a in self.alerts:
            try:
                a.send(event)
            except Exception as e:
                print(f"[ALERT][WARN] {a.__class__.__name__} failed: {e}")