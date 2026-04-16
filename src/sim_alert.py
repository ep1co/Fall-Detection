class SimAlert:
    def __init__(self, port='/dev/ttyS0', baudrate=115200):
        self.sim = serial.Serial(port, baudrate, timeout=1)
        time.sleep(2)  # Chờ module khởi động

    def call(self, number, duration=20):
        self._send('AT')
        self._send(f'ATD{number};', wait=2)
        time.sleep(duration)
        self._send('ATH')

    def sms(self, number, message):
        self._send('AT+CMGF=1')
        self._send(f'AT+CMGS="{number}"')
        self.sim.write((message + '\x1A').encode())
        time.sleep(3)

    def _send(self, cmd, wait=1):
        self.sim.write((cmd + '\r\n').encode())
        time.sleep(wait)
        return self.sim.read(self.sim.inWaiting()).decode(errors='ignore')