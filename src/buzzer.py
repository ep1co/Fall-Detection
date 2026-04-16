import RPi.GPIO as GPIO
import time

BUZZER_PIN = 17

GPIO.setmode(GPIO.BCM)
GPIO.setup(BUZZER_PIN, GPIO.OUT)

def alert_buzzer(duration=3):
    """Kêu liên tục trong duration giây"""
    GPIO.output(BUZZER_PIN, GPIO.HIGH)
    time.sleep(duration)
    GPIO.output(BUZZER_PIN, GPIO.LOW)

def alert_beep(times=3):
    """Kêu ngắt quãng"""
    for _ in range(times):
        GPIO.output(BUZZER_PIN, GPIO.HIGH)
        time.sleep(0.3)
        GPIO.output(BUZZER_PIN, GPIO.LOW)
        time.sleep(0.2)

# Cleanup khi thoát
GPIO.cleanup()