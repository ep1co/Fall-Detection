import os
import json
import threading
import time
from datetime import datetime
from typing import Callable, Optional

try:
    import paho.mqtt.client as mqtt
except ImportError:
    mqtt = None

# HiveMQ config
HIVEMQ_HOST = os.getenv("HIVEMQ_HOST")
HIVEMQ_PORT = int(os.getenv("HIVEMQ_PORT"))
HIVEMQ_USER = os.getenv("HIVEMQ_USER")
HIVEMQ_PASS = os.getenv("HIVEMQ_PASS")

DEVICE_ID = os.getenv("DEVICE_ID")


class MQTTHandler:
    """Handle MQTT communication with HiveMQ."""
    
    def __init__(self, on_mute_callback: Optional[Callable] = None):
        """
        Initialize MQTT handler.
        
        Args:
            on_mute_callback: Callback function when mute command is received.
                            Signature: on_mute_callback(payload: dict)
        """
        self.on_mute_callback = on_mute_callback
        self.client = None
        self._running = False
        self._connect_lock = threading.Lock()
        
    def start(self):
        """Connect to HiveMQ and subscribe to mute topic."""
        if mqtt is None:
            print("[MQTT] paho-mqtt not installed. Skip.")
            return False
        
        if self._running:
            return True
        
        try:
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
            self.client.username_pw_set(HIVEMQ_USER, HIVEMQ_PASS)
            self.client.tls_set()
            
            self.client.on_connect = self._on_connect
            self.client.on_message = self._on_message
            self.client.on_disconnect = self._on_disconnect
            
            print(f"[MQTT] Connecting to {HIVEMQ_HOST}:{HIVEMQ_PORT}...")
            self.client.connect(HIVEMQ_HOST, HIVEMQ_PORT, keepalive=60)
            self.client.loop_start()
            
            self._running = True
            return True
            
        except Exception as e:
            print(f"[MQTT][WARN] Connection error: {e}")
            return False
    
    def stop(self):
        """Disconnect from HiveMQ."""
        if not self._running:
            return
        
        try:
            if self.client:
                self.client.loop_stop()
                self.client.disconnect()
            self._running = False
        except Exception as e:
            print(f"[MQTT][WARN] Disconnect error: {e}")
    
    def _on_connect(self, client, userdata, flags, rc):
        """Callback when connected to HiveMQ."""
        if rc == 0:
            print("[MQTT] Connected successfully")
            # Subscribe to mute command topic
            mute_topic = f"fall/device/{DEVICE_ID}/mute"
            client.subscribe(mute_topic)
            print(f"[MQTT] Subscribed to: {mute_topic}")
        else:
            print(f"[MQTT] Connection failed with code {rc}")
    
    def _on_message(self, client, userdata, msg):
        """Callback when message is received."""
        try:
            payload = json.loads(msg.payload.decode())
            print(f"[MQTT] Received: {msg.topic} -> {payload}")
            
            if self.on_mute_callback and "mute_sec" in payload:
                self.on_mute_callback(payload)
                
        except Exception as e:
            print(f"[MQTT][WARN] Message parse error: {e}")
    
    def _on_disconnect(self, client, userdata, rc):
        """Callback when disconnected."""
        if rc != 0:
            print(f"[MQTT] Unexpected disconnection: {rc}")
        else:
            print("[MQTT] Disconnected")
    
    def publish_status(self, state: str, event_id: str = None):
        """
        Publish device status to HiveMQ.
        
        Args:
            state: Device state (ALARMING, MUTED_30S, SAFE, etc.)
            event_id: Associated event ID (optional)
        """
        if not self._running or not self.client:
            print("[MQTT] Not connected. Skip publish.")
            return False

        now = datetime.now()
        timestamp = now.strftime("%H:%M:%S %d/%m/%Y")
        
        try:
            topic = f"fall/device/{DEVICE_ID}/status"
            payload = {
                "device_id": DEVICE_ID,
                "state": state,

                "timestamp": timestamp,
            }
            if event_id:
                payload["event_id"] = event_id
            
            self.client.publish(topic, json.dumps(payload), qos=1)
            print(f"[MQTT] Published to {topic}: {payload}")
            return True
            
        except Exception as e:
            print(f"[MQTT][WARN] Publish error: {e}")
            return False


# Global MQTT handler instance
_mqtt_handler = None


def init_mqtt(on_mute_callback: Optional[Callable] = None) -> MQTTHandler:
    """Initialize and return global MQTT handler."""
    global _mqtt_handler
    _mqtt_handler = MQTTHandler(on_mute_callback)
    _mqtt_handler.start()
    return _mqtt_handler


def get_mqtt_handler() -> Optional[MQTTHandler]:
    """Get global MQTT handler instance."""
    return _mqtt_handler


def publish_fall_alarm(event_id: str):
    """Publish fall alarm status."""
    if _mqtt_handler:
        _mqtt_handler.publish_status("ALARMING", event_id)


def publish_safe_status():
    """Publish safe status."""
    if _mqtt_handler:
        _mqtt_handler.publish_status("SAFE")


def publish_muted_status(event_id: str):
    """Publish muted status."""
    if _mqtt_handler:
        _mqtt_handler.publish_status("MUTED_30S", event_id)
