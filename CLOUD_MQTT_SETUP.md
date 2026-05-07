# AI Fall Detection - Cloud & MQTT Integration Setup

## Overview

This system integrates:
1. **Supabase** - Cloud storage for images and database for events
2. **HiveMQ** - MQTT broker for real-time communication and mute commands
3. **Buzzer Mute** - 30-second cooldown when mute command is received via MQTT

## System Architecture

```
Raspberry Pi
    ↓
Fall Detection (run_realtime.py)
    ↓
├─→ [ALARMING] 
│   ├─→ Capture image & upload to Supabase Storage
│   ├─→ Insert event to Supabase Database
│   ├─→ Publish "ALARMING" status via MQTT
│   ├─→ Start buzzer & call alarm
│   └─→ Subscribe to mute commands (MQTT)
│
├─→ [MUTED_30S] (when receiving mute command)
│   ├─→ Buzzer stops for 30 seconds
│   ├─→ Publish "MUTED_30S" status via MQTT
│   └─→ Alarm calls continue
│
└─→ [SAFE/NORMAL]
    └─→ Publish "SAFE" status via MQTT
```

## Prerequisites

### 1. Install Required Packages

```bash
pip install paho-mqtt requests python-dotenv
```

### 2. Environment Variables

Copy `.env.example` to `.env` and configure with your credentials:

```bash
cp .env.example .env
```

Then edit `.env`:
```
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_KEY=your_service_key
SUPABASE_BUCKET=fall-images

HIVEMQ_HOST=your_hivemq_host
HIVEMQ_PORT=8883
HIVEMQ_USER=your_username
HIVEMQ_PASS=your_password

DEVICE_ID=pi01
```

### 3. Supabase Setup

**Create the `fall_events` table in Supabase:**

```sql
CREATE TABLE public.fall_events (
  id BIGSERIAL PRIMARY KEY,
  event_id TEXT UNIQUE NOT NULL,
  device_id TEXT NOT NULL,
  event_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  state TEXT NOT NULL,
  image_path TEXT,
  image_url TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

**Create the storage bucket:**
- In Supabase Console → Storage → Create bucket
- Bucket name: `fall-images`
- Set to public

### 4. HiveMQ Setup

- Create account at [HiveMQ Cloud](https://www.hivemq.com/cloud/)
- Create cluster with username and password
- Copy connection details to `.env`

## How It Works

### Fall Detection Workflow

1. **NORMAL → FALL_SUSPECT → ALARMING**
   - When fall is confirmed (state machine):
     - Generate unique `event_id`
     - Capture current frame
     - Start buzzer & phone call alarm
     - **In background thread:**
       - Compress and upload image to Supabase Storage
       - Insert event record to Supabase Database
       - Publish "ALARMING" status via MQTT

2. **While ALARMING (Buzzer continues ringing)**
   - Subscribe to MQTT topic: `fall/device/{DEVICE_ID}/mute`
   - Wait for mute command with format:
     ```json
     {
       "mute_sec": 30,
       "event_id": "uuid-here"
     }
     ```

3. **Mute Command Received**
   - Buzzer pauses for 30 seconds (via `buzzer.mute(30)`)
   - Phone calls continue
   - Publish "MUTED_30S" status via MQTT
   - After 30s, buzzer automatically resumes

4. **Recovery (Person stands up)**
   - Transition: ALARMING → RECOVERING → NORMAL
   - Stop all alarms (buzzer & calls)
   - Publish "SAFE" status via MQTT

### MQTT Topics

**Publish (Pi sends):**
- `fall/device/{DEVICE_ID}/status` - Device state updates

**Subscribe (Pi listens):**
- `fall/device/{DEVICE_ID}/mute` - Mute command with payload:
  ```json
  {
    "mute_sec": 30,
    "event_id": "event-id-here"
  }
  ```

### Files Modified/Created

1. **`src/utils/cloud_uploader.py`** (Updated)
   - `upload_image_to_supabase()` - Upload image to Supabase Storage
   - `insert_fall_event()` - Insert event record to database
   - `gen_event_id()` - Generate unique event ID

2. **`src/utils/mqtt_handler.py`** (New)
   - `MQTTHandler` class - Handle HiveMQ connection & communication
   - `publish_status()` - Publish device state
   - Global helpers for easy access

3. **`src/alerts/buzzer.py`** (Updated)
   - `mute(seconds)` - Mute buzzer for specified duration
   - `_is_muted()` - Check mute status
   - Thread-safe mute mechanism

4. **`src/scripts/run_realtime.py`** (Updated)
   - Integrated Supabase upload
   - Integrated MQTT communication
   - Async upload to prevent blocking realtime video
   - Mute command handling

## Testing

### 1. Test Supabase Upload

```python
from utils.cloud_uploader import upload_image_to_supabase, insert_fall_event, gen_event_id
import cv2
import time

# Test image upload
image = cv2.imread("test.jpg")
_, jpg_bytes = cv2.imencode(".jpg", image)

event_id = gen_event_id()
filename = f"{event_id}.jpg"
url = upload_image_to_supabase(jpg_bytes.tobytes(), filename)
print(f"Image URL: {url}")

# Test database insert
insert_fall_event(
    event_id=event_id,
    image_url=url,
    state="ALARMING",
    image_path=filename,
    event_time=time.time()
)
```

### 2. Test MQTT Connection

```python
from utils.mqtt_handler import init_mqtt, publish_fall_alarm

# Initialize MQTT
mqtt = init_mqtt()

# Publish test status
publish_fall_alarm("test-event-id")

# Wait for message
import time
time.sleep(2)

# Check console for "Connected" message
```

### 3. Test Mute Command

Use MQTT client to publish to `fall/device/{DEVICE_ID}/mute`:

```json
{
  "mute_sec": 30,
  "event_id": "test-event"
}
```

Buzzer should stop for 30 seconds then resume.

## Troubleshooting

### Supabase Upload Fails
- Check `SUPABASE_SERVICE_KEY` is correct
- Verify bucket exists and is public
- Check network connectivity

### MQTT Connection Fails
- Verify `HIVEMQ_HOST`, `HIVEMQ_PORT`, `HIVEMQ_USER`, `HIVEMQ_PASS`
- Check internet connection on Pi
- Port 8883 must be accessible (TLS)

### Buzzer Mute Not Working
- Check MQTT is connected
- Verify payload format includes `mute_sec`
- Check topic: `fall/device/{DEVICE_ID}/mute`

### Upload Blocks Realtime
- Already handled with background thread
- Check disk space on Pi for image encoding
- May be slow on poor network - check Pi internet speed

## Performance Notes

- **Image Upload**: ~2-5 seconds per image (depends on network)
- **Database Insert**: ~1 second
- **Upload runs in background**: Doesn't block video processing
- **Buzzer Mute**: Immediate (thread-safe)
- **MQTT Latency**: <100ms typically

## Next Steps

1. Deploy to Raspberry Pi
2. Test fall detection with real scenarios
3. Monitor Supabase for events and images
4. Set up frontend/mobile app to consume MQTT events
5. Configure auto-recovery if network disconnects

## References

- [Supabase REST API](https://supabase.com/docs/guides/api)
- [Paho MQTT Python](https://github.com/eclipse/paho.mqtt.python)
- [HiveMQ Cloud Docs](https://docs.hivemq.cloud/)
