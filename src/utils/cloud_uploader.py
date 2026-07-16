import os
import time
import uuid
import requests
from typing import Optional
from datetime import datetime, timezone

# Supabase config - load from environment
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "fall-images").strip()

DEVICE_ID = os.getenv("DEVICE_ID", "pi01").strip()

# Debug: Print config status on import
print(f"[CLOUD] Config loaded:")
print(f"  SUPABASE_URL: {SUPABASE_URL[:50]}..." if SUPABASE_URL else "  SUPABASE_URL: NOT SET")
print(f"  SUPABASE_SERVICE_KEY: {'SET' if SUPABASE_SERVICE_KEY else 'NOT SET'}")
print(f"  SUPABASE_BUCKET: {SUPABASE_BUCKET}")
print(f"  DEVICE_ID: {DEVICE_ID}")


def upload_image_to_supabase(image_bytes: bytes, filename: str) -> Optional[str]:
    """
    Upload image to Supabase Storage and return the public URL.
    
    Args:
        image_bytes: Image file bytes
        filename: Filename (e.g., "event_id.jpg")
    
    Returns:
        Public URL of the image or None if upload failed
    """
    if not SUPABASE_SERVICE_KEY:
        print("[CLOUD] SUPABASE_SERVICE_KEY not set. Skip upload.")
        print("[CLOUD] Please set SUPABASE_SERVICE_KEY in .env file")
        return None
    
    if not SUPABASE_URL:
        print("[CLOUD] SUPABASE_URL not set. Skip upload.")
        return None

    try:
        headers = {
            "apikey": SUPABASE_SERVICE_KEY,
            "Content-Type": "image/jpeg",
        }
        
        url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{filename}"
        
        print(f"[CLOUD] Uploading to: {url}")
        
        response = requests.post(
            url,
            data=image_bytes,
            headers=headers,
            timeout=15,
        )
        
        if response.status_code in (200, 201):
            # Return public URL
            public_url = f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{filename}"
            print(f"[CLOUD] Image uploaded: {public_url}")
            return public_url
        else:
            print(f"[CLOUD] Upload failed: {response.status_code}")
            print(f"[CLOUD] Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"[CLOUD][WARN] Upload error: {e}")
        return None


def insert_fall_event(
    event_id: str,
    image_url: str,
    state: str = "ALARMING",
    image_path: str = None,
    event_time: float = None,
    timeout_sec: int = 10,
) -> bool:
    """
    Insert fall event record to Supabase database.
    
    Table schema:
        - event_id (text, unique)
        - device_id (text)
        - event_time (timestamptz)
        - state (text)
        - image_path (text, optional)
        - image_url (text)
        - created_at (timestamptz, auto)
    
    Args:
        event_id: Unique event ID
        image_url: URL of the uploaded image
        state: Event state (ALARMING, MUTED_30S, SAFE, etc.)
        image_path: Optional local path to the image
        event_time: Event timestamp (defaults to current time)
        timeout_sec: Request timeout
    
    Returns:
        True if successful, False otherwise
    """
    if not SUPABASE_SERVICE_KEY:
        print("[CLOUD] SUPABASE_SERVICE_KEY not set. Skip insert.")
        return False

    try:
        headers = {
            "apikey": SUPABASE_SERVICE_KEY,
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        }
        
        url = f"{SUPABASE_URL}/rest/v1/fall_events"
        
        # Use current time if not provided
        if event_time is None:
            event_time = time.time()
        
        # Convert timestamp to ISO format for PostgreSQL
        event_datetime = datetime.fromtimestamp(event_time, tz=timezone.utc).isoformat()
        
        payload = {
            "event_id": event_id,
            "device_id": DEVICE_ID,
            "event_time": event_datetime,
            "state": state,
            "image_url": image_url,
        }
        
        # Add optional fields if provided
        if image_path:
            payload["image_path"] = image_path
        
        response = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=timeout_sec,
        )
        
        if response.status_code in (200, 201):
            print(f"[CLOUD] Event recorded: {event_id} - {state}")
            return True
        else:
            print(f"[CLOUD] Insert failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"[CLOUD][WARN] Insert error: {e}")
        return False


def gen_event_id() -> str:
    """Generate unique event ID."""
    return str(uuid.uuid4())


def post_fall_event(
    url: str,
    device_id: str,
    device_key: str,
    image_bytes: bytes,
    fall_prob: float = None,
    ts_ms: int = None,
    timeout_sec: int = 15,
):
    """
    Legacy function - Send multipart/form-data to custom endpoint.
    """
    if ts_ms is None:
        ts_ms = int(time.time() * 1000)

    data = {
        "deviceId": device_id,
        "tsMs": str(ts_ms),
    }
    if fall_prob is not None:
        data["fallProb"] = f"{fall_prob:.4f}"

    files = {
        "image": ("snapshot.jpg", image_bytes, "image/jpeg"),
    }

    headers = {
        "X-Device-Key": device_key,
    }

    r = requests.post(url, data=data, files=files, headers=headers, timeout=timeout_sec)
    r.raise_for_status()
    return r.json()