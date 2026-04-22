import time
import requests

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
    Send multipart/form-data:
      fields: deviceId, tsMs, fallProb
      file: image (jpg)
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
        "X-Device-Key": device_key,  # MVP auth
    }

    r = requests.post(url, data=data, files=files, headers=headers, timeout=timeout_sec)
    r.raise_for_status()
    return r.json()  # server trả {eventId, imageUrl, ...} nếu bạn muốn