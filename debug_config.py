#!/usr/bin/env python3
"""
Debug script to verify Supabase and MQTT configuration.
Run this to check if all credentials are set correctly.
"""

import os
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(src_path))

from dotenv import load_dotenv

# Load .env
env_path = Path(__file__).resolve().parent / ".env"
print(f"Loading .env from: {env_path}")
print(f".env exists: {env_path.exists()}\n")

load_dotenv(env_path)

# ====== SUPABASE CONFIG ======
print("=" * 60)
print("SUPABASE CONFIGURATION")
print("=" * 60)

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "").strip()
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "fall-images").strip()

print(f"✓ SUPABASE_URL: {SUPABASE_URL if SUPABASE_URL else '❌ NOT SET'}")
print(f"✓ SUPABASE_SERVICE_KEY: {'SET (' + str(len(SUPABASE_SERVICE_KEY)) + ' chars)' if SUPABASE_SERVICE_KEY else '❌ NOT SET'}")
print(f"✓ SUPABASE_BUCKET: {SUPABASE_BUCKET}")

if SUPABASE_SERVICE_KEY:
    # Verify it looks like a Supabase service key (should contain 'eyJ' at start if JWT)
    if SUPABASE_SERVICE_KEY.startswith("sb_"):
        print("  └─ ✓ Key format looks correct (starts with 'sb_')")
    elif SUPABASE_SERVICE_KEY.startswith("eyJ"):
        print("  └─ ✓ Key format looks correct (JWT format)")
    else:
        print(f"  └─ ⚠ Key format unexpected (starts with '{SUPABASE_SERVICE_KEY[:10]}...')")

# ====== HIVEMQ CONFIG ======
print("\n" + "=" * 60)
print("HIVEMQ CONFIGURATION")
print("=" * 60)

HIVEMQ_HOST = os.getenv("HIVEMQ_HOST", "").strip()
HIVEMQ_PORT = os.getenv("HIVEMQ_PORT", "8883").strip()
HIVEMQ_USER = os.getenv("HIVEMQ_USER", "").strip()
HIVEMQ_PASS = os.getenv("HIVEMQ_PASS", "").strip()

print(f"✓ HIVEMQ_HOST: {HIVEMQ_HOST if HIVEMQ_HOST else '❌ NOT SET'}")
print(f"✓ HIVEMQ_PORT: {HIVEMQ_PORT}")
print(f"✓ HIVEMQ_USER: {HIVEMQ_USER if HIVEMQ_USER else '❌ NOT SET'}")
print(f"✓ HIVEMQ_PASS: {'SET (' + str(len(HIVEMQ_PASS)) + ' chars)' if HIVEMQ_PASS else '❌ NOT SET'}")

# ====== DEVICE CONFIG ======
print("\n" + "=" * 60)
print("DEVICE CONFIGURATION")
print("=" * 60)

DEVICE_ID = os.getenv("DEVICE_ID", "pi01").strip()
print(f"✓ DEVICE_ID: {DEVICE_ID}")

# ====== TEST SUPABASE ======
print("\n" + "=" * 60)
print("TEST: Supabase Connection")
print("=" * 60)

if not SUPABASE_SERVICE_KEY or not SUPABASE_URL:
    print("❌ Cannot test: SUPABASE_URL or SUPABASE_SERVICE_KEY not set")
else:
    try:
        import requests
        
        # Test if we can connect to Supabase
        headers = {
            "apikey": SUPABASE_SERVICE_KEY,
            "Content-Type": "application/json",
        }
        
        # Try a simple GET to verify credentials
        url = f"{SUPABASE_URL}/rest/v1/"
        response = requests.get(url, headers=headers, timeout=5)
        
        if response.status_code in (200, 404):
            print(f"✓ Supabase URL is accessible")
            print(f"✓ Authentication header accepted ({response.status_code})")
        else:
            print(f"❌ Supabase returned: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot reach Supabase URL: {SUPABASE_URL}")
    except Exception as e:
        print(f"❌ Error: {e}")

# ====== TEST MQTT ======
print("\n" + "=" * 60)
print("TEST: HiveMQ Connection")
print("=" * 60)

if not HIVEMQ_HOST or not HIVEMQ_USER or not HIVEMQ_PASS:
    print("❌ Cannot test: HIVEMQ credentials not set")
else:
    try:
        import paho.mqtt.client as mqtt
        
        print(f"Connecting to {HIVEMQ_HOST}:{HIVEMQ_PORT}...")
        
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
        client.username_pw_set(HIVEMQ_USER, HIVEMQ_PASS)
        client.tls_set()
        
        try:
            client.connect(HIVEMQ_HOST, int(HIVEMQ_PORT), keepalive=5)
            client.loop_start()
            
            import time
            time.sleep(2)  # Wait for connection
            
            if client.is_connected():
                print(f"✓ Connected to HiveMQ successfully")
            else:
                print(f"⚠ Connection attempt completed but status unclear")
            
            client.loop_stop()
            client.disconnect()
            
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            
    except ImportError:
        print("❌ paho-mqtt not installed. Install with: pip install paho-mqtt")
    except Exception as e:
        print(f"❌ Error: {e}")

# ====== SUMMARY ======
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

required_ok = all([
    SUPABASE_URL,
    SUPABASE_SERVICE_KEY,
    HIVEMQ_HOST,
    HIVEMQ_USER,
    HIVEMQ_PASS,
])

if required_ok:
    print("✓ All required configuration is set!")
    print("\nYou can now run: python src/scripts/run_realtime.py")
else:
    print("❌ Some configuration is missing. Please check .env file:")
    print(f"   {env_path}")
    print("\nMissing or invalid:")
    if not SUPABASE_URL:
        print("  - SUPABASE_URL")
    if not SUPABASE_SERVICE_KEY:
        print("  - SUPABASE_SERVICE_KEY")
    if not HIVEMQ_HOST:
        print("  - HIVEMQ_HOST")
    if not HIVEMQ_USER:
        print("  - HIVEMQ_USER")
    if not HIVEMQ_PASS:
        print("  - HIVEMQ_PASS")

print()
