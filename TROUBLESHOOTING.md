# Troubleshooting: Supabase Upload Error

## Error: "Invalid Credentials. Check your API key"

This error occurs when the Supabase authentication fails, typically because:

### 1. **Missing or Wrong `.env` File**

**Check:**
```bash
ls -la /home/hank/Thesis/Fall-Detection/.env
```

**Should exist at project root:**
```
/home/hank/Thesis/Fall-Detection/.env
```

**Create from example:**
```bash
cd /home/hank/Thesis/Fall-Detection
cp .env.example .env
```

---

### 2. **Wrong SUPABASE_SERVICE_KEY**

The service key must be the **SERVICE KEY**, not the anonymous key!

**How to get the correct key:**

1. Open [Supabase Dashboard](https://app.supabase.com)
2. Go to your project
3. Navigate to **Settings** → **API**
4. Look for **Project API keys** section
5. Copy the key that starts with `sb_` (typically labeled "service_role")

**Correct format:**
- Starts with `sb_`
- Example: `sb_publishable_U1vnPq4BreHGcBlLlGlv-w_AWIPj83A`

**Wrong format:**
- Starts with `eyJ` (this is the anonymous key - won't work for upload!)
- Just random characters

**In `.env` file:**
```
SUPABASE_SERVICE_KEY=sb_your_actual_key_here
```

---

### 3. **Bucket Not Public**

The Supabase storage bucket must be **public** to generate public URLs.

**Fix:**

1. Go to Supabase → Storage
2. Click on `fall-images` bucket
3. Click **Edit bucket**
4. Check "Make it public"
5. Save

---

### 4. **Wrong SUPABASE_URL**

**Check your URL in `.env`:**
```bash
grep SUPABASE_URL /home/hank/Thesis/Fall-Detection/.env
```

**Format should be:**
```
SUPABASE_URL=https://your-project-id.supabase.co
```

**Find your actual URL:**
1. Go to Supabase Dashboard
2. Click your project
3. Look in Settings → API at the top - you'll see your URL

---

### 5. **Verify Configuration**

**Run the debug script:**
```bash
cd /home/hank/Thesis/Fall-Detection
python debug_config.py
```

**What it checks:**
- ✓ Environment variables are loaded
- ✓ .env file exists
- ✓ Credentials format is correct
- ✓ Can connect to Supabase
- ✓ Can connect to HiveMQ

**Expected output:**
```
Loading .env from: /home/hank/Thesis/Fall-Detection/.env
.env exists: True

============================================================
SUPABASE CONFIGURATION
============================================================
✓ SUPABASE_URL: https://lkchehvwekadkpbneqbb.supabase.co
✓ SUPABASE_SERVICE_KEY: SET (45 chars)
  └─ ✓ Key format looks correct (starts with 'sb_')
✓ SUPABASE_BUCKET: fall-images

✓ All required configuration is set!
```

---

## Step-by-Step Fix

1. **Backup existing .env (if any):**
   ```bash
   cp /home/hank/Thesis/Fall-Detection/.env /home/hank/Thesis/Fall-Detection/.env.backup
   ```

2. **Create fresh .env:**
   ```bash
   cd /home/hank/Thesis/Fall-Detection
   cp .env.example .env
   ```

3. **Edit .env with your actual credentials:**
   ```bash
   nano .env
   ```
   
   Replace these lines with your actual values:
   ```
   SUPABASE_URL=https://your-project-id.supabase.co
   SUPABASE_SERVICE_KEY=sb_your_actual_service_key
   HIVEMQ_HOST=your-cluster.eu.hivemq.cloud
   HIVEMQ_USER=your_username
   HIVEMQ_PASS=your_password
   DEVICE_ID=pi01
   ```

4. **Verify configuration:**
   ```bash
   python debug_config.py
   ```

5. **Run the application:**
   ```bash
   cd /home/hank/Thesis/Fall-Detection/src/scripts
   python run_realtime.py
   ```

---

## Common Issues

### Issue: `.env` file not found
**Solution:** Run `debug_config.py` - it will show the exact path expected

### Issue: "SUPABASE_SERVICE_KEY not set"
**Solution:** 
- Check file exists: `cat /home/hank/Thesis/Fall-Detection/.env`
- Check variable is there: `grep SUPABASE_SERVICE_KEY /home/hank/Thesis/Fall-Detection/.env`
- Make sure no spaces around `=`: `SUPABASE_SERVICE_KEY=sb_abc123`

### Issue: "Upload failed: 403"
**Solution:** 
- Verify you're using SERVICE KEY (not anonymous key)
- Check bucket is set to PUBLIC
- Run debug script to verify credentials

### Issue: "Upload failed: 404"
**Solution:**
- Check bucket exists: Go to Supabase → Storage
- Check bucket name in .env matches: `SUPABASE_BUCKET=fall-images`
- Create bucket if missing: Storage → New bucket → "fall-images"

---

## Getting Your Actual Credentials

### Supabase Service Key

1. Go to https://app.supabase.com
2. Click your project
3. Settings (gear icon) → API
4. Under "Project API keys":
   - Copy "service_role" key (starts with `sb_`)
   - Paste into `.env` as `SUPABASE_SERVICE_KEY`

### HiveMQ Credentials

1. Go to https://console.hivemq.cloud
2. Click your cluster
3. Cluster Details → Connection:
   - Host: `something.eu.hivemq.cloud`
   - Port: `8883`
4. Authentication Tab:
   - Username: your created username
   - Password: your created password

---

## Debug Output Examples

### If Supabase URL is wrong:
```
❌ Cannot reach Supabase URL: https://wrong-url.com
```

### If Service Key is wrong:
```
❌ Supabase returned: 401
   Response: {"statusCode":401,"error":"Unauthorized","message":"Invalid API key"}
```

### If all is correct:
```
✓ Supabase URL is accessible
✓ Authentication header accepted (200)
✓ Connected to HiveMQ successfully

✓ All required configuration is set!
You can now run: python src/scripts/run_realtime.py
```

---

## Still Having Issues?

1. **Run debug script:** `python debug_config.py`
2. **Share the output** - it will show exactly what's wrong
3. **Check .env file:** `cat /home/hank/Thesis/Fall-Detection/.env`
4. **Verify credentials on Supabase/HiveMQ websites**
5. **Check network:** `ping -c 1 lkchehvwekadkpbneqbb.supabase.co`
