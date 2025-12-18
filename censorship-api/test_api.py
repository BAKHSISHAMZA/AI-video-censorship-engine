import requests
import json
import os

# 1. Setup
url = "http://localhost:8000/detect"
image_path = "put image that you want to test model's api on "

# 2. Check file exists
if not os.path.exists(image_path):
    print(f"❌ Error: Could not find image at {image_path}")
    print("Please put a .jpg file in this folder and update the 'image_path' variable.")
    exit()

# 3. Send Request (Pure Python)
print(f"📡 Sending {image_path} to API...")

try:
    with open(image_path, "rb") as f:
        # We send the raw bytes. The Server (Docker) handles the OpenCV stuff.
        files = {"file": f}
        response = requests.post(url, files=files)

    # 4. Handle Response
    if response.status_code == 200:
        data = response.json()
        print("\n✅ SUCCESS! Server Responded:")
        print(json.dumps(data, indent=2))
    else:
        print(f"\n❌ Server Error {response.status_code}:")
        print(response.text)

except Exception as e:
    print(f"\n❌ Connection Failed: {e}")
    print("Is Docker running? (docker run -p 8000:8000 censorship-api)")