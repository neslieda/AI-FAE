"""Test script to check if server is running."""
import requests
import time
import sys

print("Waiting for server to start...")
time.sleep(3)

try:
    response = requests.get("http://localhost:8000/health", timeout=5)
    print(f"✓ Server is running!")
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        try:
            print(f"Response: {response.json()}")
        except:
            print(f"Response Text: {response.text}")
    else:
        print(f"Response Text: {response.text}")
except requests.exceptions.ConnectionError:
    print("✗ Server is not running or not accessible on port 8000")
    print("Please start the server with: python api/server.py")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

