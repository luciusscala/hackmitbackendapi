#!/usr/bin/env python3
"""
Simple test script for the Mentra + Suno HackMIT Backend API
"""

import requests
import json
import time
import os

# Test configuration
BASE_URL = "http://localhost:8000"
TEST_VIDEO_PATH = "test_data/flv_test.flv"  # Using the existing FLV test file

def test_health_check():
    """Test the health check endpoint"""
    print("🔍 Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"✅ Health check: {response.status_code}")
        print(f"   Response: {response.json()}")
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_upload():
    """Test video upload endpoint"""
    print("\n🔍 Testing video upload...")
    
    if not os.path.exists(TEST_VIDEO_PATH):
        print(f"❌ Test video not found: {TEST_VIDEO_PATH}")
        return None
    
    try:
        with open(TEST_VIDEO_PATH, 'rb') as f:
            files = {'file': ('test.flv', f, 'video/x-flv')}
            response = requests.post(f"{BASE_URL}/upload", files=files)
        
        print(f"✅ Upload: {response.status_code}")
        result = response.json()
        print(f"   Task ID: {result.get('task_id')}")
        print(f"   Status: {result.get('status')}")
        return result.get('task_id')
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return None

def test_generate(task_id):
    """Test music generation endpoint"""
    print(f"\n🔍 Testing music generation for task {task_id}...")
    
    try:
        response = requests.post(f"{BASE_URL}/generate/{task_id}")
        print(f"✅ Generate: {response.status_code}")
        result = response.json()
        print(f"   Message: {result.get('message')}")
        return True
    except Exception as e:
        print(f"❌ Generate failed: {e}")
        return False

def test_status(task_id):
    """Test status checking endpoint"""
    print(f"\n🔍 Testing status check for task {task_id}...")
    
    try:
        response = requests.get(f"{BASE_URL}/status/{task_id}")
        print(f"✅ Status: {response.status_code}")
        result = response.json()
        print(f"   Status: {result.get('status')}")
        print(f"   Progress: {result.get('progress')}%")
        return result
    except Exception as e:
        print(f"❌ Status check failed: {e}")
        return None

def main():
    """Run all tests"""
    print("🚀 Starting Mentra + Suno HackMIT Backend API Tests")
    print("=" * 60)
    
    # Test 1: Health check
    if not test_health_check():
        print("\n❌ Server not running. Please start with: python3 main.py")
        return
    
    # Test 2: Upload video
    task_id = test_upload()
    if not task_id:
        print("\n❌ Upload test failed. Cannot continue.")
        return
    
    # Test 3: Generate music
    if not test_generate(task_id):
        print("\n❌ Generate test failed.")
        return
    
    # Test 4: Check status (multiple times to see progress)
    print(f"\n🔍 Monitoring task {task_id} progress...")
    for i in range(3):
        time.sleep(2)
        status_result = test_status(task_id)
        if not status_result:
            break
    
    print("\n✅ All tests completed!")
    print(f"   Task ID: {task_id}")
    print(f"   Check status: GET {BASE_URL}/status/{task_id}")
    print(f"   Download: GET {BASE_URL}/download/{task_id}")

if __name__ == "__main__":
    main()
