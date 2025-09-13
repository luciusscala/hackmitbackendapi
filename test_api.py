#!/usr/bin/env python3
"""
End-to-End API Test for Mentra + Suno HackMIT Backend
Tests the complete workflow: upload → generate → download
"""

import requests
import json
import time
import os
from pathlib import Path

# Test configuration
BASE_URL = "http://localhost:8000"
TEST_VIDEO_PATH = "test_data/concert.flv"  # Using the existing FLV test file

def test_health_check():
    """Test the health check endpoint"""
    print("🔍 Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"✅ Health check: {response.status_code}")
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
            files = {'file': ('concert.flv', f, 'video/x-flv')}
            response = requests.post(f"{BASE_URL}/upload", files=files)
        
        print(f"✅ Upload: {response.status_code}")
        result = response.json()
        print(f"   Task ID: {result.get('task_id')}")
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
    try:
        response = requests.get(f"{BASE_URL}/status/{task_id}")
        result = response.json()
        return result
    except Exception as e:
        print(f"❌ Status check failed: {e}")
        return None

def test_download(task_id):
    """Test video download endpoint"""
    print(f"\n🔍 Testing video download for task {task_id}...")
    
    try:
        response = requests.get(f"{BASE_URL}/download/{task_id}")
        print(f"✅ Download: {response.status_code}")
        
        if response.status_code == 200:
            # Save the downloaded video
            output_path = f"test_output_{task_id}.mp4"
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            file_size = os.path.getsize(output_path)
            print(f"   Downloaded: {output_path} ({file_size:,} bytes)")
            return output_path
        else:
            print(f"   Error: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return None

def monitor_progress(task_id, max_wait=300):
    """Monitor task progress until completion"""
    print(f"\n⏳ Monitoring task {task_id} progress...")
    print("   (This may take 2-5 minutes for music generation)")
    
    start_time = time.time()
    last_progress = -1
    
    while time.time() - start_time < max_wait:
        status_result = test_status(task_id)
        if not status_result:
            print("   ❌ Failed to get status")
            break
        
        status = status_result.get('status')
        progress = status_result.get('progress', 0)
        
        # Only print when progress changes
        if progress != last_progress:
            print(f"   📊 Status: {status} ({progress}%)")
            last_progress = progress
        
        if status == "done":
            print("   ✅ Task completed successfully!")
            return True
        elif status == "error":
            error_msg = status_result.get('error_message', 'Unknown error')
            print(f"   ❌ Task failed: {error_msg}")
            return False
        
        time.sleep(5)  # Check every 5 seconds
    
    print(f"   ⏰ Timeout after {max_wait} seconds")
    return False

def verify_output_video(video_path):
    """Verify the output video has audio"""
    print(f"\n🔍 Verifying output video: {video_path}")
    
    try:
        import subprocess
        result = subprocess.run([
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_streams', video_path
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            data = json.loads(result.stdout)
            streams = data.get('streams', [])
            
            audio_streams = [s for s in streams if s.get('codec_type') == 'audio']
            video_streams = [s for s in streams if s.get('codec_type') == 'video']
            
            print(f"   📹 Video streams: {len(video_streams)}")
            print(f"   🎵 Audio streams: {len(audio_streams)}")
            
            if audio_streams:
                audio_stream = audio_streams[0]
                codec = audio_stream.get('codec_name', 'unknown')
                duration = audio_stream.get('duration', 'unknown')
                print(f"   🎵 Audio codec: {codec}")
                print(f"   ⏱️  Audio duration: {duration} seconds")
                print("   ✅ Video has audio - Suno music successfully integrated!")
                return True
            else:
                print("   ❌ Video has no audio streams")
                return False
        else:
            print(f"   ❌ Error analyzing video: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error verifying video: {e}")
        return False

def main():
    """Run complete end-to-end test"""
    print("🚀 Mentra + Suno HackMIT Backend - End-to-End Test")
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
    
    # Test 4: Monitor progress
    if not monitor_progress(task_id):
        print("\n❌ Task did not complete successfully.")
        return
    
    # Test 5: Download video
    output_path = test_download(task_id)
    if not output_path:
        print("\n❌ Download test failed.")
        return
    
    # Test 6: Verify output
    if verify_output_video(output_path):
        print(f"\n🎉 SUCCESS! Complete end-to-end test passed!")
        print(f"   📁 Output video: {output_path}")
        print(f"   🎵 Contains Suno-generated music")
        print(f"   ⏱️  Music duration matches video length")
    else:
        print(f"\n❌ Output verification failed")
    
    print(f"\n🧹 Cleanup: Remove test file with: rm {output_path}")

if __name__ == "__main__":
    main()