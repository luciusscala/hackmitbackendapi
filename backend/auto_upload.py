#!/usr/bin/env python3
"""
Auto-upload script that watches for new FLV files and uploads them to FastAPI
"""
import os
import time
import requests
import glob
from pathlib import Path

def upload_flv_file(file_path):
    """Upload FLV file to FastAPI"""
    try:
        print(f"📤 Uploading: {file_path}")
        
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f, 'video/x-flv')}
            response = requests.post('http://localhost:8000/upload', files=files)
        
        if response.status_code == 200:
            data = response.json()
            task_id = data['task_id']
            print(f"✅ Upload successful! Task ID: {task_id}")
            
            # Start processing
            process_response = requests.post(f'http://localhost:8000/generate/{task_id}')
            if process_response.status_code == 200:
                print(f"🚀 Processing started for task: {task_id}")
            else:
                print(f"❌ Failed to start processing: {process_response.text}")
                
        else:
            print(f"❌ Upload failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Error uploading {file_path}: {e}")

def watch_directory(directory):
    """Watch directory for new FLV files"""
    print(f"👀 Watching directory: {directory}")
    
    # Get initial list of files
    existing_files = set(glob.glob(os.path.join(directory, "*.flv")))
    
    while True:
        try:
            # Check for new files
            current_files = set(glob.glob(os.path.join(directory, "*.flv")))
            new_files = current_files - existing_files
            
            # Upload new files
            for file_path in new_files:
                print(f"🆕 New file detected: {file_path}")
                upload_flv_file(file_path)
                existing_files.add(file_path)
            
            # Wait before checking again
            time.sleep(2)
            
        except KeyboardInterrupt:
            print("\n⏹️  Stopping file watcher...")
            break
        except Exception as e:
            print(f"❌ Error in file watcher: {e}")
            time.sleep(5)

if __name__ == "__main__":
    # Create temp/rtmp directory if it doesn't exist
    temp_dir = Path("temp/rtmp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    print("🤖 Auto-upload script started")
    print("   - Watches temp/rtmp/ for new FLV files")
    print("   - Automatically uploads to FastAPI")
    print("   - Press Ctrl+C to stop")
    
    watch_directory(str(temp_dir))
