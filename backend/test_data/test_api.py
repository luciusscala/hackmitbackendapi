#!/usr/bin/env python3
"""
API Test for test_photo.jpg in test_data directory
Comprehensive test suite for the photo processing backend
"""

import requests
import time
import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Configuration
BACKEND_URL = "http://localhost:8000"
TEST_PHOTO_PATH = "test_photo.jpg"
OUTPUT_DIR = "test_outputs"

class APITester:
    def __init__(self, backend_url=BACKEND_URL):
        self.backend_url = backend_url
        self.test_results = []
        self.task_id = None
        
        # Create output directory
        Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    def log_test(self, test_name, success, message="", details=None):
        """Log test results"""
        result = {
            "test_name": test_name,
            "success": success,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }
        self.test_results.append(result)
        
        status = "✅" if success else "❌"
        print(f"{status} {test_name}: {message}")
        if details:
            for key, value in details.items():
                print(f"   {key}: {value}")
    
    def test_backend_health(self):
        """Test 1: Backend health check"""
        print("\n🔍 Test 1: Backend Health Check")
        try:
            response = requests.get(f"{self.backend_url}/", timeout=5)
            if response.status_code == 200:
                data = response.json()
                self.log_test(
                    "Backend Health Check",
                    True,
                    f"Backend is running: {data['message']}",
                    {
                        "active_tasks": data['active_tasks'],
                        "total_tasks": data['total_tasks'],
                        "max_concurrent": data['max_concurrent']
                    }
                )
                return True
            else:
                self.log_test("Backend Health Check", False, f"Status {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            self.log_test("Backend Health Check", False, f"Connection failed: {e}")
            return False
    
    def test_photo_validation(self):
        """Test 2: Photo file validation"""
        print("\n📸 Test 2: Photo File Validation")
        
        if not os.path.exists(TEST_PHOTO_PATH):
            self.log_test("Photo File Validation", False, f"Test photo not found: {TEST_PHOTO_PATH}")
            return False
        
        # Check file size
        file_size = os.path.getsize(TEST_PHOTO_PATH)
        file_size_mb = file_size / (1024 * 1024)
        
        if file_size > 10 * 1024 * 1024:  # 10MB limit
            self.log_test("Photo File Validation", False, f"File too large: {file_size_mb:.2f}MB")
            return False
        
        # Check file extension
        if not TEST_PHOTO_PATH.lower().endswith(('.jpg', '.jpeg', '.png')):
            self.log_test("Photo File Validation", False, "Invalid file extension")
            return False
        
        self.log_test(
            "Photo File Validation",
            True,
            f"Photo validated: {file_size_mb:.2f}MB",
            {"file_size_mb": round(file_size_mb, 2), "file_path": TEST_PHOTO_PATH}
        )
        return True
    
    def test_photo_upload(self):
        """Test 3: Photo upload"""
        print("\n📤 Test 3: Photo Upload")
        
        try:
            with open(TEST_PHOTO_PATH, 'rb') as f:
                files = {'file': (TEST_PHOTO_PATH, f, 'image/jpeg')}
                response = requests.post(f"{self.backend_url}/upload-photo", files=files, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                self.task_id = data['task_id']
                self.log_test(
                    "Photo Upload",
                    True,
                    "Photo uploaded successfully",
                    {
                        "task_id": self.task_id,
                        "status": data['status'],
                        "message": data['message']
                    }
                )
                return True
            else:
                self.log_test("Photo Upload", False, f"Upload failed: {response.status_code}")
                try:
                    error_data = response.json()
                    self.log_test("Photo Upload", False, f"Error details: {error_data}")
                except:
                    self.log_test("Photo Upload", False, f"Response: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Photo Upload", False, f"Request failed: {e}")
            return False
    
    def test_status_endpoint(self):
        """Test 4: Status endpoint"""
        print("\n📊 Test 4: Status Endpoint")
        
        if not self.task_id:
            self.log_test("Status Endpoint", False, "No task ID available")
            return False
        
        try:
            response = requests.get(f"{self.backend_url}/status/{self.task_id}", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                self.log_test(
                    "Status Endpoint",
                    True,
                    f"Status retrieved: {data['status']}",
                    {
                        "task_id": data['task_id'],
                        "status": data['status'],
                        "progress": data['progress'],
                        "message": data['message']
                    }
                )
                return data
            else:
                self.log_test("Status Endpoint", False, f"Status check failed: {response.status_code}")
                return None
                
        except requests.exceptions.RequestException as e:
            self.log_test("Status Endpoint", False, f"Request failed: {e}")
            return None
    
    def test_library_endpoint(self):
        """Test 5: Library endpoint"""
        print("\n📚 Test 5: Library Endpoint")
        
        try:
            response = requests.get(f"{self.backend_url}/library", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                self.log_test(
                    "Library Endpoint",
                    True,
                    f"Library retrieved: {data['total_count']} videos",
                    {
                        "total_count": data['total_count'],
                        "total_size_mb": data['total_size_mb'],
                        "videos": len(data['videos'])
                    }
                )
                return data
            else:
                self.log_test("Library Endpoint", False, f"Library check failed: {response.status_code}")
                return None
                
        except requests.exceptions.RequestException as e:
            self.log_test("Library Endpoint", False, f"Request failed: {e}")
            return None
    
    def test_specific_video_info(self):
        """Test 6: Specific video info endpoint"""
        print("\n🎬 Test 6: Specific Video Info")
        
        if not self.task_id:
            self.log_test("Specific Video Info", False, "No task ID available")
            return False
        
        try:
            response = requests.get(f"{self.backend_url}/library/{self.task_id}", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                self.log_test(
                    "Specific Video Info",
                    True,
                    f"Video info retrieved: {data['filename']}",
                    {
                        "task_id": data['task_id'],
                        "filename": data['filename'],
                        "file_size_mb": data['file_size_mb'],
                        "status": data['status']
                    }
                )
                return data
            else:
                self.log_test("Specific Video Info", False, f"Video info failed: {response.status_code}")
                return None
                
        except requests.exceptions.RequestException as e:
            self.log_test("Specific Video Info", False, f"Request failed: {e}")
            return None
    
    def monitor_processing(self, max_wait_time=300):
        """Test 7: Monitor processing progress"""
        print("\n⏳ Test 7: Monitor Processing Progress")
        
        if not self.task_id:
            self.log_test("Monitor Processing", False, "No task ID available")
            return False
        
        start_time = time.time()
        last_status = None
        status_changes = []
        
        print(f"   Monitoring task {self.task_id} (max {max_wait_time}s)")
        
        while time.time() - start_time < max_wait_time:
            status_data = self.test_status_endpoint()
            if not status_data:
                self.log_test("Monitor Processing", False, "Failed to get status")
                return False
            
            current_status = status_data['status']
            progress = status_data['progress']
            
            # Track status changes
            if current_status != last_status:
                status_changes.append({
                    "status": current_status,
                    "progress": progress,
                    "timestamp": datetime.now().isoformat()
                })
                print(f"   Status: {current_status} ({progress}%)")
                last_status = current_status
            
            if current_status == "done":
                self.log_test(
                    "Monitor Processing",
                    True,
                    f"Processing completed in {time.time() - start_time:.1f}s",
                    {
                        "duration_seconds": round(time.time() - start_time, 1),
                        "status_changes": status_changes,
                        "final_progress": progress
                    }
                )
                return True
            elif current_status == "error":
                error_msg = status_data.get('error_message', 'Unknown error')
                self.log_test("Monitor Processing", False, f"Processing failed: {error_msg}")
                return False
            
            time.sleep(5)
        
        self.log_test("Monitor Processing", False, f"Timeout after {max_wait_time}s")
        return False
    
    def test_video_download(self):
        """Test 8: Video download"""
        print("\n⬇️  Test 8: Video Download")
        
        if not self.task_id:
            self.log_test("Video Download", False, "No task ID available")
            return False
        
        try:
            response = requests.get(f"{self.backend_url}/download/{self.task_id}", timeout=60)
            
            if response.status_code == 200:
                output_path = os.path.join(OUTPUT_DIR, f"test_video_{self.task_id}.mp4")
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                
                file_size = os.path.getsize(output_path)
                file_size_mb = file_size / (1024 * 1024)
                
                self.log_test(
                    "Video Download",
                    True,
                    f"Video downloaded: {file_size_mb:.2f}MB",
                    {
                        "output_path": output_path,
                        "file_size_mb": round(file_size_mb, 2),
                        "content_type": response.headers.get('content-type', 'unknown')
                    }
                )
                return True
            else:
                self.log_test("Video Download", False, f"Download failed: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Video Download", False, f"Request failed: {e}")
            return False
    
    def test_error_handling(self):
        """Test 9: Error handling"""
        print("\n🚨 Test 9: Error Handling")
        
        # Test invalid task ID
        try:
            response = requests.get(f"{self.backend_url}/status/invalid-task-id", timeout=5)
            if response.status_code == 404:
                self.log_test("Error Handling - Invalid Task ID", True, "404 returned for invalid task")
            else:
                self.log_test("Error Handling - Invalid Task ID", False, f"Expected 404, got {response.status_code}")
        except Exception as e:
            self.log_test("Error Handling - Invalid Task ID", False, f"Request failed: {e}")
        
        # Test invalid download
        try:
            response = requests.get(f"{self.backend_url}/download/invalid-task-id", timeout=5)
            if response.status_code == 404:
                self.log_test("Error Handling - Invalid Download", True, "404 returned for invalid download")
            else:
                self.log_test("Error Handling - Invalid Download", False, f"Expected 404, got {response.status_code}")
        except Exception as e:
            self.log_test("Error Handling - Invalid Download", False, f"Request failed: {e}")
    
    def run_all_tests(self):
        """Run the complete test suite"""
        print("🧪 API Test Suite for test_photo.jpg")
        print("=" * 50)
        
        # Test sequence
        tests = [
            ("Backend Health", self.test_backend_health),
            ("Photo Validation", self.test_photo_validation),
            ("Photo Upload", self.test_photo_upload),
            ("Library Endpoint", self.test_library_endpoint),
            ("Monitor Processing", lambda: self.monitor_processing()),
            ("Specific Video Info", self.test_specific_video_info),
            ("Video Download", self.test_video_download),
            ("Error Handling", self.test_error_handling)
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            try:
                if test_func():
                    passed += 1
            except Exception as e:
                self.log_test(test_name, False, f"Test exception: {e}")
        
        # Summary
        print(f"\n📊 Test Summary")
        print("=" * 30)
        print(f"Passed: {passed}/{total}")
        print(f"Success Rate: {(passed/total)*100:.1f}%")
        
        # Save results
        results_file = os.path.join(OUTPUT_DIR, f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(results_file, 'w') as f:
            json.dump({
                "summary": {
                    "passed": passed,
                    "total": total,
                    "success_rate": (passed/total)*100,
                    "timestamp": datetime.now().isoformat()
                },
                "tests": self.test_results
            }, f, indent=2)
        
        print(f"📄 Detailed results saved to: {results_file}")
        
        return passed == total

def main():
    """Main test runner"""
    # Change to test_data directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    tester = APITester()
    
    if not os.path.exists(TEST_PHOTO_PATH):
        print(f"❌ Test photo not found: {TEST_PHOTO_PATH}")
        print("   Please ensure test_photo.jpg exists in the test_data directory")
        sys.exit(1)
    
    success = tester.run_all_tests()
    
    if success:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Check the output above for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
