#!/usr/bin/env python3
"""
Network test script for RTMP setup
"""
import socket
import subprocess
import platform
import os

def get_local_ip():
    """Get local IP address"""
    try:
        # Connect to a remote address to get local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except:
        return "localhost"

def test_port(ip, port):
    """Test if port is accessible"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(3)
        result = s.connect_ex((ip, port))
        s.close()
        return result == 0
    except:
        return False

def check_firewall():
    """Check firewall status"""
    system = platform.system().lower()
    
    if system == "darwin":  # macOS
        try:
            result = subprocess.run(
                ["sudo", "/usr/libexec/ApplicationFirewall/socketfilterfw", "--getglobalstate"],
                capture_output=True, text=True, timeout=5
            )
            return "enabled" in result.stdout.lower()
        except:
            return "unknown"
    elif system == "windows":
        try:
            result = subprocess.run(
                ["netsh", "advfirewall", "show", "allprofiles", "state"],
                capture_output=True, text=True, timeout=5
            )
            return "on" in result.stdout.lower()
        except:
            return "unknown"
    else:
        return "unknown"

def main():
    print("🔍 Network Test for RTMP Setup")
    print("=" * 40)
    
    # Get local IP
    local_ip = get_local_ip()
    print(f"📍 Local IP: {local_ip}")
    
    # Test port 1935
    port_open = test_port(local_ip, 1935)
    print(f"🔌 Port 1935: {'✅ Open' if port_open else '❌ Closed'}")
    
    # Check firewall
    firewall_status = check_firewall()
    print(f"🛡️  Firewall: {firewall_status}")
    
    print("\n📋 RTMP Connection Info:")
    print(f"   Server: rtmp://{local_ip}:1935/live")
    print(f"   Stream Key: any_name_you_want")
    
    print("\n📱 For OBS Studio:")
    print(f"   Service: Custom")
    print(f"   Server: rtmp://{local_ip}:1935/live")
    print(f"   Stream Key: test_stream")
    
    print("\n📱 For Mobile Apps:")
    print(f"   RTMP URL: rtmp://{local_ip}:1935/live")
    print(f"   Stream Key: mobile_test")
    
    if not port_open:
        print("\n⚠️  Port 1935 is not accessible!")
        print("   Make sure to:")
        print("   1. Start docker-compose up")
        print("   2. Open port 1935 in firewall")
        print("   3. Check if device is on same WiFi")
    
    print("\n🧪 Test Commands:")
    print(f"   # Test streaming")
    print(f"   ffmpeg -re -i test_data/concert.flv -c copy -f flv rtmp://{local_ip}:1935/live/test")
    print(f"   ")
    print(f"   # Check library")
    print(f"   curl http://localhost:8000/library")

if __name__ == "__main__":
    main()
