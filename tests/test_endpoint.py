#!/usr/bin/env python3
"""
Endpoint validation script for Voice Detection API.
Tests all endpoints and validates response formats.
"""
import requests
import base64
import sys
import os
import time


# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "dev-key-change-in-production")


def test_health_endpoint():
    """Test the health check endpoint."""
    print("\n🔍 Testing /health endpoint...")
    
    try:
        response = requests.get(f"{API_URL}/health", timeout=10)
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert data.get("status") == "healthy", "Status should be 'healthy'"
        
        print(f"   ✅ Health check passed: {data}")
        return True
    except Exception as e:
        print(f"   ❌ Health check failed: {e}")
        return False


def test_auth_rejection():
    """Test that requests without API key are rejected."""
    print("\n🔍 Testing authentication rejection...")
    
    try:
        response = requests.post(
            f"{API_URL}/api/voice-detection",
            json={"language": "Tamil", "audioFormat": "mp3", "audioBase64": "test"},
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        assert response.status_code == 401, f"Expected 401, got {response.status_code}"
        data = response.json()
        assert data.get("status") == "error", "Status should be 'error'"
        assert "errorCode" in data, "Should have errorCode"
        
        print(f"   ✅ Auth rejection works: {data}")
        return True
    except Exception as e:
        print(f"   ❌ Auth rejection test failed: {e}")
        return False


def test_invalid_api_key():
    """Test that invalid API key is rejected."""
    print("\n🔍 Testing invalid API key rejection...")
    
    try:
        response = requests.post(
            f"{API_URL}/api/voice-detection",
            json={"language": "Tamil", "audioFormat": "mp3", "audioBase64": "test"},
            headers={
                "Content-Type": "application/json",
                "x-api-key": "invalid-key"
            },
            timeout=10
        )
        
        assert response.status_code == 401, f"Expected 401, got {response.status_code}"
        data = response.json()
        assert data.get("errorCode") == "AUTH_INVALID", "Should have AUTH_INVALID error code"
        
        print(f"   ✅ Invalid API key rejection works: {data}")
        return True
    except Exception as e:
        print(f"   ❌ Invalid API key test failed: {e}")
        return False


def test_voice_detection(audio_path: str = None):
    """Test voice detection with actual audio file."""
    print("\n🔍 Testing /api/voice-detection endpoint...")
    
    # If no audio file provided, create a minimal test
    if audio_path and os.path.exists(audio_path):
        print(f"   Using audio file: {audio_path}")
        with open(audio_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode()
    else:
        print("   ⚠️  No audio file provided, using validation test only")
        # Test with invalid base64 that should fail validation
        response = requests.post(
            f"{API_URL}/api/voice-detection",
            json={
                "language": "Tamil",
                "audioFormat": "mp3",
                "audioBase64": "a" * 200  # Invalid MP3 data
            },
            headers={
                "Content-Type": "application/json",
                "x-api-key": API_KEY
            },
            timeout=30
        )
        
        # Should return 422 (validation error) or 400
        if response.status_code in [422, 400]:
            print(f"   ✅ Validation correctly rejected invalid audio: {response.status_code}")
            return True
        else:
            print(f"   ❌ Unexpected status code: {response.status_code}")
            print(f"      Response: {response.text}")
            return False
    
    try:
        start_time = time.time()
        
        response = requests.post(
            f"{API_URL}/api/voice-detection",
            json={
                "language": "English",
                "audioFormat": "mp3",
                "audioBase64": audio_b64
            },
            headers={
                "Content-Type": "application/json",
                "x-api-key": API_KEY
            },
            timeout=60
        )
        
        elapsed = time.time() - start_time
        
        print(f"   Response time: {elapsed:.2f}s")
        print(f"   Status code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            # Validate response schema
            assert data.get("status") == "success", "Status should be 'success'"
            assert data.get("classification") in ["AI_GENERATED", "HUMAN"], "Invalid classification"
            assert 0 <= data.get("confidenceScore", -1) <= 1, "Confidence must be 0-1"
            assert len(data.get("explanation", "")) >= 10, "Explanation too short"
            
            print(f"   ✅ Detection successful!")
            print(f"      Classification: {data['classification']}")
            print(f"      Confidence: {data['confidenceScore']}")
            print(f"      Explanation: {data['explanation']}")
            return True
        else:
            print(f"   ❌ Detection failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ Detection test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("🎤 Voice Detection API - Endpoint Validation")
    print(f"   API URL: {API_URL}")
    print("=" * 60)
    
    results = []
    
    # Basic tests
    results.append(("Health Check", test_health_endpoint()))
    results.append(("Auth Rejection", test_auth_rejection()))
    results.append(("Invalid API Key", test_invalid_api_key()))
    
    # Voice detection test (with optional audio file)
    audio_file = sys.argv[1] if len(sys.argv) > 1 else None
    results.append(("Voice Detection", test_voice_detection(audio_file)))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status}: {name}")
    
    print(f"\n   Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! API is ready for deployment.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
