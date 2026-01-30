#!/usr/bin/env python3
"""
Comprehensive local testing script for Voice Detection API.
Run this while the server is running: python -m app.main
"""
import base64
import requests
import json
import sys
import os

API_URL = 'http://localhost:8000'
API_KEY = 'dev-key-change-in-production'
SAMPLE_AUDIO = 'app/sample-Audio/sample voice 1.mp3'

def colored(text, color):
    colors = {'green': '\033[92m', 'red': '\033[91m', 'yellow': '\033[93m', 'reset': '\033[0m'}
    return f"{colors.get(color, '')}{text}{colors['reset']}"

def test_result(passed, message=""):
    if passed:
        return colored("✅ PASSED", "green") + (f" - {message}" if message else "")
    return colored("❌ FAILED", "red") + (f" - {message}" if message else "")

results = []

print("=" * 70)
print("🎤 Voice Detection API - Comprehensive Local Tests")
print("=" * 70)
print()

# =====================================================
# TEST 1: Health Check
# =====================================================
print("TEST 1: Health Check")
print("-" * 40)
try:
    r = requests.get(f'{API_URL}/health', timeout=5)
    passed = r.status_code == 200 and r.json().get('status') == 'healthy'
    print(f"Status: {r.status_code}")
    print(f"Response: {json.dumps(r.json(), indent=2)}")
    print(test_result(passed))
    results.append(("Health Check", passed))
except Exception as e:
    print(test_result(False, str(e)))
    results.append(("Health Check", False))
print()

# =====================================================
# TEST 2: Missing API Key (should return 401)
# =====================================================
print("TEST 2: Missing API Key")
print("-" * 40)
try:
    r = requests.post(f'{API_URL}/api/voice-detection', 
        json={'language': 'English', 'audioFormat': 'mp3', 'audioBase64': 'test'},
        headers={'Content-Type': 'application/json'}, timeout=5)
    passed = r.status_code == 401 and r.json().get('errorCode') == 'AUTH_MISSING'
    print(f"Status: {r.status_code}")
    print(f"Response: {json.dumps(r.json(), indent=2)}")
    print(test_result(passed))
    results.append(("Missing API Key", passed))
except Exception as e:
    print(test_result(False, str(e)))
    results.append(("Missing API Key", False))
print()

# =====================================================
# TEST 3: Invalid API Key (should return 401)
# =====================================================
print("TEST 3: Invalid API Key")
print("-" * 40)
try:
    r = requests.post(f'{API_URL}/api/voice-detection', 
        json={'language': 'English', 'audioFormat': 'mp3', 'audioBase64': 'test'},
        headers={'Content-Type': 'application/json', 'x-api-key': 'wrong-key'}, timeout=5)
    passed = r.status_code == 401 and r.json().get('errorCode') == 'AUTH_INVALID'
    print(f"Status: {r.status_code}")
    print(f"Response: {json.dumps(r.json(), indent=2)}")
    print(test_result(passed))
    results.append(("Invalid API Key", passed))
except Exception as e:
    print(test_result(False, str(e)))
    results.append(("Invalid API Key", False))
print()

# =====================================================
# TEST 4: Invalid Language (should return 422)
# =====================================================
print("TEST 4: Invalid Language")
print("-" * 40)
try:
    with open(SAMPLE_AUDIO, 'rb') as f:
        audio_b64 = base64.b64encode(f.read()).decode()
    r = requests.post(f'{API_URL}/api/voice-detection', 
        json={'language': 'French', 'audioFormat': 'mp3', 'audioBase64': audio_b64},
        headers={'Content-Type': 'application/json', 'x-api-key': API_KEY}, timeout=30)
    passed = r.status_code == 422  # Validation error
    print(f"Status: {r.status_code}")
    print(f"Response: {json.dumps(r.json(), indent=2)[:500]}")
    print(test_result(passed, "Correctly rejected invalid language"))
    results.append(("Invalid Language", passed))
except Exception as e:
    print(test_result(False, str(e)))
    results.append(("Invalid Language", False))
print()

# =====================================================
# TEST 5: Case-insensitive language (english -> English)
# =====================================================
print("TEST 5: Case-insensitive Language ('english' -> 'English')")
print("-" * 40)
try:
    with open(SAMPLE_AUDIO, 'rb') as f:
        audio_b64 = base64.b64encode(f.read()).decode()
    r = requests.post(f'{API_URL}/api/voice-detection', 
        json={'language': 'english', 'audioFormat': 'mp3', 'audioBase64': audio_b64},
        headers={'Content-Type': 'application/json', 'x-api-key': API_KEY}, timeout=60)
    if r.status_code == 200:
        data = r.json()
        passed = data.get('language') == 'English'
        print(f"Status: {r.status_code}")
        print(f"Language in response: {data.get('language')}")
        print(f"Classification: {data.get('classification')}")
        print(f"Confidence: {data.get('confidenceScore')}")
        print(test_result(passed, "Language correctly normalized"))
    else:
        print(f"Status: {r.status_code}")
        print(f"Response: {r.text[:500]}")
        passed = False
        print(test_result(passed))
    results.append(("Case-insensitive Language", passed))
except Exception as e:
    print(test_result(False, str(e)))
    results.append(("Case-insensitive Language", False))
print()

# =====================================================
# TEST 6: Uppercase language (TAMIL -> Tamil)
# =====================================================
print("TEST 6: Uppercase Language ('TAMIL' -> 'Tamil')")
print("-" * 40)
try:
    with open(SAMPLE_AUDIO, 'rb') as f:
        audio_b64 = base64.b64encode(f.read()).decode()
    r = requests.post(f'{API_URL}/api/voice-detection', 
        json={'language': 'TAMIL', 'audioFormat': 'mp3', 'audioBase64': audio_b64},
        headers={'Content-Type': 'application/json', 'x-api-key': API_KEY}, timeout=60)
    if r.status_code == 200:
        data = r.json()
        passed = data.get('language') == 'Tamil'
        print(f"Status: {r.status_code}")
        print(f"Language in response: {data.get('language')}")
        print(test_result(passed, "Uppercase correctly normalized"))
    else:
        passed = False
        print(f"Response: {r.text[:500]}")
        print(test_result(passed))
    results.append(("Uppercase Language", passed))
except Exception as e:
    print(test_result(False, str(e)))
    results.append(("Uppercase Language", False))
print()

# =====================================================
# TEST 7: Full detection with proper response format
# =====================================================
print("TEST 7: Full Detection - Response Format Validation")
print("-" * 40)
try:
    with open(SAMPLE_AUDIO, 'rb') as f:
        audio_b64 = base64.b64encode(f.read()).decode()
    r = requests.post(f'{API_URL}/api/voice-detection', 
        json={'language': 'English', 'audioFormat': 'mp3', 'audioBase64': audio_b64},
        headers={'Content-Type': 'application/json', 'x-api-key': API_KEY}, timeout=60)
    if r.status_code == 200:
        data = r.json()
        checks = [
            ("status == 'success'", data.get('status') == 'success'),
            ("classification in ['AI_GENERATED', 'HUMAN']", data.get('classification') in ['AI_GENERATED', 'HUMAN']),
            ("0 <= confidenceScore <= 1", 0 <= data.get('confidenceScore', -1) <= 1),
            ("explanation exists and len >= 10", len(data.get('explanation', '')) >= 10),
            ("language == 'English'", data.get('language') == 'English'),
        ]
        
        print(f"Response:")
        print(json.dumps(data, indent=2))
        print()
        print("Validation checks:")
        all_passed = True
        for check_name, passed in checks:
            status = colored("✓", "green") if passed else colored("✗", "red")
            print(f"  {status} {check_name}")
            all_passed = all_passed and passed
        
        print()
        print(test_result(all_passed))
        results.append(("Response Format", all_passed))
    else:
        print(f"Status: {r.status_code}")
        print(f"Response: {r.text[:500]}")
        print(test_result(False))
        results.append(("Response Format", False))
except Exception as e:
    print(test_result(False, str(e)))
    results.append(("Response Format", False))
print()

# =====================================================
# TEST 8: Invalid base64 data
# =====================================================
print("TEST 8: Invalid Base64 Data")
print("-" * 40)
try:
    r = requests.post(f'{API_URL}/api/voice-detection', 
        json={'language': 'English', 'audioFormat': 'mp3', 'audioBase64': 'not-valid-base64!!!'},
        headers={'Content-Type': 'application/json', 'x-api-key': API_KEY}, timeout=10)
    passed = r.status_code == 422  # Validation error
    print(f"Status: {r.status_code}")
    print(test_result(passed, "Correctly rejected invalid base64"))
    results.append(("Invalid Base64", passed))
except Exception as e:
    print(test_result(False, str(e)))
    results.append(("Invalid Base64", False))
print()

# =====================================================
# TEST 9: Valid base64 but not MP3 data
# =====================================================
print("TEST 9: Valid Base64 but Not MP3")
print("-" * 40)
try:
    fake_audio = base64.b64encode(b'This is just text, not an MP3 file at all. ' * 10).decode()
    r = requests.post(f'{API_URL}/api/voice-detection', 
        json={'language': 'English', 'audioFormat': 'mp3', 'audioBase64': fake_audio},
        headers={'Content-Type': 'application/json', 'x-api-key': API_KEY}, timeout=10)
    passed = r.status_code == 422  # Validation error
    print(f"Status: {r.status_code}")
    print(test_result(passed, "Correctly rejected non-MP3 data"))
    results.append(("Non-MP3 Data", passed))
except Exception as e:
    print(test_result(False, str(e)))
    results.append(("Non-MP3 Data", False))
print()

# =====================================================
# SUMMARY
# =====================================================
print("=" * 70)
print("📊 TEST SUMMARY")
print("=" * 70)

passed_count = sum(1 for _, p in results if p)
total_count = len(results)

for name, passed in results:
    status = colored("✅ PASS", "green") if passed else colored("❌ FAIL", "red")
    print(f"  {status}: {name}")

print()
print(f"Total: {passed_count}/{total_count} tests passed")

if passed_count == total_count:
    print(colored("\n🎉 All tests passed! API is ready for deployment.", "green"))
    sys.exit(0)
else:
    print(colored(f"\n⚠️  {total_count - passed_count} test(s) failed.", "yellow"))
    sys.exit(1)
