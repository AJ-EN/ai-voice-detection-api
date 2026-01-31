# Voice Detection API

AI-Generated Voice Detection API for detecting synthetic vs human voices across 5 Indian languages.

## Supported Languages
- Tamil
- English  
- Hindi
- Malayalam
- Telugu

## API Endpoint

```
POST /api/voice-detection
```

### Headers
```
Content-Type: application/json
x-api-key: YOUR_API_KEY
```

### Request Body
```json
{
  "language": "English",
  "audioFormat": "mp3",
  "audioBase64": "base64_encoded_audio_data"
}
```

### Response
```json
{
  "status": "success",
  "language": "English",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.92,
  "explanation": "Detected: unnatural pitch consistency, robotic amplitude stability"
}
```

## Local Development

```bash
# Install dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run server
python -m app.main
```

## Health Check
```
GET /health
```

## License
MIT
