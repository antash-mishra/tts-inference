# Parler TTS RunPod Worker

This repository contains a RunPod worker for running Parler TTS inference in a serverless environment.

## Overview

This worker accepts text as input and returns the generated speech audio as base64-encoded WAV data. The TTS implementation is optimized for single-pass generation rather than streaming.

## Input Format

```json
{
  "text": "Text to be converted to speech",
  "description": "A description of how the voice should sound (optional)"
}
```

If no description is provided, a default voice description will be used.

## Output Format

```json
{
  "text": "The input text (echoed back)",
  "audio_base64": "base64-encoded WAV audio data",
  "sample_rate": 24000,
  "status": "success"
}
```

## Local Testing

To test the worker locally:

1. Create a Python virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the test script:
   ```
   python test_and_save.py
   ```

This will use the `test_input.json` file as input and save the generated audio to `test_output.wav`.

## Performance Benchmarks

When running locally on a machine with decent GPU support, you can expect:
- Processing ratio: Typically 10-20x faster than real-time
- Latency: Usually under 1-2 seconds for short texts
- Audio quality: High-quality natural-sounding voices

## Deployment to RunPod

1. Build and tag the Docker image:
   ```
   docker build -t your-dockerhub-username/parler-tts:latest .
   ```

2. Push the image to Docker Hub:
   ```
   docker push your-dockerhub-username/parler-tts:latest
   ```

3. Create a new endpoint on RunPod:
   - Go to https://www.runpod.io/console/serverless
   - Click "New Endpoint"
   - Select "Docker Image" and click "Next"
   - Enter your Docker image URL: `docker.io/your-dockerhub-username/parler-tts:latest`
   - Configure GPU requirements (recommend at least 16GB GPU for TTS models)
   - Create the endpoint

4. Test your endpoint using the RunPod console or API:
   ```
   curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer YOUR_API_KEY" \
        -d '{"input": {"text": "Hello world"}}'
   ```

## API Integration

To integrate with your application, make POST requests to the RunPod endpoint URL:

```javascript
async function generateSpeech(text, description = "") {
  const response = await fetch('https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': 'Bearer YOUR_API_KEY'
    },
    body: JSON.stringify({
      input: {
        text: text,
        description: description
      }
    })
  });
  
  return await response.json();
}
```

To extract and play the audio in a web browser:

```javascript
function playAudio(responseData) {
  const audioData = responseData.output.audio_base64;
  const audioSrc = `data:audio/wav;base64,${audioData}`;
  
  const audio = new Audio(audioSrc);
  audio.play();
}
```

For synchronous endpoints, you'll get the result directly. For asynchronous endpoints, you'll need to poll for results using the job ID. 