# rp_handler.py
import runpod
import torch
import base64
import os
import numpy as np
from io import BytesIO

# Import your existing generation function
from ai import generate

def handler(event):
    """
    Process incoming requests to generate audio from text using Parler TTS model.
    
    Args:
        event (dict): Contains the input data with text to be processed
        
    Returns:
        dict: Contains the generated audio data as base64
    """
    print("Worker starting...")
    
    # Extract input data
    input_data = event["input"]
    text = input_data.get("text", "")
    description = input_data.get("description", "A female speaker delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up.")
    
    # Check if we have valid text
    if not text:
        return {"error": "No text provided in input"}
    
    try:
        print(f"Generating audio for text: {text}")
        
        # Generate audio from the text
        # Non-streaming version now returns (sample_rate, audio) directly
        sample_rate, audio = generate(text, description)
        
        print(f"Generated audio with length: {len(audio)}")
        
        # Convert audio to base64 for response
        audio_bytes = BytesIO()
        import soundfile as sf
        sf.write(audio_bytes, audio, sample_rate, format='WAV')
        audio_bytes.seek(0)
        audio_base64 = base64.b64encode(audio_bytes.read()).decode('utf-8')
        
        # Return audio data
        return {
            "text": text,  # Echo back the input text
            "audio_base64": audio_base64,
            "sample_rate": sample_rate,
            "status": "success"
        }
        
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return {"error": str(e)}

# Start the serverless function
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})