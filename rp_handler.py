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
    description = input_data.get("description", "A natural sounding voice")
    
    # Check if we have valid text
    if not text:
        return {"error": "No text provided in input"}
    
    try:
        # Collect all audio chunks
        all_audio = []
        sample_rates = []
        
        print(f"Generating audio for text: {text}")
        
        # Generate audio from the text
        for sample_rate, audio_chunk in generate(text, description):
            # Store audio chunks and sample rate
            all_audio.append(audio_chunk)
            sample_rates.append(sample_rate)
            print(f"Generated audio chunk of length: {len(audio_chunk)}")
        
        # Combine all audio chunks
        combined_audio = np.concatenate(all_audio) if all_audio else np.array([])
        
        # Convert audio to base64 for response
        audio_bytes = BytesIO()
        import soundfile as sf
        sf.write(audio_bytes, combined_audio, sample_rates[0] if sample_rates else 24000, format='WAV')
        audio_bytes.seek(0)
        audio_base64 = base64.b64encode(audio_bytes.read()).decode('utf-8')
        
        # Return audio data
        return {
            "text": text,  # Echo back the input text
            "audio_base64": audio_base64,
            "sample_rate": sample_rates[0] if sample_rates else 24000,
            "status": "success"
        }
        
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return {"error": str(e)}

# Start the serverless function
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})