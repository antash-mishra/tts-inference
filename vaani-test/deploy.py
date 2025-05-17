import runpod
import torch
import base64
import os
import numpy as np
import openai
from io import BytesIO

# Import your existing generation function
from ai import generate

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def handler(event):
    """
    Process incoming requests to generate audio from text using OpenAI and the Parler TTS model.
    
    Args:
        event (dict): Contains the input data with text to be processed
        
    Returns:
        dict: Contains the generated audio data as base64 and the text response
    """
    print("Worker starting...")
    
    # Extract input data
    input_data = event["input"]
    user_message = input_data.get("message", "")
    
    # Check if we have a valid message
    if not user_message:
        return {"error": "No message provided in input"}
    
    try:
        # Process with OpenAI to get response text
        print(f"Processing message with OpenAI: {user_message}")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful voice assistant who talks in santali language only and write santali script only."},
                {"role": "user", "content": user_message}
            ]
        )
        
        # Get response text from OpenAI
        ai_response = response.choices[0].message.content
        print(f"Generated response: {ai_response}")
        
        # Generate audio from the AI response
        description = "A natural sounding voice"
        
        # Collect all audio chunks
        all_audio = []
        sample_rates = []
        
        print("Generating audio with Parler TTS...")
        for sample_rate, audio_chunk in generate(ai_response, description):
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
        
        # Return both text and audio
        return {
            "text_response": ai_response,
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
