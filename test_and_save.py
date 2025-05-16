import json
import base64
import os
import time
from rp_handler import handler

def run_test_and_save_audio():
    """
    Runs the handler with test_input.json and saves the resulting audio to a file
    """
    print("Loading test input...")
    
    # Load test input
    with open('test_input.json', 'r') as f:
        test_data = json.load(f)
    
    print(f"Testing with text: {test_data['input']['text']}")
    
    # Measure execution time
    start_time = time.time()
    
    # Run the handler with the test input
    result = handler(test_data)
    
    # Calculate execution time
    execution_time = time.time() - start_time
    
    if 'error' in result:
        print(f"Error: {result['error']}")
        return
    
    # Print the response info
    print(f"Response status: {result['status']}")
    print(f"Sample rate: {result['sample_rate']}")
    print(f"Execution time: {execution_time:.2f} seconds")
    
    # Save the audio to a file
    if 'audio_base64' in result:
        audio_data = base64.b64decode(result['audio_base64'])
        output_file = 'test_output.wav'
        
        with open(output_file, 'wb') as f:
            f.write(audio_data)
        
        file_size = os.path.getsize(output_file) / 1024  # Size in KB
        print(f"Audio saved to {output_file} ({file_size:.2f} KB)")
        
        # Print metrics
        audio_duration = len(audio_data) / (result['sample_rate'] * 4)  # Approximate duration in seconds (assuming 16-bit stereo)
        print(f"Approximate audio duration: {audio_duration:.2f} seconds")
        print(f"Processing ratio: {execution_time/audio_duration:.2f}x realtime")
    else:
        print("No audio data in response")

if __name__ == "__main__":
    run_test_and_save_audio() 