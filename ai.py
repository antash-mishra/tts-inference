import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
from threading import Thread
import numpy as np

# Using torch.no_grad() for inference to save memory
@torch.no_grad()
def generate(text, description="A natural sounding voice"):
    """
    Generate audio from text in a single pass (non-streaming)
    
    Args:
        text (str): The text to convert to speech
        description (str): Description of the voice style
        
    Returns:
        tuple: (sample_rate, audio_data)
    """
    # Set device - prefer CUDA if available
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model and tokenizer if not already loaded
    # Use singleton pattern to avoid reloading
    if not hasattr(generate, "model") or generate.model is None:
        print("Loading model and tokenizer...")
        generate.tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
        
        # Load model with optimization flags
        generate.model = ParlerTTSForConditionalGeneration.from_pretrained(
            "ai4bharat/indic-parler-tts"
        ).to(device)
        
        # Get sampling rate from model config
        generate.sampling_rate = generate.model.audio_encoder.config.sampling_rate
        print(f"Model loaded. Sampling rate: {generate.sampling_rate}")
    
    # Tokenize input text and description
    print(f"Tokenizing input text: \"{text}\"")
    inputs = generate.tokenizer(description, return_tensors="pt").to(device)
    prompt = generate.tokenizer(text, return_tensors="pt").to(device)
    
    # Setup generation parameters
    # Batch everything in a single run for better efficiency
    generation_kwargs = dict(
        input_ids=inputs.input_ids,
        prompt_input_ids=prompt.input_ids,
        attention_mask=inputs.attention_mask,
        prompt_attention_mask=prompt.attention_mask,
        do_sample=True,
        temperature=1.0,
        max_new_tokens=500,  # Limit generation length for safety
    )
    
    # Generate audio directly
    print("Generating audio...")
    with torch.inference_mode():  # Further memory optimization
        # The generate method directly returns the audio values
        audio_values = generate.model.generate(**generation_kwargs)
    
    # Convert tensor to numpy array
    audio_np = audio_values.cpu().numpy().squeeze()
    
    print(f"Generated audio with length: {len(audio_np)} samples")
    return generate.sampling_rate, audio_np

