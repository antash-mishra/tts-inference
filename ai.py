import torch
from parler_tts import ParlerTTSForConditionalGeneration, ParlerTTSStreamer
from transformers import AutoTokenizer, AutoFeatureExtractor
import numpy as np
import nltk

nltk.download('punkt_tab')

device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
torch_dtype = torch.bfloat16 if device != "cpu" else torch.float32
repo_id = "ai4bharat/indic-parler-tts"

model = ParlerTTSForConditionalGeneration.from_pretrained(
    repo_id, attn_implementation="eager", torch_dtype=torch_dtype,
).to(device)
tokenizer = AutoTokenizer.from_pretrained(repo_id)
description_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
feature_extractor = AutoFeatureExtractor.from_pretrained(repo_id)

SAMPLE_RATE = feature_extractor.sampling_rate
SEED = 42

# Using torch.no_grad() for inference to save memory
@torch.no_grad()
def generate(text, description):
    """
    Generate audio from text in a single pass (non-streaming)
    
    Args:
        text (str): The text to convert to speech
        description (str): Description of the voice style
        
    Returns:
        tuple: (sample_rate, audio_data)
    """
    model.generation_config.cache_implementation = "static"

    print(f"Using device: {device}")
    play_steps = int(SAMPLE_RATE * 0.5)
    streamer = ParlerTTSStreamer(model, device=device, play_steps=play_steps)

            
    # Tokenize input text and description
    print(f"Tokenizing input text: \"{text}\"")
    inputs = description_tokenizer(description, return_tensors="pt").to(device)
    # prompt = generate.tokenizer(text, return_tensors="pt").to(device)

    sentences_text = nltk.sent_tokenize(text)
    curr_sentence = ""
    chunks = []

    for sentence in sentences_text:
        candidate = " ".join([curr_sentence, sentence])
        if len(tokenizer.encode(candidate)) > 500:
            chunks.append(curr_sentence)
            curr_sentence = sentence
        else:
            curr_sentence = candidate       

    if curr_sentence != "":
        chunks.append(curr_sentence)
  
    print(chunks)

    all_audio = []
    

    for chunk in chunks:
        # Setup generation parameters
        # Batch everything in a single run for better efficiency
        prompt = tokenizer(chunk, return_tensors="pt").to(device)
        generation_kwargs = dict(
            input_ids=inputs.input_ids,
            prompt_input_ids=prompt.input_ids,
            attention_mask=inputs.attention_mask,
            prompt_attention_mask=prompt.attention_mask,
            streamer=streamer,
            do_sample=True,
            return_dict_in_generate=True
        )

        print("Generating audio...")
        with torch.inference_mode():  # Further memory optimization
            # The generate method directly returns the audio values
            generation = model.generate(**generation_kwargs)

        # Extract audio from generation
        if hasattr(generation, 'sequences') and hasattr(generation, 'audios_length'):
            audio = generation.sequences[0, :generation.audios_length[0]]
            audio_np = audio.to(torch.float32).cpu().numpy().squeeze()
            if len(audio_np.shape) > 1:
                audio_np = audio_np.flatten()
            all_audio.append(audio_np)
    
    # Combine all audio chunks
    combined_audio = np.concatenate(all_audio)
    
    
    print(f"Generated audio with length: {len(combined_audio)} samples")
    return SAMPLE_RATE, combined_audio