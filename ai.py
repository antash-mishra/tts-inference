import torch
from parler_tts import ParlerTTSForConditionalGeneration, ParlerTTSStreamer
from transformers import AutoTokenizer, AutoFeatureExtractor, AutoModelForSeq2SeqLM
from IndicTransToolkit.processor import IndicProcessor
import numpy as np
import nltk

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
torch_dtype = torch.bfloat16 if device != "cpu" else torch.float32
repo_id = "ai4bharat/indic-parler-tts"

model = ParlerTTSForConditionalGeneration.from_pretrained(
    repo_id, attn_implementation="eager", torch_dtype=torch_dtype,
).to(device)
tokenizer = AutoTokenizer.from_pretrained(repo_id)
description_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
feature_extractor = AutoFeatureExtractor.from_pretrained(repo_id)

SAMPLE_RATE = model.audio_encoder.config.sampling_rate
SEED = 42

translation_repo_id = "ai4bharat/indictrans2-en-indic-1B"
translation_tokenizer = AutoTokenizer.from_pretrained(translation_repo_id, trust_remote_code=True)

translation_model = AutoModelForSeq2SeqLM.from_pretrained(
    translation_repo_id,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    attn_implementation="flash_attention_2"
).to(device)

ip = IndicProcessor(inference=True)


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

    chunk_size = 25
    # Tokenize input text and description
    print(f"Tokenizing input text: \"{text}\"")
    inputs = description_tokenizer(description, return_tensors="pt").to(device)
    # prompt = generate.tokenizer(text, return_tensors="pt").to(device)

    sentences_text = nltk.sent_tokenize(text)
    curr_sentence = ""
    chunks = []

    for sentence in sentences_text:
        candidate = " ".join([curr_sentence, sentence])
        if len(candidate.split()) > chunk_size:
            chunks.append(curr_sentence)
            curr_sentence = sentence
        else:
            curr_sentence = candidate       

    if curr_sentence != "":
        chunks.append(curr_sentence)
  
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

@torch.no_grad()
def translate(text, src_lang, tgt_lang):
    """
    Translate text from source language to target language using IndicTrans.
    
    Args:
        text (str): The text to translate
        src_lang (str): Source language code
        tgt_lang (str): Target language code
        
    Returns:
        str: Translated text
    """
    input_sentences = [text]
    print(f"Translating text: \"{input_sentences}\"")

    # Preprocess the text for translation
    batch = ip.preprocess_batch(
        input_sentences,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
    )

    print(f"Translating text: \"{batch}\"")
    
    # Tokenize input text
    inputs = translation_tokenizer(
        batch,
        truncation=True,
        padding="longest",
        return_tensors="pt",
        return_attention_mask=True,
    ).to(device)
    
    # Generate translation
    generated_tokens = translation_model.generate(
        **inputs,
        use_cache=True,
        min_length=0,
        max_length=256,
        num_beams=5,
        num_return_sequences=1,
        early_stopping=True
    )

    # Decode the generated translation
    generated_tokens = translation_tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    translations = ip.postprocess_batch(generated_tokens, lang=tgt_lang)

    return translations[0]
