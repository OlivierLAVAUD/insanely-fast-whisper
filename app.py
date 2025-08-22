import torch
from transformers import pipeline
from pyannote.audio import Pipeline
import torchaudio
import numpy as np
from pyannote.core import Segment

# Set your Hugging Face token
HF_TOKEN = "hf_your_token_here"  # Replace with your classic token


def resample_audio(audio_file_path, target_sr=16000):
    """Resample audio to target sample rate for pyannote compatibility"""
    waveform, original_sr = torchaudio.load(audio_file_path)
    if original_sr != target_sr:
        resampler = torchaudio.transforms.Resample(original_sr, target_sr)
        waveform = resampler(waveform)
    return waveform, target_sr

def transcribe_with_diarization(audio_file_path):
    # Check if CUDA is available
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load Whisper model
    print("Loading Whisper model...")
    whisper_pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3",
        torch_dtype=torch.float16,
        device=device,
        token=HF_TOKEN
    )
    
    # Transcribe audio
    print("Transcribing audio...")
    outputs = whisper_pipe(
        audio_file_path,
        chunk_length_s=30,
        batch_size=8,
        return_timestamps=True
    )
    
    # Load speaker diarization model with error handling
    print("Loading speaker diarization model...")
    try:
        diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=HF_TOKEN
        )
        diarization_pipeline.to(torch.device(device))
    except Exception as e:
        print(f"Diarization model loading failed: {e}")
        print("Returning transcription without speaker labels")
        return outputs["chunks"]
    
    # Perform speaker diarization with resampled audio
    print("Performing speaker diarization...")
    try:
        # Create a temporary resampled audio file
        import tempfile
        import os
        
        waveform, sr = resample_audio(audio_file_path)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            torchaudio.save(tmp_file.name, waveform, sr)
            diarization = diarization_pipeline(tmp_file.name)
        os.unlink(tmp_file.name)
        
    except Exception as e:
        print(f"Diarization failed: {e}")
        print("Returning transcription without speaker labels")
        return outputs["chunks"]
    
    # Combine transcription with speaker labels
    print("Combining results...")
    segments = outputs["chunks"]
    
    final_output = []
    for segment in segments:
        start_time = segment["timestamp"][0]
        end_time = segment["timestamp"][1]
        text = segment["text"]
        
        # Find speaker for this time segment
        speaker = "SPEAKER_00"
        try:
            for turn, _, speaker_label in diarization.itertracks(yield_label=True):
                if turn.start <= start_time <= turn.end:
                    speaker = speaker_label
                    break
        except:
            pass  # Fallback to default speaker if diarization fails
        
        final_output.append({
            "start": start_time,
            "end": end_time,
            "speaker": speaker,
            "text": text.strip()
        })
    
    return final_output

def save_transcription(results, output_file="transcription.txt"):
    """Save transcription to file"""
    with open(output_file, "w", encoding="utf-8") as f:
        for segment in results:
            f.write(f"[{segment['speaker']}] {segment['start']:.1f}-{segment['end']:.1f}: {segment['text']}\n")
    
    print(f"Transcription saved to {output_file}")

# Alternative: Simple transcription without diarization
def simple_transcribe(audio_file_path):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3",
        device=device,
        torch_dtype=torch.float16,
        token=HF_TOKEN
    )
    
    result = pipe(audio_file_path, return_timestamps=True)
    return result["chunks"]

# Main execution
if __name__ == "__main__":
    audio_file = "audio.mp3"  # Replace with your audio file path
    
    try:
        # Try with diarization first
        results = transcribe_with_diarization(audio_file)
        
        # Print results
        print("\n=== TRANSCRIPTION RESULTS ===")
        for segment in results:
            print(f"[{segment['speaker']}] {segment['start']:.1f}-{segment['end']:.1f}: {segment['text']}")
        
        # Save to file
        save_transcription(results)
        
    except Exception as e:
        print(f"Main error: {e}")
        print("Falling back to simple transcription...")
        
        # Fallback to simple transcription
        try:
            results = simple_transcribe(audio_file)
            print("\n=== SIMPLE TRANSCRIPTION (No speakers) ===")
            for segment in results:
                print(f"{segment['timestamp'][0]:.1f}-{segment['timestamp'][1]:.1f}: {segment['text']}")
            
            # Save simple version
            with open("transcription_simple.txt", "w", encoding="utf-8") as f:
                for segment in results:
                    f.write(f"{segment['timestamp'][0]:.1f}-{segment['timestamp'][1]:.1f}: {segment['text']}\n")
            
        except Exception as e2:
            print(f"Simple transcription also failed: {e2}")