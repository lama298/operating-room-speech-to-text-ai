import time
import os
import torch
import whisper
import soundfile as sf #reads the WAV file

device = "cuda" if torch.cuda.is_available() else "cpu"
#this part make the displayment organized
print("=" * 50)
print("WHISPER TRANSCRIPTION DEMO")
print("=" * 50)
print(f"Using device: {device}")

print("\nLoading Whisper model...")
model = whisper.load_model("small").to(device)

print("Model loaded successfully!")
print(f"Model is running on: {next(model.parameters()).device}")

audio_path = "audio/processed/Recording.wav"

print(f"\nAudio file: {audio_path}")
print("File exists:", os.path.exists(audio_path))

# Load WAV without FFmpeg
audio, sample_rate = sf.read(audio_path)

print("Sample rate:", sample_rate)
print("Audio shape:", audio.shape)

# If stereo, convert to mono
if len(audio.shape) > 1:
    audio = audio.mean(axis=1)

# Whisper expects float32 audio
audio = audio.astype("float32")

start = time.time()

result = model.transcribe(audio)

end = time.time()

print("\n==============================")
print("Detected Language")
print("==============================")
print(result["language"])

print("\n==============================")
print("Full Transcript")
print("==============================")
print(result["text"])

print("\n==============================")
print("Segments")
print("==============================")

for segment in result["segments"]:
    print(
        f"[{segment['start']:.2f}s --> {segment['end']:.2f}s] "
        f"{segment['text']}"
    )

print("\n==============================")
print(f"Inference Time: {end-start:.2f} seconds")
print("==============================")