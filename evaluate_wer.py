from jiwer import wer
from whisper.normalizers import EnglishTextNormalizer
import torch
import whisper
import soundfile as sf

reference= "Patient is stable, blood pressure is  120 over 80" #correct sentence
audio_path= "audio/processed/Recording.wav" #input

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

#load whisper
model= whisper.load_model("small").to(device)
# Read the already-preprocessed WAV directly
audio, sample_rate = sf.read(audio_path)

if sample_rate != 16000:
    raise ValueError(f"Expected 16000 Hz, but received {sample_rate} Hz")

# Convert stereo to mono if necessary
if len(audio.shape) > 1:
    audio = audio.mean(axis=1)

# Whisper expects float32 audio samples
audio = audio.astype("float32")

# Transcribe without asking Whisper to run FFmpeg
result = model.transcribe(audio, language="en")
prediction = result["text"]

#normalization prevemts unimportant formatting differences, such as capitalization and puncuation from being
#counted as speeech recognition errors


normalizer= EnglishTextNormalizer()

normalized_reference= normalizer(reference)
normalized_prediction=normalizer(prediction)

wer_score=wer(normalized_reference,normalized_prediction)

print("Original reference: ", reference)
print("Original prediction: ",prediction)

print("\nNormalized reference: ", normalized_reference)
print("\nNormalized prediction: ", normalized_prediction)

print(f"\nWER: {wer_score:.4f}")
print(f"WER percentage: {wer_score * 100: .2f}%")
