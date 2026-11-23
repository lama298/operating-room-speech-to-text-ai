from jiwer import wer
from whisper.normalizers import EnglishTextNormalizer
import torch
import whisper
import soundfile as sf
import time

reference= "The preoperative diagnosis is acute appendicitis." #correct sentence
audio_path= "audio/pilot/processed/STD_001.wav" #input

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

#load whisper
model= whisper.load_model("turbo").to(device)
# Read the already-preprocessed WAV directly
audio, sample_rate = sf.read(audio_path)

if sample_rate != 16000:
    raise ValueError(f"Expected 16000 Hz, but received {sample_rate} Hz")

# Convert stereo to mono if necessary
if len(audio.shape) > 1:
    audio = audio.mean(axis=1)

# Whisper expects float32 audio samples
audio = audio.astype("float32")
audio_duration=len(audio) / sample_rate
# Transcribe without asking Whisper to run FFmpeg
if device =="cuda":
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

start_time= time.perf_counter()
result = model.transcribe(audio, language="en",task="transcribe")
if device =="cuda":
    torch.cuda.synchronize()

inference_time= time.perf_counter() - start_time
real_time_factor= inference_time / audio_duration
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
print(f"\nAudio duration: {audio_duration:.2f} seconds")
print(f"Inferemce time: {inference_time:.2f} seconds")
print(f"Real-time factor: {real_time_factor:.3f}")
if device =="cuda":
    peak_memory_gb=(torch.cuda.max_memory_allocated() / 1024**3)
    print(f"Peak GPU memory: {peak_memory_gb:.2f} GB")

print(f"\nWER: {wer_score:.4f}")
print(f"WER percentage: {wer_score * 100: .2f}%")
