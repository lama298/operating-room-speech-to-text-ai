import time
import torch
import whisper

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

start = time.time()

model = whisper.load_model("small").to(device)

end = time.time()

print("Model loaded successfully!")
print("Model device:", next(model.parameters()).device)
print(f"Loading time: {end-start:.2f} seconds")