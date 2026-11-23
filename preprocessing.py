import os #handles path and files
import subprocess #allows python to execute an external program
import imageio_ffmpeg #gives us path to working ffmpeg program


def preprocess_audio(input_path, output_path):
    #input: location of original recording, output: location whre thee converted recording should be saved
    """
    Convert input audio to:
    - WAV
    - mono
    - 16 kHz
    """

    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe() #locates the working standalone FFmpeg and saves it

    os.makedirs(os.path.dirname(output_path), exist_ok=True) #makes sure the destination folder exists

    command = [
        ffmpeg_path, #run working program
        "-y", #automatically overwrite existing output
        "-i", input_path, #next value is the input file
        "-ac", "1", # produce one audio channel:mono
        "-ar", "16000", #prroduce sampling rate of 16,000 hz
        "-acodec", "pcm_s16le", #store the sound as uncompressed 16 bit PCM
        output_path #destination of the procesed WAV
    ]

    subprocess.run(command, check=True) #runs the command

    print("Audio preprocessing completed.")
    print("Saved to:", output_path)