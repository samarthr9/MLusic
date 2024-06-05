import torch
import numpy as np
import soundfile as sf
from pydub import AudioSegment

# Load the saved tensor
predicted_chunk = torch.load('/Users/samarthrao/Downloads/Todi_Audio_Files/KVN_todi_1/predicted_chunk.pt')

# Assuming the tensor needs to be converted to a 1D numpy array for audio conversion
predicted_chunk_numpy = predicted_chunk.numpy().flatten()

# Normalize the predicted_chunk_numpy to the expected range for audio data
predicted_chunk_numpy = (predicted_chunk_numpy - predicted_chunk_numpy.min()) / (predicted_chunk_numpy.max() - predicted_chunk_numpy.min()) * 2 - 1

# Define the sampling rate (use the same sampling rate as your original audio files)
sr = 22050  # Replace this with your actual sampling rate if different

# Save the numpy array as a wav file
output_wav_file = '/Users/samarthrao/Downloads/Todi_Audio_Files/KVN_todi_1/predicted_chunk.wav'
sf.write(output_wav_file, predicted_chunk_numpy, sr)

# Convert the wav file to mp3
output_mp3_file = '/Users/samarthrao/Downloads/Todi_Audio_Files/KVN_todi_1/predicted_chunk.mp3'
audio = AudioSegment.from_wav(output_wav_file)
audio.export(output_mp3_file, format='mp3')

print(f"Predicted chunk saved as MP3: {output_mp3_file}")
