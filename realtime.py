#!/usr/bin/env python3

import argparse
import numpy as np
import torch
import pyaudio
from scipy import signal
import queue
import threading
from main import Net, num_tags, prediction_to_str
from morse import ALPHABET, SAMPLE_FREQ, get_spectrogram

# PyAudio Configuration
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = SAMPLE_FREQ
CHUNK = 1024  # Number of audio samples per frame

# Initialize PyAudio and Stream
audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

# Buffer to store audio
audio_buffer = queue.Queue()

def process_stream():
    """ Continuously captures audio data and processes it. """
    while True:
        data = np.frombuffer(stream.read(CHUNK), dtype=np.float32)
        if not data.size: continue
        audio_buffer.put(data)

def preprocess_data(data):
    """ Preprocess the audio data for the model. """
    # Resample and rescale
    length = len(data) / RATE
    new_length = int(length * SAMPLE_FREQ)

    data = signal.resample(data, new_length)
    data = data.astype(np.float32)
    data /= np.max(np.abs(data))

    # Create spectrogram
    spec = get_spectrogram(data)
    spectrogram_size = spec.shape[0]
    return spec, spectrogram_size

def predict_and_decode(model):
    """ Predicts and decodes the Morse code from audio data. """
    while True:
        if audio_buffer.empty(): continue

        data = audio_buffer.get()
        spec, _ = preprocess_data(data)

        # Convert to PyTorch tensor and predict
        spec = torch.from_numpy(spec).permute(1, 0).unsqueeze(0)
        with torch.no_grad():
            y_pred = model(spec)
            predicted_indices = torch.argmax(y_pred[0], 1)
            decoded_prediction = prediction_to_str(predicted_indices)
            print(decoded_prediction)

def main(model_path):
    # Load model
    device = torch.device("cpu")
    model = Net(num_tags, SAMPLE_FREQ)  # Adjust the second parameter if needed
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Start audio processing thread
    audio_thread = threading.Thread(target=process_stream)
    audio_thread.start()

    # Start prediction thread
    predict_thread = threading.Thread(target=predict_and_decode, args=(model,))
    predict_thread.start()

    audio_thread.join()
    predict_thread.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    main(args.model)