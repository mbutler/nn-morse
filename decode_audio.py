#!/usr/bin/env python3

import argparse

import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile
import torch
from scipy import signal

from main import Net, num_tags, prediction_to_str, beam_prediction_to_str
from morse import ALPHABET, SAMPLE_FREQ, get_spectrogram

def beam_search(y_pred, beam_width=3, max_len=50):
    sequences = [([], 0)]  # each sequence is a tuple (tag sequence, score)

    for t in range(y_pred.shape[0]):
        all_candidates = []
        for seq, score in sequences:
            if len(seq) > 0 and seq[-1] == 0:  # if the last tag is <blank>, consider sequence complete
                all_candidates.append((seq, score))
                continue

            for tag in range(y_pred.shape[1]):
                candidate_seq = seq + [tag]
                candidate_score = score - np.log(y_pred[t, tag])  # using negative log likelihood
                all_candidates.append((candidate_seq, candidate_score))

        # sort candidates by score and select top beam_width
        ordered = sorted(all_candidates, key=lambda tup: tup[1])
        sequences = ordered[:beam_width]

        if all(seq[-1] == 0 for seq, _ in sequences):  # all sequences ended
            break

    return sequences[0][0]  # return the sequence with the highest score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("input")
    args = parser.parse_args()

    rate, data = scipy.io.wavfile.read(args.input)

    # Resample and rescale
    length = len(data) / rate
    new_length = int(length * SAMPLE_FREQ)

    data = signal.resample(data, new_length)
    data = data.astype(np.float32)
    data /= np.max(np.abs(data))

    # Create spectrogram
    spec = get_spectrogram(data)
    spec_orig = spec.copy()
    spectrogram_size = spec.shape[0]

    # Load model
    device = torch.device("cpu")
    model = Net(num_tags, spectrogram_size)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    # Run model on audio
    spec = torch.from_numpy(spec)
    spec = spec.permute(1, 0)
    spec = spec.unsqueeze(0)
    y_pred = model(spec)
    y_pred_l = np.exp(y_pred[0].tolist())

    
    # TODO: proper beam search
    # beam_width = 3  # set your beam width
    # max_len = 50    # set your max sequence length
    # best_sequence = beam_search(np.array(y_pred_l), beam_width, max_len)
    # print(beam_prediction_to_str(torch.tensor(best_sequence)))
    
    # Convert prediction into string
    m = torch.argmax(y_pred[0], 1)
    print(prediction_to_str(m))

    # write to file
    with open("output.txt", "w") as f:
        f.write(prediction_to_str(m))

    # Only show letters with > 5% prob somewhere in the sequence
    labels = np.asarray(["<blank>", "<space>"] + list(ALPHABET[1:]))
    sum_prob = np.max(y_pred_l, axis=0)
    show_letters = sum_prob > .05

    #plt.figure()
    #plt.subplot(2, 1, 1)
    #plt.pcolormesh(spec_orig)
    #plt.subplot(2, 1, 2)
    #plt.plot(y_pred_l[:, show_letters])
    #plt.legend(labels[show_letters])
    #plt.autoscale(enable=True, axis='x', tight=True)
    #plt.show()
