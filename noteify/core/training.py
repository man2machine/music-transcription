# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 01:21:39 2020

@author: Shahir
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.io import wavfile

import librosa

import torch
from nnAudio import Spectrogram as nn_spectrogram

DEFAULT_SAMPLE_RATE = 22050
DEFAULT_FMIN = 27.500 # A0
DEFAULT_BINS_PER_NOTE = 8
DEFAULT_BINS_PER_OCTAVE = 12 * DEFAULT_BINS_PER_NOTE
DEFAULT_N_BINS = 88 * DEFAULT_BINS_PER_NOTE

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

sr, x = wavfile.read("fantasia.wav")
x = x / (1 << 15)

# x, sr = librosa.load("fantasia.wav")

x = librosa.resample(x.astype(np.float), sr, DEFAULT_SAMPLE_RATE)
sr = DEFAULT_SAMPLE_RATE

x = x[:sr*5]

x = x.copy()
if len(x.shape) > 1:
    x = x.mean(axis=1)
x = torch.tensor(x).float().view(1, -1)

spec_layer = nn_spectrogram.CQT1992v2(sr=sr,
                                      fmin=DEFAULT_FMIN,
                                      n_bins=DEFAULT_N_BINS,
                                      bins_per_octave=DEFAULT_BINS_PER_OCTAVE,
                                      hop_length=512,
                                      window='hann',
                                      output_format='Magnitude')

freqs = librosa.cqt_frequencies(n_bins=DEFAULT_N_BINS,
                                fmin=DEFAULT_FMIN,
                                bins_per_octave=DEFAULT_BINS_PER_OCTAVE)

# shape (batch_size, freq_bins, time_steps)
# time_steps = sr / hop_length
# frequency for bin n = freqs[n]

z = spec_layer(x)
z = torch.log(z)

fig = plt.figure(figsize=(5, 10))
ax = plt.subplot()
ax.imshow(z.numpy()[0], cmap='plasma')
ax.invert_yaxis()
fig.show()
