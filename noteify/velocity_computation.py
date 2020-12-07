# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 17:11:01 2020

@author: skarnik
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa

MAX_VELOCITY = 127
EPSILON = 1e-5

def get_velocities_from_spectrogram(cqt, frequency_bins, start_and_end_buckets, num_time_steps = 10):
    """
    cqt is (time buckets) x (frequency bins)
    
    start_and_end_buckets is list of tuples of 
    (start time bucket, end time bucket)

    """
    start_and_end_buckets = np.array(start_and_end_buckets)
    ranges = np.concatenate([[np.arange(num_time_steps)] for i in range(len(start_and_end_buckets))], 0)
    start_buckets = np.transpose([start_and_end_buckets[:, 0]])
    frequency_bins_transposed = np.transpose([frequency_bins])
    
    cqt_mean = np.mean(cqt[np.where(cqt > EPSILON)])
    cqt_std = np.std(cqt[np.where(cqt > EPSILON)])
    modified_cqt = np.zeros(np.shape(cqt))
    indices = np.where(abs(cqt - cqt_mean) < 3 * cqt_std)
    modified_cqt[indices] = cqt[indices] / np.max(cqt)
    
    velocities = np.max(modified_cqt[frequency_bins_transposed, start_buckets + ranges], axis=(1,)) * MAX_VELOCITY
    velocities = np.array(velocities, dtype=int)
    
    return velocities

y, sr = librosa.load(librosa.ex('trumpet'))
C = np.abs(librosa.cqt(y, sr=sr))
fig, ax = plt.subplots()
frequency_bins = [60, 70, 80]
start_and_end_buckets = [[1, 4], [2, 5], [3, 7]]
print(get_velocities_from_spectrogram(C, frequency_bins, start_and_end_buckets))

img = librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max),
                               sr=sr, x_axis='time', y_axis='cqt_note', ax=ax)
ax.set_title('Constant-Q power spectrum')
fig.colorbar(img, ax=ax, format="%+2.0f dB")

