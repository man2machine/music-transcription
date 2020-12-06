# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 14:01:42 2020

@author: Shahir
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from nnAudio import Spectrogram as nn_spectrogram

from noteify.core.config import (SAMPLE_RATE, FMIN_FREQ, BINS_PER_NOTE, BINS_PER_OCTAVE, NUM_BINS,
                                 HOP_LENGTH, SEGMENT_LENGTH, SEGMENT_SAMPLES, SEGMENT_FRAMES, EPSILON)

class SpectrogramLayer(nn.Module):
    """
    Input: (batch_size, num_samples)
    Output: (batch_size, num_frames, freq_bins)
    """

    def __init__(self, trainable=False):
        super().__init__()

        self.spec_layer = nn_spectrogram.CQT1992v2(sr=SAMPLE_RATE,
                                                   fmin=FMIN_FREQ,
                                                   n_bins=NUM_BINS,
                                                   bins_per_octave=BINS_PER_OCTAVE,
                                                   hop_length=HOP_LENGTH,
                                                   window='hann',
                                                   output_format='Magnitude',
                                                   trainable=trainable)
    
    def forward(self, x):
        z = self.spec_layer(x)
        z = torch.log(z + EPSILON)
        return z

class AcousticCRNN(nn.Module):
    """
    Input: (batch_size, num_frames, freq_bins)
    Output: (batch_size, num_frames, num_classes)
    """
    
    def __init__(self):
        super().__init__()

        

class TranscriptionNN(nn.Module):
    """
    Input: (batch_size, num_samples)
    Output: (batch_size, num_frames, num_notes) x 3
    """
    
    def __init__(self):
        super().__init__()


