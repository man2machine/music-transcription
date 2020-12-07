# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 14:01:42 2020

@author: Shahir
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from nnAudio import Spectrogram as nn_spectrogram

from noteify.core.config import (SAMPLE_RATE, MIN_FREQ, BINS_PER_OCTAVE, NUM_NOTES, NUM_BINS,
                                 HOP_LENGTH, SEGMENT_SAMPLES, SEGMENT_FRAMES, EPSILON)

class SpectrogramLayer(nn.Module):
    """
    Input: (batch_size, num_samples)
    Output: (batch_size, num_frames, freq_bins)
    """

    def __init__(self):
        super().__init__()

        self.spec_layer = nn_spectrogram.CQT1992v2(sr=SAMPLE_RATE,
                                                   fmin=MIN_FREQ,
                                                   n_bins=NUM_BINS,
                                                   bins_per_octave=BINS_PER_OCTAVE,
                                                   hop_length=HOP_LENGTH,
                                                   window='hann',
                                                   output_format='Magnitude',
                                                   trainable=False)
    
    def forward(self, x):
        z = self.spec_layer(x)
        z = torch.log(z + EPSILON)
        return z

class BasicConvBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, input):
        x = F.leaky_relu(self.bn1(self.conv1(input)), inplace=True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), inplace=True)
        
        return x

class ResNetFreqConvBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel1_size,
                 kernel2_size,
                 padding1_size=(0, 0),
                 padding2_size=(0, 0),
                 stride1_size=(1, 1),
                 stride2_size=(1, 1)):
        
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                               out_channels=out_channels,
                               kernel_size=kernel1_size,
                               stride=stride1_size,
                               padding=padding1_size,
                               bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                               out_channels=out_channels,
                               kernel_size=kernel2_size,
                               stride=stride2_size,
                               padding=padding2_size,
                               bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        residual = x
        x = F.leaky_relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.leaky_relu(x)

        return x

class AcousticCRNN(nn.Module):
    """
    Input: (batch_size, num_frames, freq_bins)
    Output: (batch_size, num_frames, num_notes)
    """
    
    def __init__(self):
        super().__init__()

        self.conv_block1 = BasicConvBlock(in_channels=1, out_channels=48)
        self.conv_block2 = BasicConvBlock(in_channels=48, out_channels=64)
        self.res_conv_block2 = ResNetFreqConvBlock(
            in_channels=64,
            out_channels=64,
            kernel1_size=(1, NUM_BINS//4 + 1),
            padding1_size=(0, NUM_BINS//8),
            kernel2_size=(9, 1),
            padding2_size=(4, 0)
        )
        self.conv_block3 = BasicConvBlock(in_channels=64, out_channels=96)
        self.conv_block4 = BasicConvBlock(in_channels=96, out_channels=128)

        self.pool = nn.AvgPool2d(kernel_size=(1, 2))

        self.num_conv_feats = 128 * (NUM_BINS//16)
        self.linear_feats = 768
        self.rnn_feats = 256

        self.fc1 = nn.Linear(self.num_conv_feats, self.linear_feats, bias=False)
        self.bn1 = nn.BatchNorm1d(self.linear_feats)

        self.rnn = nn.GRU(input_size=self.linear_feats, hidden_size=self.rnn_feats, num_layers=2, 
            bias=True, batch_first=True, dropout=0., bidirectional=True)

        self.fc2 = nn.Linear(self.rnn_feats*2, NUM_NOTES, bias=True)
        
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        # x is (batch_size, num_channels, num_frames, freq_bins)
              
        x = self.conv_block1(x) # (batch_size, num_channels, num_frames, freq_bins1)
        x = self.pool(x) # (batch_size, num_channels, num_frames, freq_bins2)

        x = self.conv_block2(x) # (batch_size, num_channels, num_frames, freq_bins2)
        x = self.pool(x) # (batch_size, num_channels, num_frames, freq_bins3)

        x = self.res_conv_block2(x)
        
        x = self.conv_block3(x) # (batch_size, num_channels, num_frames, freq_bins3)
        x = self.pool(x) # (batch_size, num_channels, num_frames, freq_bins4)

        x = self.conv_block4(x) # (batch_size, num_channels, num_frames, freq_bins4)
        x = self.pool(x) # (batch_size, num_channels, num_frames, freq_bins5)

        x =  x.transpose(1, 2).flatten(2) # (batch_size, time_steps, channels_num * freq_bins5)
        
        x = self.fc1(x) # (batch_size, time_steps, linear_feats)
        x = self.bn1(x.transpose(1, 2)).transpose(1, 2) # (batch_size, time_steps, linear_feats) normed on linear_feats axis

        x, _ = self.rnn(x) # (batch_size, time_steps, rnn_feats * 2)
        x = self.dropout(x)
        x = self.fc2(x) # (batch_size, time_steps, num_notes)

        return x

class TranscriptionNN(nn.Module):
    """
    Input: (batch_size, num_samples)
    Outputs: (batch_size, num_frames, num_notes)
    """
    
    def __init__(self):
        super().__init__()

        self.spec_layer = SpectrogramLayer()
        self.bn = nn.BatchNorm2d(NUM_BINS)

        self.frame_model = AcousticCRNN()
        self.onset_model = AcousticCRNN()
        self.offset_model = AcousticCRNN()

        self.rnn_feats = 256

        self.frame_rnn = nn.GRU(input_size=NUM_NOTES * 3, hidden_size=self.rnn_feats, num_layers=1, 
            bias=True, batch_first=True, dropout=0.0, bidirectional=True)
        self.frame_fc = nn.Linear(self.rnn_feats*2, NUM_NOTES, bias=True)
        self.dropout = nn.Dropout(0.5)

        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.zero_()
    
    def forward(self, x, apply_sigmoid=False):
        # x is (batch_size, num_samples)
        x = self.spec_layer(x) # (batch_size, freq_bins, num_frames)
        x = x.transpose(1, 2)
        x = x.unsqueeze(1) # (batch_size, num_channels, num_frames, freq_bins)
        x = x.transpose(1, 3)
        x = self.bn(x) # normalize spectrogram accross frequency
        x = x.transpose(1, 3) # (batch_size, num_channels, num_frames, freq_bins)

        frame_output = self.frame_model(x) # (batch_size, num_frames, num_notes)
        onset_output = self.onset_model(x) # (batch_size, num_frames, num_notes)
        offset_output = self.offset_model(x) # (batch_size, num_frames, num_notes)

        cond_input = torch.cat((frame_output,
                                onset_output.detach(),
                                offset_output.detach()), dim=2) # (batch_size, num_frames, num_notes * 3)
        frame_output, _ = self.frame_rnn(cond_input) # (batch_size, num_channels, num_frames, rnn_feats * 2)
        frame_output = self.dropout(frame_output)
        frame_output = self.frame_fc(frame_output) # (batch_size, num_channels, num_frames, num_notes)
        
        if apply_sigmoid:
            frame_output = torch.sigmoid(frame_output)
            onset_output = torch.sigmoid(onset_output)
            offset_output = torch.sigmoid(offset_output)
        
        outputs = {
            'reg_onset_output': onset_output, 
            'reg_offset_output': offset_output, 
            'frame_output': frame_output}
        
        return outputs

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    from torchsummary import summary
    model = AcousticCRNN()
    summary(model, (1, SEGMENT_FRAMES, NUM_BINS))

    model = TranscriptionNN()
    summary(model, (SEGMENT_SAMPLES,))
    
