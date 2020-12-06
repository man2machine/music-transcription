# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 14:11:37 2020

@author: Shahir
"""


import numpy as np
import matplotlib.pyplot as plt
import librosa
import mido

from noteify.core.config import SAMPLE_RATE

def load_wavfile(fname, resample_sr=SAMPLE_RATE):
    sr, x = wavfile.read(fname)
    x = x / (1 << 15)
    if x is not None:
        x = librosa.resample(x.astype(np.float), sr, resample_sr)
    return x

def plot_audio(x):
    fig = plt.figure(figsize=(10, 4))
    ax = plt.subplot()
    ax.set_xlabel("Amplitude")
    ax.set_ylabel("Samples")
    ax.plot(np.arange(0, len(x)), x)
    fig.show()

def plot_spectrogram(z):
    fig = plt.figure(figsize=(5, 10))
    ax = plt.subplot()
    ax.imshow(z.numpy()[0], cmap='inferno')
    ax.invert_yaxis()
    fig.show()

def plot_roll_info(roll_info):
    for roll_name, value in roll_info.items():
        fig = plt.figure(figsize=(10, 4))
        ax = plt.subplot()
        ax.set_title(roll_name)
        ax.imshow(value.T, interpolation=None, cmap='inferno')
        ax.invert_yaxis()
        fig.show()

def write_events_to_midi(start_time, note_events, midi_path):
    """Write out note events to MIDI file.

    Args:
      start_time: float
      note_events: list of dict, e.g. [
        {'midi_note': 51, 'onset_time': 696.63544, 'offset_time': 696.9948, 'velocity': 44}, 
        {'midi_note': 58, 'onset_time': 696.99585, 'offset_time': 697.18646, 'velocity': 50}
        ...]
      midi_path: str
    """
    
    # This configuration is the same as MIDIs in MAESTRO dataset
    ticks_per_beat = 384
    beats_per_second = 2
    ticks_per_second = ticks_per_beat * beats_per_second
    microseconds_per_beat = int(1e6 // beats_per_second)

    midi_file = mido.MidiFile()
    midi_file.ticks_per_beat = ticks_per_beat

    # Track 0
    track0 = mido.MidiTrack()
    track0.append(mido.MetaMessage('set_tempo', tempo=microseconds_per_beat, time=0))
    track0.append(mido.MetaMessage('time_signature', numerator=4, denominator=4, time=0))
    track0.append(mido.MetaMessage('end_of_track', time=1))
    midi_file.tracks.append(track0)

    # Track 1
    track1 = mido.MidiTrack()
    
    # Message rolls of MIDI
    message_infos = []

    for note_event in note_events:
        # Onset
        message_infos.append({
            'time': note_event['onset_time'], 
            'midi_note': note_event['midi_note'], 
            'velocity': note_event['velocity']})

        # Offset
        message_infos.append({
            'time': note_event['offset_time'], 
            'midi_note': note_event['midi_note'], 
            'velocity': 0})

    # Sort MIDI messages by time
    message_infos.sort(key=lambda n: n['time'])

    previous_ticks = 0
    for message in message_infos:
        current_ticks = int((message['time'] - start_time) * ticks_per_second)
        if current_ticks >= 0:
            diff_ticks = current_ticks - previous_ticks
            previous_ticks = current_ticks
            if 'midi_note' in message.keys():
                track1.append(mido.Message('note_on', note=message['midi_note'], velocity=message['velocity'], time=diff_ticks))
    track1.append(mido.MetaMessage('end_of_track', time=1))
    midi_file.tracks.append(track1)

    midi_file.save(midi_path)