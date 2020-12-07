# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 18:37:24 2020

@author: Shahir
"""

import numpy as np
import scipy as sp
from scipy import signal
import matplotlib.pyplot as plt
import librosa
import librosa.display

from notify.song import Note, DecodedSong

FMIN = 65.41 # 32.70 # C0
NUM_KEYS = 72
BINS_PER_NOTE = 4

def onset_filter(x):
    """
    Filters out negative values in the spectral flux, as in those values the
    amplitude of a note is not increasing
    """
    return np.maximum(0, x)

def get_onset_envelope(Sxx, ts, timestep=None, draw=False):
    """
    Gets the strength of note onsets throughout a recording given a
    spectrogram. Calculates spectral flux and then uses a Gaussian
    filter to find the onset strength envelope.
    """
    # calculate spectral flux
    dt = ts[1] - ts[0]
    if timestep is None:
        timestep = 0
    shift_len = int(timestep/dt) + 1
    n, m = Sxx.shape
    blank = np.zeros((n, shift_len))
    old = np.hstack((blank, Sxx))
    new = np.hstack((Sxx, blank))
    diff = (new - old)[:, :m]
    diff = onset_filter(diff)
    diff = np.sum(diff, axis=0)
    diff[:shift_len] = 0
    
    if draw:
        plt.plot(diff)
    
    # gaussian filter
    diff += 1
    size = 2
    window = signal.general_gaussian(size*2+1, p=1, sig=20)
    filtered = signal.fftconvolve(window, diff)
    filtered = (np.average(diff) / np.average(filtered)) * filtered
    filtered = np.roll(filtered, -size)
    diff = filtered
    diff = diff[:m]
    
    l = np.average(diff) - np.std(diff)
    smallest = (np.max(diff) - np.min(diff))*0.1 + np.min(diff)
    l = smallest
    diff[diff < l] = l
    if draw:
        plt.plot(diff)
        plt.show()
    
    return diff

def moving_average(a, n=3):
    """
    Moving average filter
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def get_note_bins_at_index(Sxx, fs, i, omit_notes=None, peak_note=False, draw=False, output=False):
    """
    Determines the peak note frequency/frequencies at a certain index of a
    spectrogram
    """
    if omit_notes is None:
        omit_notes = []
    
    # normalization and high pass filter
    Sx = Sxx[:, i].copy()
    low = np.average(Sx) #+ np.std(Sx)*1.3
    Sx[Sx < low] = low
    #Sx = sp.ndimage.maximum_filter1d(Sx, 4)
    #Sx = moving_average(Sx, n=4)
    a = Sx.copy()
    l = np.average(a) - np.std(a)
    smallest = (np.max(a) - np.min(a))*0.4 + np.min(a)
    l = smallest
    a[a < l] = l
    Sx = a
    peaks = signal.find_peaks(Sx)[0]
    if not len(peaks):
        return [[], [], [], []]
    amps = Sx[peaks]
    ind = np.argmax(amps)
    
    if draw:
        a = max(0, i-10)
        librosa.display.specshow(Sxx[:, i:i+80])
        plt.show()
        plt.plot(Sxx[:, i])
        plt.plot(Sx)
        plt.scatter(peaks, amps)
        plt.show()
    
    peak_notes = []
    for k in peaks:
        original_note = librosa.hz_to_note(fs[k])
        if original_note in omit_notes:
            continue
        peak_notes.append(original_note)
    
    peak_note_notes = [librosa.hz_to_note(fs[peaks[ind]])]
    peak_notes_original = [librosa.hz_to_note(fs[i]) for i in peaks]
    if output:
        print(peak_notes)
        print(peak_note_notes)
    if peak_note:
        return [[peaks[ind]], peak_note_notes, [fs[peaks[ind]]], [1]]
    
    # iterate through all keys in the piano and find harmonics
    all_notes = []
    for n in range(0, NUM_KEYS):
        all_notes.append(librosa.hz_to_note(FMIN*2**(n/12)))
    note_dict = {}
    for note in all_notes:
        for peak_note in peak_notes:
            peak_hz = librosa.note_to_hz(peak_note)
            possible_hz = librosa.note_to_hz(note)
            ratio = peak_hz/possible_hz
            if abs(ratio - round(ratio)) < 0.005:
                if note not in note_dict:
                    note_dict[note] = []
                if note not in omit_notes:
                    note_dict[note].append(peak_note)
    
    if len(note_dict) == 0:
        return [[peaks[ind]], peak_note_notes, [fs[peaks[ind]]], [1]]
    
    # recursively filter harmonics for each note
    last_considered_note = None
    filtered_notes = {}
    while True:
        max_harmonics = max(len(note_dict[x]) for x in note_dict)
        most_harmonic_notes = []
        for note in note_dict:
            if len(note_dict[note]) == max_harmonics:
                most_harmonic_notes.append(note)
        most_harmonic_notes = [(librosa.note_to_hz(n), n) for n in most_harmonic_notes]
        if len(most_harmonic_notes) == 0:
            break
        most_harmonic_note = sorted(most_harmonic_notes)[-1][1]
        harmonics = note_dict[most_harmonic_note][:]
        delete_harmonics = False
        for harmonic in harmonics:
            for note in note_dict:
                if note == most_harmonic_note:
                    continue
                if harmonic in note_dict[note]:
                    note_dict[note].remove(harmonic)
                    delete_harmonics = True
        if most_harmonic_note == last_considered_note and not delete_harmonics:
            if most_harmonic_note not in omit_notes:
                harmonics = [h for h in harmonics if h not in omit_notes]
                if len(harmonics):
                    filtered_notes[most_harmonic_note] = harmonics
            note_dict.pop(most_harmonic_note)
        if sum(len(note_dict[x]) for x in note_dict) == 0:
            break
        last_considered_note = most_harmonic_note
    if output:
        print(filtered_notes)
    note_freqs = [fs[peaks[ind]]]
    note_freqs = [librosa.note_to_hz(note) for note in filtered_notes.keys()]
    
    if len(filtered_notes) == 0:
        raise ValueError("Failed")
    
    # picks most harmonic note for monophonic output
    max_harmonics = max(len(filtered_notes[x]) for x in filtered_notes)
    most_harmonic_notes = []
    for note in filtered_notes:
        if len(filtered_notes[note]) == max_harmonics:
            most_harmonic_notes.append(note)
    most_harmonic_notes = [(librosa.note_to_hz(n), n) for n in most_harmonic_notes]
    most_harmonic_note = sorted(most_harmonic_notes)[0][1]
    note_freqs = [librosa.note_to_hz(most_harmonic_note)]
    if output:
        print(most_harmonic_note)
    
    return [[peaks[ind]], most_harmonic_note, note_freqs, [1]]

def detect_note_end(bin_val, onset_frame, Sxx, cutoff=0.5):
    """
    Looks ahead from a note onset in the spectrogram to find when the note ended
    """

    amps = Sxx[bin_val][onset_frame:]
    amps = (amps - np.min(amps))/(np.max(amps) - np.min(amps) + 1e-20)
    onset_amp = amps[0]
    end = np.argwhere(amps < onset_amp*cutoff)
    if not len(end):
        end = Sxx.shape[1] - 1 # end of song
    else:
        end = end[0][0] + onset_frame
    
    return end

def decode_song(x, sr, detect_ends=True, draw=False):    
    """
    Given an audio signal, returns the transcribed music as a DecodedSong class
    """
    n_bins = NUM_KEYS*BINS_PER_NOTE
    bins_per_octave = 12*BINS_PER_NOTE
    Cxx = librosa.cqt(x, sr=sr, n_bins=n_bins, bins_per_octave=bins_per_octave, fmin=FMIN)
    fs = librosa.cqt_frequencies(n_bins, FMIN, bins_per_octave)
    ts = librosa.frames_to_time(np.arange(Cxx.shape[1]), sr=sr)
    
    Sxx = librosa.amplitude_to_db(np.abs(Cxx)**2)
    if draw:
        librosa.display.specshow(Sxx, sr=sr, x_axis='time', y_axis='off')
        plt.show()
    
    # compute onset strength envelope
    onset_envelope = get_onset_envelope(Sxx, ts, draw=draw)
    times = librosa.frames_to_time(np.arange(len(onset_envelope)), sr=sr)
    
    min_note_length = 0.075*sr//512 + 1 # ~ 75ms
    #onset_envelope = librosa.onset.onset_strength(y=y, sr=sr, feature=librosa.cqt)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_envelope, sr=sr, wait=min_note_length)
    
    # look ahead of note onsets to find best frames to detect note
    note_detect_frames = []
    for n in range(len(onset_frames)):
        if n == len(onset_frames) - 1:
            next_onset = len(onset_envelope) - 1
        else:
            next_onset = onset_frames[n + 1]
        onset_start = onset_frames[n]
        note_frame = int((next_onset - onset_start)*0.5) + onset_start
        diff = np.diff(onset_envelope[onset_start:])
        decreasing = np.argwhere(diff < 0).flatten()
        if len(decreasing): # found a place to detect note
            onset_forward_shift = int(0.15//(ts[1] - ts[0]) + 1)
            note_frame2 = decreasing[0] + onset_forward_shift + onset_start
            note_frame = min(note_frame, note_frame2) # don't detect after next note
        note_detect_frames.append(note_frame)
    note_detect_frames = np.array(note_detect_frames)
    
    onset_times = ts[onset_frames]
    note_detect_times = ts[note_detect_frames]
    
    if draw: # display onset envelope, onsets and note detection frames
        plt.plot(times, onset_envelope, label='Onset strength')
        plt.vlines(onset_times, 0, onset_envelope.max(), color='r', alpha=0.9,
                   linestyle='--', label='Onsets')
        plt.vlines(note_detect_times, 0, onset_envelope.max(), color='g', alpha=0.9,
                   linestyle='--', label='Note detect')
        plt.axis('tight')
        plt.legend(frameon=True, framealpha=0.75)
        plt.show()
    
    Sxx = librosa.amplitude_to_db(np.power(np.abs(Cxx).T, np.linspace(3, 2, num=Cxx.shape[0])).T)
    omit_notes = []
    song = DecodedSong([])
    for n in list(range(len(onset_frames))): # iterate through onsets and add notes
        onset_forward_shift = int(0.05//(ts[1] - ts[0]) + 1)
        t = onset_times[n]
        if n == len(onset_frames) - 1:
            tnext = ts[-1]
        else:
            tnext = onset_times[n + 1]
        i = onset_frames[n] + onset_forward_shift
        i = note_detect_frames[n]
        i = min(i, Sxx.shape[1] - 1)
        bins, notes, freqs, volumes = get_note_bins_at_index(Sxx, fs, i, omit_notes=omit_notes, peak_note=False, draw=False)
        omit_notes = notes
        
        # if last note has NOT ended next note has to be of different frequency
        end_i = detect_note_end(bins[0], i, Sxx)
        tnext_detect = ts[end_i]
        if tnext_detect < tnext:
            omit_notes = []
        if freqs:
            for n in range(len(freqs)):
                freq = freqs[n]
                if detect_ends: 
                    tnext = min(tnext_detect, tnext)
                song.notes.append(Note(freq, t, tnext - t, volume=1))
    
    return song

if __name__ == '__main__':
    from wav_utils import wav_read, wav_write

    fname = 'example_audio/fantasia'

    x, sr = wav_read(fname+".wav")
    
    t_cutoff = 2*sr
    x = x[:t_cutoff]
    song = decode_song(x, sr, draw=True)
    #print([n.note for n in song.notes])
    y = song.resynthesize_song(sr)
    wav_write(y, sr, fname+"_resynthesized.wav")