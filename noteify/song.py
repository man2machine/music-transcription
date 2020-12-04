# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 18:37:41 2020

@author: Shahir
"""

import collections
import json

import librosa
import numpy as np

from noteify.utils import JSONDictSerializable

class Note(JSONDictSerializable):
    def __init__(self, freq, start_time, duration, volume=1):
        assert 0 <= volume <= 1
        assert start_time >= 0
        assert freq > 0
        assert duration > 0

        # detremine note and MIDI value using librosa
        self._actual_freq = freq
        self.freq = librosa.note_to_hz(librosa.hz_to_note(freq))
        self.note = librosa.hz_to_note(self.freq)
        self.midi = int(librosa.hz_to_midi(self.freq))
        self.start_time = start_time
        self.duration = duration
        self.volume = volume

    def __repr__(self):
        return str(self.to_dict())
    
    def get_freq(self):
        """
        Returns the frequency of the note
        """
        return self.freq
    
    def get_note(self):
        """
        Returns the note as a string (ex. C#4)
        """
        return self.note
    
    def get_midi(self):
        """
        Returns MIDI value of note as integer
        """
        return self.midi
    
    def get_start_time(self):
        """
        Returns start time of note
        """
        return self.start_time
    
    def get_duration(self):
        """
        Returns duration of note
        """
        return self.duration
    
    def get_volume(self):
        """
        Returns volume of note
        """
        return self.volume
    
    def to_dict(self):
        data = {'freq': self._actual_freq,
                'start_time': self.start_time,
                'duration': self.duration,
                'volume': self.volume}
        return data
    
    @classmethod
    def from_dict(cls, data):
        note = cls(data['freq'], data['start_time'], data['duration'], volume=data['volume'])
        return note

class DecodedSong(JSONDictSerializable):
    def __init__(self, notes):
        self.notes = list(notes) # stores list of Note objects
    
    def resynthesize_song(self, sr, max_amp=0.7):
        """
        Resynthesizes the song by taking each note from the note list and
        adding a cosine wave with a start time, frequency and duration
        corresponding to the Note object.
        Normalizes the output audio to the maximum amplitude specified by max_amp
        """

        song_length = 0
        for note in self.notes:
            song_length = max(song_length, note.get_start_time() + note.get_duration())
        x = np.zeros(int(sr*song_length + 1))
        for note in self.notes:
            ts = np.arange(0, note.get_duration(), 1/sr)
            freq = note.get_freq()
            note_x = np.sin(2*np.pi*ts*freq)*note.get_volume()
            start_index = int(note.start_time*sr)
            end_index = start_index + len(note_x)
            x[start_index:end_index] += note_x
        x = 2*(x - x.min())/(x.max() - x.min()) - 1
        x *= max_amp

        return x
    
    def to_dict(self):
        data = [n.to_dict() for n in self.notes]
        return data
    
    @classmethod
    def from_dict(cls, data):
        notes = [Note.from_dict(n) for n in data]
        song = cls(notes)
        return song
