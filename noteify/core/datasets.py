# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 14:12:40 2020

@author: Shahir
"""

import os
import mmap
import pickle
import errno
import math
import json
import csv
import subprocess
import tarfile
import urllib.request
import collections

import numpy as np
import torch
import torch.utils.data as torch_data
import librosa
import sox
from scipy.io import wavfile
from intervaltree import IntervalTree
from tqdm import tqdm
import mido

from noteify.core.config import (
    SAMPLE_RATE, HOP_LENGTH, NUM_NOTES, MIN_MIDI,
    SEGMENT_LENGTH, SEGMENT_SAMPLES, SEGMENT_FRAMES,
    FRAMES_PER_SECOND)

SIZE_FLOAT = 4 # size of a float

def create_note_event(midi_note, onset_time, offset_time, velocity=127):
    return {'midi_note': midi_note,
            'onset_time': onset_time,
            'offset_time': offset_time,
            'velocity': velocity}

def create_pedal_event(onset_time, offset_time):
    return {'onset_time': onset_time,
            'offset_time': offset_time}

class MusicNetDataset:
    DATA_URL = "https://homes.cs.washington.edu/~thickstn/media/musicnet.tar.gz"
    METADATA_URL = "https://homes.cs.washington.edu/~thickstn/media/musicnet_metadata.csv"
    RAW_FOLDER = "raw"
    TRAIN_DATA_DIR = "train_data"
    TRAIN_LABELS_DIR = "train_labels"
    TRAIN_TREE_FNAME = "train_tree.pckl"
    TEST_DATA_DIR = "test_data"
    TEST_LABELS_DIR = "test_labels"
    TEST_TREE_FNAME = "test_tree.pckl"
    EXTRACTED_FOLDERS = [TRAIN_DATA_DIR, TRAIN_LABELS_DIR, TEST_DATA_DIR, TEST_LABELS_DIR]
    INPUT_SAMPLE_RATE = 44100
    BIN_SAMPLE_RATE = SAMPLE_RATE

    def __init__(self,
                 data_dir,
                 download=False,
                 refresh_cache=False,
                 delete_wav=False,
                 train=True,
                 use_mmap=False,
                 numpy_cache=False,
                 numpy_resample_sr=None,
                 piano_only=False,
                 filter_records=None):

        self.root_dir = os.path.abspath(os.path.expanduser(data_dir))
        self.refresh_cache = refresh_cache
        self.delete_wav = delete_wav
        self.train = train
        self.use_mmap = use_mmap
        self.numpy_cache = numpy_cache
        self.numpy_resample_sr = numpy_resample_sr
        
        self.sample_rate = self.BIN_SAMPLE_RATE
        if self.numpy_resample_sr:
            assert numpy_cache
            self.sample_rate = self.numpy_resample_sr
        self.piano_only = piano_only
        self.filter_records = filter_records

        found_data = False
        if download:
            found_data = self.download()
        else:
            found_data = self.check_exists()

        if not found_data:
            raise RuntimeError("Dataset not found")

        if train:
            self.data_path = os.path.join(self.root_dir, self.TRAIN_DATA_DIR)
            labels_path = os.path.join(
                self.root_dir, self.TRAIN_LABELS_DIR, self.TRAIN_TREE_FNAME)
        else:
            self.data_path = os.path.join(self.root_dir, self.TEST_DATA_DIR)
            labels_path = os.path.join(
                self.root_dir, self.TEST_LABELS_DIR, self.TEST_TREE_FNAME)

        with open(labels_path, 'rb') as f:
            self.labels = pickle.load(f)

        self.rec_ids = []
        self.records = dict()
        self.open_files = []
        
        if self.piano_only:
            with open(os.path.join(self.root_dir, "musicnet_metadata.csv")) as f:
                data = f.read().split('\n')[1:-1]

            data = [n.split(",") for n in data]
            piano_records = []
            for item in data:
                rec_id = int(item[0])
                has_piano = "piano" in item[4].lower()
                if has_piano:
                    piano_records.append(rec_id)
            self.filter_records = piano_records

        print("Loading audio")
        for record in tqdm(os.listdir(self.data_path)):
            if not record.endswith('.bin'):
                continue
            rec_id = int(record[:-4])
            if self.filter_records is not None:
                if rec_id not in self.filter_records:
                    continue
            self.rec_ids.append(rec_id)

            fname = os.path.join(self.data_path, record)
            if self.use_mmap:
                fd = os.open(fname, os.O_RDONLY)
                buf = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
                self.records[rec_id] = (buf, len(buf)//SIZE_FLOAT)
                self.open_files.append(fd)
            if self.numpy_cache:
                f = open(fname, 'rb')
                x = np.fromfile(f, dtype=np.float32)
                if self.numpy_resample_sr:
                    x = librosa.resample(x, self.INPUT_SAMPLE_RATE, self.sample_rate)
                self.records[rec_id] = (x, len(x))
                f.close()
            else:
                f = open(fname)
                self.records[int(record[:-4])] = (os.path.join(self.data_path, record),
                                                  os.fstat(f.fileno()).st_size//SIZE_FLOAT)
                f.close()

    def close(self):
        if self.use_mmap:
            for mm in self.records.values():
                mm[0].close()
            for fd in self.open_files:
                os.close(fd)
            self.records = dict()
            self.open_files = []

    def __close__(self):
        self.close()

    def check_exists(self):
        paths = (os.path.join(self.root_dir, self.TRAIN_DATA_DIR),
                 os.path.join(self.root_dir, self.TEST_DATA_DIR),
                 os.path.join(self.root_dir, self.TRAIN_LABELS_DIR,
                              self.TRAIN_TREE_FNAME),
                 os.path.join(self.root_dir, self.TEST_LABELS_DIR, self.TEST_TREE_FNAME))
        exists = all(os.path.exists(n) for n in paths)

        return exists and not self.refresh_cache

    def download(self):
        if self.check_exists():
            return True

        os.makedirs(os.path.join(self.root_dir,
                                 self.RAW_FOLDER), exist_ok=True)

        filename = self.DATA_URL.rpartition('/')[2]
        file_path = os.path.join(self.root_dir, self.RAW_FOLDER, filename)
        if not os.path.exists(file_path):
            print("Downloading", self.DATA_URL)
            data = urllib.request.urlopen(self.DATA_URL)
            with open(file_path, 'wb') as f:
                # stream the download to disk
                while True:
                    chunk = data.read(16*1024)
                    if not chunk:
                        break
                    f.write(chunk)

        found = all(map(lambda f: os.path.exists(
            os.path.join(self.root_dir, f)), self.EXTRACTED_FOLDERS))
        if not found:
            print("Extracting", filename)
            if subprocess.call(["tar", "-xf", file_path, '-C', self.root_dir, '--strip', '1']) != 0:
                raise OSError("Failed tarball extraction")
        
        filename = self.METADATA_URL.rpartition('/')[2]
        file_path = os.path.join(self.root_dir, filename)
        if not os.path.exists(file_path):
            print("Downloading", self.METADATA_URL)
            data = urllib.request.urlopen(self.METADATA_URL)
            with open(file_path, 'wb') as f:
                # stream the download to disk
                while True:
                    chunk = data.read(16*1024)
                    if not chunk:
                        break
                    f.write(chunk)
        
        print("Processing")

        self.process_data(self.TEST_DATA_DIR)
        trees = self.process_labels(self.TEST_LABELS_DIR)
        with open(os.path.join(self.root_dir, self.TEST_LABELS_DIR, self.TEST_TREE_FNAME), 'wb') as f:
            pickle.dump(trees, f)

        self.process_data(self.TRAIN_DATA_DIR)
        trees = self.process_labels(self.TRAIN_LABELS_DIR)
        with open(os.path.join(self.root_dir, self.TRAIN_LABELS_DIR, self.TRAIN_TREE_FNAME), 'wb') as f:
            pickle.dump(trees, f)

        self.refresh_cache = False
        print("Download Complete")

        return True
    
    def process_data(self, path):
        sub_dir = os.path.join(self.root_dir, path)
        for item in tqdm(os.listdir(sub_dir)):
            if not item.endswith('.wav'):
                continue
            sr, data = wavfile.read(os.path.join(self.root_dir, path, item))
            assert sr == self.INPUT_SAMPLE_RATE
            if self.BIN_SAMPLE_RATE != self.INPUT_SAMPLE_RATE:
                data = librosa.resample(data, self.INPUT_SAMPLE_RATE, self.BIN_SAMPLE_RATE)
            data.tofile(os.path.join(self.root_dir, path, item[:-4]+'.bin'))
            if self.delete_wav:
                os.remove(os.path.join(sub_dir, item))
    
    def process_labels(self, path):
        trees = dict()
        for item in tqdm(os.listdir(os.path.join(self.root_dir, path))):
            if not item.endswith('.csv'):
                continue
            rec_id = int(item[:-4])
            tree = IntervalTree()
            with open(os.path.join(self.root_dir, path, item), 'r') as f:
                reader = csv.DictReader(f, delimiter=',')
                for label in reader:
                    start_sample = int(label['start_time'])
                    end_sample = int(label['end_time'])
                    instrument = int(label['instrument'])
                    note = int(label['note'])
                    start_beat = float(label['start_beat'])
                    end_beat = float(label['end_beat'])
                    note_value = label['note_value']
                    tree[start_sample:end_sample] = {
                        'start_sample': start_sample,
                        'end_sample': end_sample,
                        'instrument': instrument,
                        'note': note,
                        'start_beat': start_beat,
                        'end_beat': end_beat,
                        'note_value': note_value
                    }
            trees[rec_id] = tree
        return trees

    def get_record_num_samples(self, rec_id):
        """
        Outputs (sample rate, num samples)
        """

        return self.records[rec_id][1]

    def get_record_data(self, rec_id, start_sample=None, end_sample=None):
        if start_sample is None:
            start_sample = 0
        if end_sample is None:
            end_sample = self.records[rec_id][1]
        
        start_pos = int(start_sample)*SIZE_FLOAT
        end_pos = int(end_sample)*SIZE_FLOAT
        if self.use_mmap:
            x = np.frombuffer(self.records[rec_id][0][start_pos:end_pos], dtype=np.float32).copy()
        elif self.numpy_cache:
            x = self.records[rec_id][0][start_sample:end_sample].copy()
        else:
            with open(self.records[rec_id][0], 'rb') as f:
                f.seek(start_pos, os.SEEK_SET)
                x = np.fromfile(f, dtype=np.float32, count=(end_sample - start_sample))
        
        factor = self.INPUT_SAMPLE_RATE/self.sample_rate
        start_sample = math.floor(start_sample*factor)
        end_sample = math.ceil(end_sample*factor)
        intervals = self.labels[rec_id][start_sample:end_sample + 1]
        note_infos = [n.data.copy() for n in intervals]
        
        factor = self.sample_rate/self.INPUT_SAMPLE_RATE
        for n, info in enumerate(note_infos):
            note_infos[n]['start_sample'] = info['start_sample']*factor
            note_infos[n]['end_sample'] = info['end_sample']*factor

        return x, note_infos

class MaestroDataset:
    BIN_SAMPLE_RATE = SAMPLE_RATE

    def __init__(self,
                 data_dir,
                 download=False,
                 refresh_cache=False,
                 delete_wav=False,
                 train=True,
                 test=False,
                 validation=False,
                 use_mmap=False,
                 numpy_cache=False,
                 numpy_resample_sr=None,
                 filter_records=None,
                 reduce_record_factor=None):

        self.root_dir = os.path.abspath(os.path.expanduser(data_dir))
        self.refresh_cache = refresh_cache
        self.delete_wav = delete_wav
        self.train = train
        self.test = test
        self.validation = validation
        self.use_mmap = use_mmap
        self.numpy_cache = numpy_cache
        self.numpy_resample_sr = numpy_resample_sr
        
        self.sample_rate = self.BIN_SAMPLE_RATE
        if self.numpy_resample_sr:
            assert numpy_cache
            self.sample_rate = self.numpy_resample_sr
        
        if self.numpy_cache or self.use_mmap:
            raise ValueError("This is cannot fit in memory")

        if not self.check_dataset_exists():
            raise ValueError("Could not find dataset")

        with open(os.path.join(self.root_dir, "maestro-v3.0.0.json")) as f:
            orig_metadata = json.load(f)

        self.metadata = {}
        for category in orig_metadata.keys():
            category_data = orig_metadata[category]
            
            for element_id in category_data.keys():
                if element_id not in self.metadata.keys():
                    self.metadata[element_id] = {}
                self.metadata[element_id][category] = orig_metadata[category][element_id]
    
        self.all_rec_ids = list(self.metadata.keys())
        self.rec_ids = []
        if self.train:
            self.rec_ids = [n for n in self.all_rec_ids if self.metadata[n]['split'] == 'train']
        else:
            self.rec_ids = [n for n in self.all_rec_ids if self.metadata[n]['split'] == 'test']
        if self.validation:
            self.rec_ids += [n for n in self.all_rec_ids if self.metadata[n]['split'] == 'validation'] 
        
        if filter_records is not None:
            self.rec_ids = [n for n in self.rec_ids if n in filter_records]

        if reduce_record_factor:
            self.rec_ids = self.rec_ids[:int(len(self.rec_ids)/reduce_record_factor)]

        if not self.check_processed():
            print("Processing data")
            self.process_data()
        
        self.records = dict()
        self.open_files = []
        self.sample_rate = self.BIN_SAMPLE_RATE

        print("Loading audio")
        for rec_id in tqdm(self.rec_ids):
            audio_fname = self.metadata[rec_id]['audio_filename']
            midi_fname = self.metadata[rec_id]['midi_filename']
            tree_fname = audio_fname[:-4]+'_tree.pckl'
            events_fname = audio_fname[:-4]+'_events.pckl'
            bin_fname = audio_fname[:-4]+'.bin'

            audio_fname = os.path.join(self.root_dir, audio_fname)
            midi_fname = os.path.join(self.root_dir, midi_fname)
            tree_fname = os.path.join(self.root_dir, tree_fname)
            events_fname = os.path.join(self.root_dir, events_fname)
            bin_fname = os.path.join(self.root_dir, bin_fname)
            
            if self.use_mmap:
                fd = os.open(bin_fname, os.O_RDONLY)
                buf = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
                self.records[rec_id] = (buf, len(buf)//SIZE_FLOAT,
                    bin_fname, tree_fname, events_fname)
                self.open_files.append(fd)
            if self.numpy_cache:
                f = open(bin_fname, 'rb')
                x = np.fromfile(f, dtype=np.float32)
                if self.numpy_resample_sr:
                    x = librosa.resample(x, self.BIN_SAMPLE_RATE, self.sample_rate)
                self.records[rec_id] = (x, len(x),
                    bin_fname, tree_fname, events_fname)
                f.close()
            else:
                f = open(bin_fname)
                fsize = os.fstat(f.fileno()).st_size//SIZE_FLOAT
                self.records[rec_id] = (bin_fname, fsize,
                    bin_fname, tree_fname, events_fname)
                f.close()
    
    def close(self):
        if self.use_mmap:
            for mm in self.records.values():
                mm[0].close()
            for fd in self.open_files:
                os.close(fd)
            self.records = dict()
            self.open_files = []

    def __close__(self):
        self.close()

    def check_dataset_exists(self):
        return os.path.exists(os.path.join(self.root_dir, "maestro-v3.0.0.json"))
    
    def check_processed(self):
        check_fname = os.path.join(self.root_dir, "check.json")
        if os.path.exists(check_fname):
            with open(check_fname, 'r') as f:
                check = json.load(f)
            processed_ids = check['processed_ids']
            for rec_id in self.rec_ids:
                if rec_id not in processed_ids:
                    return False
            return True
        return False
    
    def get_record_length(self, rec_id):
        return self.metadata[rec_id]['duration']
    
    def get_record_num_samples(self, rec_id):
        return self.records[rec_id][1]

    def process_data(self):
        pbar = tqdm(self.rec_ids)
        for rec_id in pbar:
            audio_fname = self.metadata[rec_id]['audio_filename']
            bin_fname = audio_fname[:-4]+'.bin'
            pbar.set_description(audio_fname)

            bin_fname = os.path.join(self.root_dir, bin_fname)
            exist = os.path.exists(bin_fname)
            if not exist or (exist and os.path.getsize(bin_fname) == 0):
                data, sr = librosa.core.load(os.path.join(self.root_dir, audio_fname),
                    mono=True, sr=self.BIN_SAMPLE_RATE)
                data.tofile(bin_fname)
        
        pbar = tqdm(self.rec_ids)
        for rec_id in pbar:
            midi_fname = self.metadata[rec_id]['midi_filename']
            audio_fname = self.metadata[rec_id]['audio_filename']
            events_fname = audio_fname[:-4]+'_events.pckl'
            tree_fname = audio_fname[:-4]+'_tree.pckl'
            pbar.set_description(midi_fname)

            midi_fname = os.path.join(self.root_dir, midi_fname)

            events_fname = os.path.join(self.root_dir, events_fname)
            event_exist = os.path.exists(events_fname)
            tree_fname = os.path.join(self.root_dir, tree_fname)
            tree_exist = os.path.exists(tree_fname)

            if (not event_exist) or (not tree_exist):
                event_info, tree_info = self.get_label_info(midi_fname)
            
            if not event_exist:
                with open(events_fname, 'wb') as f:
                    pickle.dump(event_info, f)
            
            if not tree_exist:
                with open(tree_fname, 'wb') as f:
                    pickle.dump(tree_info, f)
        
        check_fname = os.path.join(self.root_dir, "check.json")
        processed_ids = set(self.rec_ids)
        if os.path.exists(check_fname):
            with open(check_fname, 'r') as f:
                check = json.load(f)
            processed_ids.update(check['processed_ids'])
        
        check = {'processed_ids': list(processed_ids)}
        with open(check_fname, 'w') as f:
            json.dump(check, f)
    
    def read_midi(self, midi_fname):
        midi_file = mido.MidiFile(midi_fname)
        ticks_per_beat = midi_file.ticks_per_beat

        assert len(midi_file.tracks) == 2
        # The first track contains tempo, time signature. The second track contains piano events

        microseconds_per_beat = midi_file.tracks[0][0].tempo
        beats_per_second = 1e6 / microseconds_per_beat
        ticks_per_second = ticks_per_beat * beats_per_second

        message_list = []

        ticks = 0
        time_in_second = []

        for message in midi_file.tracks[1]:
            message_list.append(message)
            ticks += message.time
            time_in_second.append(ticks / ticks_per_second)

        result = {
            'midi_events': message_list, 
            'midi_event_times': np.array(time_in_second)}
        
        return result
    
    def extend_pedal(self, note_events, pedal_events):
        note_events = collections.deque(note_events)
        pedal_events = collections.deque(pedal_events)
        ex_note_events = []

        index = 0 # Index of note events
        while pedal_events: # Go through all pedal events
            pedal_event = pedal_events.popleft()
            note_buffer = {} # keys: midi notes, value for each key: event index

            while note_events:
                note_event = note_events.popleft()

                # If a note offset is between the onset and offset of a pedal, 
                # Then set the note offset to when the pedal is released.
                if pedal_event['onset_time'] < note_event['offset_time'] < pedal_event['offset_time']:
                    
                    midi_note = note_event['midi_note']

                    if midi_note in note_buffer.keys():
                        # Multiple same note inside a pedal
                        _idx = note_buffer[midi_note]
                        note_buffer.pop(midi_note)
                        ex_note_events[_idx]['offset_time'] = note_event['onset_time']

                    # Set note offset to pedal offset
                    note_event['offset_time'] = pedal_event['offset_time']
                    note_buffer[midi_note] = index
                
                ex_note_events.append(note_event)
                index += 1

                # Break loop and pop next pedal
                if note_event['offset_time'] > pedal_event['offset_time']:
                    break

        while note_events:
            # Append any other notes
            ex_note_events.append(note_events.popleft())
        
        return ex_note_events
    
    def parse_midi(self, midi_info, extend_pedal=True):
        midi_events = midi_info['midi_events']
        midi_event_times = midi_info['midi_event_times']
        start_time = min(midi_event_times)
        end_time = max(midi_event_times)

        note_events = []
        pedal_events = []

        note_buffer = {}
        pedal_buffer = {}

        for i in range(len(midi_events)):
            event = midi_events[i]
            event_time = midi_event_times[i]
            if event.type == 'note_on' or event.type == 'note_off':
                midi_note = event.note
                velocity = event.velocity

                if event.type == 'note_on' and velocity > 0:
                    note_buffer[midi_note] = {
                        'onset_time': event_time,
                        'velocity': velocity}
                else:
                    if midi_note in note_buffer.keys():
                        note_events.append(create_note_event(
                            midi_note,
                            note_buffer[midi_note]['onset_time'],
                            event_time,
                            note_buffer[midi_note]['velocity']))
                        note_buffer.pop(midi_note)
            
            elif event.type == 'control_change' and event.control == 64:
                pedal_value = event.value
                if pedal_value >= 64:
                    if 'onset_time' not in pedal_buffer:
                        pedal_buffer['onset_time'] = event_time
                else:
                    if 'onset_time' in pedal_buffer:
                        pedal_events.append(create_pedal_event(
                            pedal_buffer['onset_time'],
                            event_time))
                        pedal_buffer = {}
        
        for midi_note, info in note_buffer.items():
            note_events.append(create_note_event(
                midi_note,
                info['onset_time'],
                end_time,
                info['velocity']))
        
        if pedal_buffer:
            pedal_events.append(create_pedal_event(
                pedal_buffer['onset_time'],
                end_time))

        # Set notes to on until pedal is released
        if extend_pedal:
            note_events = self.extend_pedal(note_events, pedal_events)
    
        return note_events, pedal_events
    
    def get_label_info(self, midi_fname):
        midi_info = self.read_midi(midi_fname)
        note_events, pedal_events = self.parse_midi(midi_info)

        note_tree = IntervalTree()
        pedal_tree = IntervalTree()

        for note_event in note_events:
            start_sample = math.floor(note_event['onset_time']*self.BIN_SAMPLE_RATE)
            end_sample = math.floor(note_event['offset_time']*self.BIN_SAMPLE_RATE)
            note_tree[start_sample:end_sample] = note_event
        
        for pedal_event in pedal_events:
            start_sample = math.floor(pedal_event['onset_time']*self.BIN_SAMPLE_RATE)
            end_sample = math.floor(pedal_event['offset_time']*self.BIN_SAMPLE_RATE)
            pedal_tree[start_sample:end_sample] = pedal_events
        
        event_info = {'note_events': note_events, 'pedal_events': pedal_events}
        tree_info = {'note_tree': note_tree, 'pedal_tree': pedal_tree}

        return event_info, tree_info
    
    def get_record_data(self, rec_id, start_sample=None, end_sample=None):
        if start_sample is None:
            start_sample = 0
        if end_sample is None:
            end_sample = self.records[rec_id][1]
        
        bin_fname = self.records[rec_id][2]
        start_pos = int(start_sample)*SIZE_FLOAT
        end_pos = int(end_sample)*SIZE_FLOAT
        if self.use_mmap:
            x = np.frombuffer(self.records[rec_id][0][start_pos:end_pos], dtype=np.float32).copy()
        elif self.numpy_cache:
            x = self.records[rec_id][0][start_sample:end_sample].copy()
        else:
            with open(bin_fname, 'rb') as f:
                f.seek(start_pos, os.SEEK_SET)
                x = np.fromfile(f, dtype=np.float32, count=(end_sample - start_sample))
        
        tree_fname = self.records[rec_id][3]
        with open(tree_fname, 'rb') as f:
            tree = pickle.load(f)
        
        intervals = tree['note_tree'][start_sample:end_sample + 1]
        note_events = [n.data.copy() for n in intervals]

        return x, note_events

class MusicAugmentor:
    def __init__(self, noise=True):
        self.rng = np.random.default_rng()
        self.noise = noise

    def augment(self, x):
        clip_samples = len(x)

        tfm = sox.Transformer()
        tfm.set_globals(verbosity=0)

        tfm.pitch(self.rng.uniform(-0.1, 0.1, 1)[0])
        tfm.contrast(self.rng.uniform(0, 100, 1)[0])

        tfm.equalizer(frequency=self.loguniform(32, 4096, 1)[0],
                      width_q=self.rng.uniform(1, 2, 1)[0],
                      gain_db=self.rng.uniform(-30, 10, 1)[0])

        tfm.equalizer(frequency=self.loguniform(32, 4096, 1)[0],
                      width_q=self.rng.uniform(1, 2, 1)[0],
                      gain_db=self.rng.uniform(-30, 10, 1)[0])

        tfm.reverb(reverberance=self.rng.uniform(0, 70, 1)[0])

        aug_x = tfm.build_array(input_array=x, sample_rate_in=SAMPLE_RATE)
        if len(aug_x) < clip_samples:
            aug_x = np.concatenate(
                (aug_x, np.zeros(clip_samples - len(aug_x))))
        else:
            aug_x = aug_x[:clip_samples]
        aug_x = aug_x.copy()

        if self.noise and self.rng.uniform(0, 1) > 0.7:
            aug_x += self.rng.normal(0, self.rng.uniform(0, 0.0014), len(aug_x))

        return aug_x

    def loguniform(self, low, high, size):
        return np.exp(self.rng.uniform(np.log(low), np.log(high), size))

def get_regression(reg_input):
    """
    Get regression target. See Fig. 2 of [1] for an example.
    [1] Q. Kong, et al., High-resolution Piano Transcription with Pedals by 
    Regressing Onsets and Offsets Times, 2020.

    Input: (frames_num,) 

    Output: (frames_num,), e.g., [0, 0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.9, 0.7, 0.5, 0.3, 0.1, 0, 0, ...]
    """
    step = 1. / FRAMES_PER_SECOND
    output = np.ones_like(reg_input)

    locts = np.where(reg_input < 0.5)[0]
    if len(locts) > 0:
        for t in range(0, locts[0]):
            output[t] = step * (t - locts[0]) - reg_input[locts[0]]

        for i in range(0, len(locts) - 1):
            for t in range(locts[i], (locts[i] + locts[i + 1]) // 2):
                output[t] = step * (t - locts[i]) - reg_input[locts[i]]

            for t in range((locts[i] + locts[i + 1]) // 2, locts[i + 1]):
                output[t] = step * (t - locts[i + 1]) - reg_input[locts[i]]

        for t in range(locts[-1], len(reg_input)):
            output[t] = step * (t - locts[-1]) - reg_input[locts[-1]]

    output = np.clip(np.abs(output), 0., 0.05) * 20
    output = (1. - output)

    return output

def generate_rolls(note_events, start_time, end_time):
    frame_roll = np.zeros((SEGMENT_FRAMES, NUM_NOTES))
    onset_roll = np.zeros((SEGMENT_FRAMES, NUM_NOTES))
    offset_roll = np.zeros((SEGMENT_FRAMES, NUM_NOTES))
    reg_onset_roll = np.ones((SEGMENT_FRAMES, NUM_NOTES))
    reg_offset_roll = np.ones((SEGMENT_FRAMES, NUM_NOTES))
    velocity_roll = np.zeros((SEGMENT_FRAMES, NUM_NOTES))
    mask_roll = np.ones((SEGMENT_FRAMES, NUM_NOTES))

    for note_event in note_events:
        note_index = note_event['midi_note'] - MIN_MIDI
        note_index = np.clip(note_index, 0, NUM_NOTES)

        if 0 <= note_index <= NUM_NOTES:
            onset_time = note_event['onset_time']
            offset_time = note_event['offset_time']

            if onset_time >= end_time:
                continue

            start_frame = int(
                round((onset_time - start_time)*FRAMES_PER_SECOND))
            end_frame = int(
                round((offset_time - start_time)*FRAMES_PER_SECOND))
            
            if start_frame >= SEGMENT_FRAMES:
                continue

            if end_frame >= 0:
                clip_start_frame = max(start_frame, 0)

                frame_roll[clip_start_frame:end_frame+1, note_index] = 1
                velocity_roll[clip_start_frame:end_frame +
                              1, note_index] = note_event['velocity']
                
                if end_frame > SEGMENT_FRAMES - 1:
                    mask_roll[start_frame:, note_index] = 0
                else:
                    offset_roll[end_frame, note_index] = 1
                    # difference in seconds
                    reg_offset_roll[end_frame, note_index] = \
                        (offset_time - start_time) - (end_frame/FRAMES_PER_SECOND)

                if start_frame >= 0:
                    onset_roll[start_frame, note_index] = 1
                    # difference in seconds
                    reg_onset_roll[start_frame, note_index] = \
                        (onset_time - start_time) - (start_frame/FRAMES_PER_SECOND)
                else:
                    mask_roll[:end_frame + 1, note_index] = 0
    
    for n in range(NUM_NOTES):
        reg_onset_roll[:, n] = get_regression(reg_onset_roll[:, n])
        reg_offset_roll[:, n] = get_regression(reg_offset_roll[:, n])

    roll_info = {
        'frame_roll': frame_roll,
        'onset_roll': onset_roll,
        'offset_roll': offset_roll,
        'reg_onset_roll': reg_onset_roll,
        'reg_offset_roll': reg_offset_roll,
        'velocity_roll': velocity_roll,
        'mask_roll': mask_roll,
    }

    return roll_info

class MusicNetDatasetProcessed:
    def __init__(self, raw_dataset, augmentor=None):
        self.raw_dataset = raw_dataset
        self.augmentor = augmentor
        self.rng = np.random.default_rng()
    
    def get_processed_data(self, rec_id, start_sample):
        orig_sr = self.raw_dataset.sample_rate
        end_sample = start_sample + SEGMENT_LENGTH*orig_sr
        x, note_infos = self.raw_dataset.get_record_data(
            rec_id,
            start_sample=start_sample,
            end_sample=end_sample)
        
        if orig_sr != SAMPLE_RATE:
            x = librosa.resample(x, orig_sr, SAMPLE_RATE)
        assert len(x) == SEGMENT_SAMPLES

        if self.augmentor:
            x = self.augmentor.augment(x)

        segment_start_time = start_sample/orig_sr
        segment_end_time = segment_start_time + SEGMENT_LENGTH
        note_events = []
        for note_info in note_infos:
            note_events.append(create_note_event(
                note_info['note'],
                note_info['start_sample']/orig_sr,
                note_info['end_sample']/orig_sr)
            )

        roll_info = generate_rolls(note_events, segment_start_time, segment_end_time)

        return x, note_events, roll_info

    def __getitem__(self, index):
        rec_id, start_sample = index
        x, _, roll_info = self.get_processed_data(rec_id, start_sample)

        return x, roll_info

class MaestroDatasetProcessed:
    def __init__(self, raw_dataset, augmentor=None):
        self.raw_dataset = raw_dataset
        self.augmentor = augmentor
        self.rng = np.random.default_rng()
    
    def get_processed_data(self, rec_id, start_sample):
        orig_sr = self.raw_dataset.sample_rate
        end_sample = start_sample + SEGMENT_LENGTH*orig_sr
        x, note_events = self.raw_dataset.get_record_data(
            rec_id,
            start_sample=start_sample,
            end_sample=end_sample)
        
        if orig_sr != SAMPLE_RATE:
            x = librosa.resample(x, orig_sr, SAMPLE_RATE)
        assert len(x) == SEGMENT_SAMPLES

        if self.augmentor:
            x = self.augmentor.augment(x)

        segment_start_time = start_sample/orig_sr
        segment_end_time = segment_start_time + SEGMENT_LENGTH

        if orig_sr != SAMPLE_RATE:
            new_note_events = []
            for event in note_events:
                event = event.copy()
                event['onset_time'] *= SAMPLE_RATE/orig_sr
                event['offset_time'] *= SAMPLE_RATE/orig_sr
                new_note_events.append(event)
            note_events = new_note_events

        roll_info = generate_rolls(note_events, segment_start_time, segment_end_time)

        return x, note_events, roll_info

    def __getitem__(self, index):
        rec_id, start_sample = index
        x, _, roll_info = self.get_processed_data(rec_id, start_sample)

        return x, roll_info

class MusicSegmentSampler:
    def __init__(self, proc_dataset, batch_size, num_batches=None, shuffle=True, random_start_times=False):
        self.proc_dataset = proc_dataset
        self.raw_dataset = proc_dataset.raw_dataset
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.shuffle = shuffle
        self.random_start_times = random_start_times
        self.rng = np.random.default_rng()

        self.generate_segments()
        if self.num_batches is None:
            self.num_batches = (len(self.segment_infos) + self.batch_size - 1) // self.batch_size
        
    def generate_segments(self):
        self.segment_infos = []
        rec_ids = self.raw_dataset.rec_ids
        for rec_id in rec_ids:
            if self.random_start_times:
                start_time = np.random.uniform(0, SEGMENT_LENGTH)
            else:
                start_time = 0
            rec_length = self.raw_dataset.get_record_num_samples(rec_id)/self.raw_dataset.sample_rate
            while (start_time + SEGMENT_LENGTH < (rec_length - 0.01)):
                start_sample = int(start_time*self.raw_dataset.sample_rate)
                self.segment_infos.append((rec_id, start_sample))
                start_time += SEGMENT_LENGTH
        if self.shuffle:
            self.rng.shuffle(self.segment_infos)
    
    def __iter__(self):
        segment_index = 0
        for _ in range(self.num_batches):
            batch_indices = self.segment_infos[segment_index:segment_index+self.batch_size]
            yield batch_indices

            segment_index = segment_index + self.batch_size
            if segment_index >= len(self.segment_infos):
                segment_index = 0
                if self.random_start_times:
                    self.generate_segments()
                else:
                    self.rng.shuffle(self.segment_infos)

    def __len__(self):
        return self.num_batches

def music_data_collate_fn(unbatched_data):
    batched_inputs = torch.tensor(np.stack([n[0] for n in unbatched_data]), dtype=torch.float)
    batched_targets = {}
    for key in unbatched_data[0][1].keys():
        batched_targets[key] = torch.tensor(np.stack([n[1][key] for n in unbatched_data]), dtype=torch.float)
    return batched_inputs, batched_targets

def get_music_dataloader(proc_dataset, sampler, num_workers=None, pin_memory=False):
    if num_workers is not None:
        dataloader = torch_data.DataLoader(
            dataset=proc_dataset,
            batch_sampler=sampler,
            collate_fn=music_data_collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory)
    else:
        dataloader = torch_data.DataLoader(
            dataset=proc_dataset,
            batch_sampler=sampler,
            collate_fn=music_data_collate_fn,
            pin_memory=pin_memory)
    
    return dataloader
