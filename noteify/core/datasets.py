# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 14:12:40 2020

@author: Shahir
"""

import os
import mmap
import pickle
import errno
import csv
import subprocess
import tarfile
import urllib.request

import numpy as np
import torch
import torch.utils.data as torch_data
import librosa
import sox
from scipy.io import wavfile
from intervaltree import IntervalTree

from noteify.core.config import (
    SAMPLE_RATE, HOP_LENGTH, NUM_NOTES, MIN_MIDI,
    SEGMENT_LENGTH, SEGMENT_SAMPLES, SEGMENT_FRAMES,
    FRAMES_PER_SECOND)

SIZE_FLOAT = 4 # size of a float

class MusicNetDataset:
    URL = "https://homes.cs.washington.edu/~thickstn/media/musicnet.tar.gz"
    RAW_FOLDER = "raw"
    TRAIN_DATA_DIR = "train_data"
    TRAIN_LABELS_DIR = "train_labels"
    TRAIN_TREE_FNAME = "train_tree.pckl"
    TEST_DATA_DIR = "test_data"
    TEST_LABELS_DIR = "test_labels"
    TEST_TREE_FNAME = "test_tree.pckl"
    EXTRACTED_FOLDERS = [TRAIN_DATA_DIR, TRAIN_LABELS_DIR, TEST_DATA_DIR, TEST_LABELS_DIR]
    SAMPLE_RATE = 44100

    def __init__(self,
                 data_dir,
                 download=False,
                 refresh_cache=False,
                 delete_wav=True,
                 train=True,
                 mmap=False):

        self.root_dir = os.path.abspath(os.path.expanduser(data_dir))
        self.refresh_cache = refresh_cache
        self.delete_wav = delete_wav
        self.train = train
        self.mmap = mmap

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

        self.rec_ids = list(self.labels.keys())
        self.records = dict()
        self.open_files = []

        for record in os.listdir(self.data_path):
            if not record.endswith('.bin'):
                continue
            if self.mmap:
                fd = os.open(os.path.join(self.data_path, record), os.O_RDONLY)
                buf = mmap.mmap(fd, 0, mmap.MAP_SHARED, mmap.PROT_READ)
                self.records[int(record[:-4])] = (buf, len(buf)//SIZE_FLOAT)
                self.open_files.append(fd)
            else:
                f = open(os.path.join(self.data_path, record))
                self.records[int(record[:-4])] = (os.path.join(self.data_path, record),
                                                  os.fstat(f.fileno()).st_size//SIZE_FLOAT)
                f.close()

    def close(self):
        if self.mmap:
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

        filename = self.URL.rpartition('/')[2]
        file_path = os.path.join(self.root_dir, self.RAW_FOLDER, filename)
        if not os.path.exists(file_path):
            print("Downloading", self.URL)
            data = urllib.request.urlopen(self.URL)
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
        for item in os.listdir(sub_dir):
            if not item.endswith('.wav'):
                continue
            sr, data = wavfile.read(os.path.join(self.root_dir, path, item))
            assert sr == self.SAMPLE_RATE
            data.tofile(os.path.join(self.root_dir, path, item[:-4]+'.bin'))
            if self.delete_wav:
                os.remove(os.path.join(sub_dir, item))
    
    def process_labels(self, path):
        trees = dict()
        for item in os.listdir(os.path.join(self.root_dir, path)):
            if not item.endswith('.csv'):
                continue
            rec_id = int(item[:-4])
            tree = IntervalTree()
            with open(os.path.join(self.root_dir, path, item), 'r') as f:
                reader = csv.DictReader(f, delimiter=',')
                for label in reader:
                    start_time = int(label['start_time'])
                    end_time = int(label['end_time'])
                    instrument = int(label['instrument'])
                    note = int(label['note'])
                    start_beat = float(label['start_beat'])
                    end_beat = float(label['end_beat'])
                    note_value = label['note_value']
                    tree[start_time:end_time] = {
                        'start_time': start_time,
                        'end_time': end_time,
                        'instrument': instrument,
                        'note': note,
                        'start_beat': start_beat,
                        'end_beat': end_beat,
                        'note_value': note_value,
                    }
            trees[rec_id] = tree
        return trees

    def get_record_length(self, rec_id):
        """
        Outputs (sample rate, num samples)
        """

        return self.records[rec_id][1]

    def get_record_data(self, rec_id, start_sample=None, end_sample=None):
        if start_sample is None:
            start_sample = 0
        if end_sample is None:
            end_sample = self.records[rec_id][1]*SIZE_FLOAT

        start_pos = int(start_sample)*SIZE_FLOAT
        end_pos = int(end_sample)*SIZE_FLOAT
        if self.mmap:
            x = np.frombuffer(
                self.records[rec_id][0][start_pos:end_pos], dtype=np.float32).copy()
        else:
            with open(self.records[rec_id][0], 'rb') as f:
                x = np.fromfile(f, dtype=np.float32,
                                count=(end_sample - start_sample))

        intervals = self.labels[rec_id][start_sample:end_sample + 1]
        note_infos = [n.data for n in intervals]

        return x, note_infos

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

        if self.noise:
            aug_x += self.rng.normal(0, self.rng.uniform(0, 0.01), len(aug_x))

        return aug_x

    def loguniform(self, low, high, size):
        return np.exp(self.rng.uniform(np.log(low), np.log(high), size))

def create_note_event(midi_note, onset_time, offset_time, velocity=127):
    return {'midi_note': midi_note,
            'onset_time': onset_time,
            'offset_time': offset_time,
            'velocity': velocity}

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

def generate_rolls(note_events, start_time):
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
            start_frame = int(
                round((onset_time - start_time)*FRAMES_PER_SECOND))
            end_frame = int(
                round((offset_time - start_time)*FRAMES_PER_SECOND))

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

    def __getitem__(self, index):
        rec_id, start_sample = index
        orig_sr = self.raw_dataset.SAMPLE_RATE
        end_sample = start_sample + SEGMENT_LENGTH*orig_sr
        x, note_infos = self.raw_dataset.get_record_data(
            rec_id,
            start_sample=start_sample,
            end_sample=end_sample)

        x = librosa.resample(x, orig_sr, SAMPLE_RATE)
        assert len(x) == SEGMENT_SAMPLES

        if self.augmentor:
            x = self.augmentor.augment(x)

        segment_start_time = start_sample/orig_sr
        note_events = []
        for note_info in note_infos:
            note_events.append(create_note_event(
                note_info['note'],
                note_info['start_time']/orig_sr,
                note_info['end_time']/orig_sr
            ))

        roll_info = generate_rolls(note_events, segment_start_time)

        return x, roll_info

class MusicNetSampler:
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
            rec_length = self.raw_dataset.get_record_length(rec_id)/self.raw_dataset.SAMPLE_RATE
            while (start_time + SEGMENT_LENGTH < rec_length):
                self.segment_infos.append((rec_id, start_time))
                start_time += SEGMENT_LENGTH
        if self.shuffle:
            self.rng.shuffle(self.segment_infos)
    
    def __iter__(self):
        segment_index = 0
        for _ in range(num_batches):
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
    batched_inputs = torch.tensor(np.stack([n[0] for n in unbatched_data]))
    batched_targets = {}
    for key in unbatched_data[0][1].keys():
        batched_targets[key] = torch.tensor(np.stack([n[1][key] for n in unbatched_data]))
    return batched_inputs, batched_targets

def get_musicnet_dataloader(proc_dataset, sampler, num_workers=4):
    if num_workers is not None:
        dataloader = torch_data.DataLoader(
            dataset=proc_dataset,
            batch_sampler=sampler,
            collate_fn=music_data_collate_fn,
            num_workers=num_workers,
            pin_memory=False)
    else:
        dataloader = torch_data.DataLoader(
            dataset=proc_dataset,
            batch_sampler=sampler,
            collate_fn=music_data_collate_fn,
            pin_memory=False)
    
    return dataloader

def get_musicnet_data(data_dir, batch_size):
    raw_dataset_train = MusicNetDataset(data_dir, download=True, train=True)
    raw_dataset_test = MusicNetDataset(data_dir, download=True, train=False)

    dataset_train = MusicNetDatasetProcessed(raw_dataset_train, augmentor=MusicAugmentor())
    sampler_train = MusicNetSampler(dataset_train, batch_size)
    dataloader_train = get_musicnet_dataloader(dataset_train, sampler_train)

    dataset_test = MusicNetDatasetProcessed(raw_dataset_test)
    sampler_test = MusicNetSampler(dataset_test, batch_size)
    dataloader_test = get_musicnet_dataloader(dataset_test, sampler_test)

    datasets = {'train': dataset_train, 'test': dataset_test}
    dataloaders = {'train': dataloader_train, 'test': dataloader_test}

    return datasets, dataloaders
