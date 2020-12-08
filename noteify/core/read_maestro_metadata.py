# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 21:21:14 2020

@author: skarnik
"""

import json

with open('/Users/skarn/OneDrive/Documents/MIT/year_3/21M.080/music-transcription/noteify/maestro-v3.0.0.json') as f:
    data = json.load(f)

class MaestroDataset:
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

        self.reformatted_metadata = {}
        for category in data.keys():
            category_data = data[category]
            
            for element_id in category_data.keys():
                if element_id not in self.reformatted_metadata.keys():
                    self.reformatted_metadata[element_id] = {}
                self.reformatted_metadata[element_id][category] = data[category][element_id]
    
        self.rec_ids = self.reformatted_metadata.keys()
        self.records = dict()
        self.open_files = []

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
        

    def get_record_length(self, rec_id):
        """
        Return float duration in seconds

        """
        return self.reformatted_metadata[rec_id]['duration']
    
    def get_record_train_data_files(self):
        """
        Return train data file names

        """
        rec_ids = self.reformatted_metadata.keys()
        
        return [self.reformatted_metadata[rec_id]['audio_filename'] for rec_id in rec_ids if self.reformatted_metadata[rec_id]['split'] == 'train']

    def get_record_test_data_files(self):
        """
        Return train data file names

        """
        rec_ids = self.reformatted_metadata.keys()
        
        return [self.reformatted_metadata[rec_id]['audio_filename'] for rec_id in rec_ids if self.reformatted_metadata[rec_id]['split'] == 'test']

    def get_record_validation_data_files(self):
        """
        Return train data file names

        """
        rec_ids = self.reformatted_metadata.keys()
        
        return [self.reformatted_metadata[rec_id]['audio_filename'] for rec_id in rec_ids if self.reformatted_metadata[rec_id]['split'] == 'validation']
