# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 14:11:29 2020

@author: Shahir
"""

import math
import collections

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from skimage.transform import resize as sk_resize
import torch
from tqdm import tqdm

from noteify.core.config import (NUM_NOTES, SEGMENT_SAMPLES, HOP_LENGTH,
                                 FRAMES_PER_SECOND, MIN_MIDI, BINS_PER_NOTE)
from noteify.core.datasets import create_note_event
from noteify.core.models import SpectrogramLayer

def targets_to_device(targets, device):
    for key, value in targets.items():
        targets[key] = value.to(device)

def get_model_outputs(model,
                      dataloader,
                      device,
                      pbar=False,
                      max_iter=None,
                      return_inputs=False,
                      return_targets=False,
                      apply_sigmoid=False):
    
    inputs = []
    targets = collections.defaultdict(list)
    outputs = collections.defaultdict(list)
    
    n = 0
    if pbar:
        dataloader = tqdm(dataloader)
    for batch_inputs, batch_targets in dataloader:
        if return_inputs:
            inputs.append(batch_inputs)
        batch_inputs = batch_inputs.to(device)

        if return_targets:
            for key, value in batch_targets.items():
                targets[key].append(value.detach().cpu())
        
        with torch.set_grad_enabled(False):
            model.eval()
            batch_outputs = model(batch_inputs, apply_sigmoid=apply_sigmoid)

        for key, value in batch_outputs.items():
            outputs[key].append(value.detach().cpu())
        
        n += 1
        if max_iter is not None and n >= max_iter:
            break
    if pbar:
        dataloader.close()
    
    targets = dict(targets)
    outputs = dict(outputs)
    
    if return_inputs:
        inputs = torch.cat(inputs, dim=0)
    if return_targets:
        for key in targets:
            targets[key] = torch.cat(targets[key], dim=0)
    for key in outputs:
        outputs[key] = torch.cat(outputs[key], dim=0)
    
    result = {
        'inputs': inputs,
        'targets': targets,
        'outputs': outputs
    }
    
    return result

def get_model_predictions(model,
                          inputs,
                          device,
                          batch_size,
                          apply_sigmoid=False,
                          pbar=False):
    
    outputs = collections.defaultdict(list)

    if pbar:
        bar = tqdm(total=math.ceil(len(inputs)/batch_size))
    index = 0
    while index < len(inputs):
        batch_inputs = inputs[index:index+batch_size].to(device)
        index += batch_size

        with torch.set_grad_enabled(False):
            model.eval()
            batch_outputs = model(batch_inputs, apply_sigmoid=apply_sigmoid)
        
        for key, value in batch_outputs.items():
            outputs[key].append(value.detach().cpu().numpy())
        
        if pbar:
            bar.update(1)
    if pbar:
        bar.close()
    
    outputs = dict(outputs)
    for key in outputs:
        outputs[key] = np.concatenate(outputs[key], axis=0)

    return outputs

def masked_average_error(target, output, mask):
    """
    Calculate average error between target and output, only at locations where mask is nonzero
    Inputs are numpy arrays all of same shape
    """
    if mask is None:
        return torch.mean(torch.abs(target - output))
    else:
        target *= mask
        output *= mask
        return torch.sum(torch.abs(target - output)) / torch.clamp(torch.sum(mask), min=1e-8)

def get_evaluation_stats(model, dataloader, device, max_iter=10, pbar=True):
    stats = {}
    
    result = get_model_outputs(model,
                               dataloader,
                               device,
                               return_targets=True,
                               apply_sigmoid=True,
                               max_iter=max_iter,
                               pbar=pbar)
    targets = result['targets']
    outputs = result['outputs']

    # Frame, onset and offset evaluation
    if 'frame_output' in outputs.keys():
        stats['frame_avg_precision'] = metrics.average_precision_score(
            targets['frame_roll'].numpy().flatten(), 
            outputs['frame_output'].numpy().flatten(),
            average='macro')
    
    if 'onset_output' in outputs.keys():
        stats['onset_macro_avg_precision'] = metrics.average_precision_score(
            targets['onset_roll'].numpy().flatten(), 
            outputs['onset_output'].numpy().flatten(),
            average='macro')

    if 'offset_output' in outputs.keys():
        stats['offset_avg_precision'] = metrics.average_precision_score(
            targets['offset_roll'].numpy().flatten(), 
            outputs['offset_output'].numpy().flatten(),
            average='macro')
    
    with torch.set_grad_enabled(False):
        # We use masked error calculation in order to only evaluate locations
        # where either the prediction or ground truth actually exists
        if 'reg_onset_output' in outputs.keys():
            mask = (torch.sign(outputs['reg_onset_output'] + targets['reg_onset_roll'] - 0.01) + 1) / 2
            stats['reg_onset_mae'] = masked_average_error(
                outputs['reg_onset_output'].to(device), 
                targets['reg_onset_roll'].to(device),
                mask.to(device)).detach().item()
        
        if 'reg_offset_output' in outputs.keys():
            mask = (torch.sign(outputs['reg_offset_output'] + targets['reg_offset_roll'] - 0.01) + 1) / 2
            stats['reg_offset_mae'] = masked_average_error(
                outputs['reg_offset_output'].to(device),
                targets['reg_offset_roll'].to(device),
                mask.to(device)).detach().item()
    
    return stats

class TranscriptionProcessor:
    def __init__(self, model, device, batch_size):
        self.model = model
        self.device = device
        self.batch_size = batch_size

        self.onset_threshold = 0.3
        self.offset_threshold = 0.3
        self.frame_threshold = 0.1
    
    def generate_segments(self, x):
        assert len(x) % SEGMENT_SAMPLES == 0
        segments = []
        
        start_index = 0
        while True:
            end_index = start_index + SEGMENT_SAMPLES
            if end_index > len(x):
                break
            segments.append(x[start_index:end_index])
            start_index += SEGMENT_SAMPLES // 2
        
        segments = np.stack(segments, axis=0)
        return segments

    def combine_segment_predictions(self, segment_outputs, num_samples):
        if len(segment_outputs) == 1:
            return segment_outputs[0]
        
        # segment_outputs is (num_segments, num_frames, num_notes)
        zs = segment_outputs[:, :-1] # remove last frame
        num_segments, num_frames, num_notes = segment_outputs.shape

        segment_samples_diff = SEGMENT_SAMPLES//2

        segment_predictions = []
        current_frame_start_sample = 0
        last_end_cut_sample = 0
        expected_start_cut_sample = 0
        expected_end_cut_sample = SEGMENT_SAMPLES*0.75

        for n in range(0, num_segments):
            if n == (num_segments - 1):
                expected_end_cut_sample = num_samples
            
            start_cut_frame = int((expected_start_cut_sample - current_frame_start_sample)//HOP_LENGTH)
            end_cut_frame = int((expected_end_cut_sample - current_frame_start_sample)//HOP_LENGTH)
            section = zs[n, start_cut_frame:end_cut_frame]
            segment_predictions.append(section)
            
            # sample the segment we just added after cutting actually ends on
            last_end_cut_sample += (end_cut_frame - start_cut_frame) * HOP_LENGTH

            # what sample the next frame starts on
            current_frame_start_sample += segment_samples_diff
            # what frame we want the cut to start on
            expected_start_cut_sample = last_end_cut_sample
            # what frame we want the cut to end on
            expected_end_cut_sample += segment_samples_diff

        pred = np.concatenate(segment_predictions, axis=0)

        return pred

    def regression_to_binary_output(self, reg_output, threshold, neighbours):
        """
        See Section III-D in [1] for deduction.
        [1] Q. Kong, et al., High-resolution Piano Transcription 
        with Pedals by Regressing Onsets and Offsets Times, 2020.
        """

        binary_output = np.zeros_like(reg_output)
        shift_output = np.zeros_like(reg_output)

        num_frames, num_notes = reg_output.shape
        for k in range(num_notes):
            x = reg_output[:, k]
            for n in range(neighbours, num_frames - neighbours):
                if x[n] > threshold and self.is_monotonic_neighbour(x, n, neighbours):
                    binary_output[n, k] = 1

                    if x[n - 1] > x[n + 1]:
                        shift = (x[n + 1] - x[n - 1]) / (x[n] - x[n + 1]) / 2
                    else:
                        shift = (x[n + 1] - x[n - 1]) / (x[n] - x[n - 1]) / 2
                    shift_output[n, k] = shift
        
        return binary_output, shift_output

    def is_monotonic_neighbour(self, x, n, neighbours):
        monotonic = True
        for i in range(neighbours):
            if x[n - i] < x[n - i - 1]:
                monotonic = False
            if x[n + i] < x[n + i + 1]:
                monotonic = False

        return monotonic
    
    def compute_note_intervals(self,
        onset_output_bin,
        onset_shift_output,
        offset_output_bin,
        offset_shift_output,
        frame_output):

        note_frame_intervals = []
        num_frames = len(onset_output_bin)
        curr_start_index = None
        frame_disappear_index = None
        curr_end_index = None
        
        for i in range(num_frames):
            if onset_output_bin[i]:
                # onset detected
                if curr_start_index is not None:
                    offset_frame = max(i - 1, 0)
                    onset_shift = onset_shift_output[curr_start_index]
                    onset_frame = curr_start_index + onset_shift
                    
                    note_frame_intervals.append((onset_frame, offset_frame))
                    curr_start_index = None # technically this is overwritten right after, it is kept for consistency
                    frame_disappear_index = None
                    curr_end_index = None
                
                curr_start_index = i

            if curr_start_index is not None and i > curr_start_index:
                if frame_output[i] <= self.frame_threshold and frame_disappear_index is not None:
                    # frame_output "disappeared"
                    frame_disappear_index = i
                
                if offset_output_bin[i] and curr_end_index is not None:
                    # offset detected
                    curr_end_index = i
                
                if frame_disappear_index is not None:
                    if curr_end_index is not None and (curr_end_index - curr_start_index) > (frame_disappear_index - curr_end_index):
                        # use offset detection index if it is valid
                        offset_frame = curr_end_index
                    else:
                        # use frame disappear index otherwise
                        offset_frame = frame_disappear_index
                    onset_shift = onset_shift_output[curr_start_index]
                    onset_frame = curr_start_index + onset_shift
                    offset_shift = offset_shift_output[offset_frame]
                    offset_frame += offset_shift

                    note_frame_intervals.append((onset_frame, offset_frame))
                    curr_start_index = None
                    frame_disappear_index = None
                    curr_end_index = None
            
                if (curr_start_index is not None) and (i - curr_start_index >= 600 or i == num_frames - 1):
                    # offset not detected
                    onset_shift = onset_shift_output[i]
                    onset_frame = curr_start_index + onset_shift
                    offset_shift = offset_shift_output[i]
                    offset_frame = i + offset_shift
                    
                    note_frame_intervals.append((onset_frame, offset_frame))
                    curr_start_index = None
                    frame_disappear_index = None
                    curr_start_index = None
        
        note_frame_intervals.sort(key=lambda n: n[0])

        return note_frame_intervals

    def recompute_velocities(self, x, note_events):
        spec_model = SpectrogramLayer()
        spec_model = spec_model.to(self.device)
        x = torch.tensor(x, dtype=torch.float).unsqueeze(0).to(self.device)

        with torch.set_grad_enabled(False):
            spec = spec_model(x).transpose(1, 2) # (batch_size, num_frames, freq_bins)
        spec = spec[0].cpu().numpy()

        lookahead = 10
        num_frames, freq_bins = spec.shape
        spec = sk_resize(spec, (num_frames, NUM_NOTES))
        spec = np.concatenate((spec, np.zeros((lookahead, NUM_NOTES))), axis=0)

        note_indices = []
        frame_intervals = []
        for note_event in note_events:
            onset_time = note_event['onset_time']
            offset_time = note_event['offset_time']
            midi_note = note_event['midi_note']

            onset_frame = int(round(onset_time * FRAMES_PER_SECOND))
            offset_frame = int(round(offset_time * FRAMES_PER_SECOND))

            start_bin = midi_note - MIN_MIDI
            note_indices.append(start_bin)
            frame_range = np.arange(onset_frame, onset_frame + lookahead)
            frame_intervals.append(frame_range)
        
        note_indices = np.array(note_indices)
        note_indices = note_indices.reshape(len(note_indices), 1)
        frame_intervals = np.array(frame_intervals)

        indexed_spec = spec[frame_intervals, note_indices]
        magnitudes = np.average(indexed_spec, axis=1)

        min_mag = np.min(magnitudes)
        max_mag = np.max(magnitudes)

        min_vel = 48
        max_vel = 120

        velocities = (magnitudes - min_mag)/(max_mag - min_mag)
        velocities = (velocities*(max_vel - min_vel) + min_vel).astype(np.int)
        velocities = np.clip(velocities, min_vel, max_vel)

        for n, note_event in enumerate(note_events):
            note_event['velocity'] = int(velocities[n])

    def transcribe(self, x, pbar=True):
        k = math.ceil(len(x)/SEGMENT_SAMPLES)
        pad_length = k*SEGMENT_SAMPLES - len(x)
        x = np.concatenate((x, np.zeros(pad_length)), axis=0)

        segments = torch.tensor(self.generate_segments(x), dtype=torch.float)
        outputs = get_model_predictions(self.model, segments, self.device,
                                        batch_size=self.batch_size, apply_sigmoid=True,
                                        pbar=pbar)
        for key, value in outputs.items():
            outputs[key] = self.combine_segment_predictions(value, len(x))
        
        frame_output = outputs['frame_output']

        onset_output_bin, onset_shift_output = self.regression_to_binary_output(
            reg_output=outputs['reg_onset_output'],
            threshold=self.onset_threshold,
            neighbours=2)
        offset_output_bin, offset_shift_output = self.regression_to_binary_output(
            reg_output=outputs['reg_offset_output'],
            threshold=self.offset_threshold,
            neighbours=4)
        
        note_events = []
        for note_index in range(NUM_NOTES):
            note_intervals = self.compute_note_intervals(
                onset_output_bin[:, note_index],
                onset_shift_output[:, note_index],
                offset_output_bin[:, note_index],
                offset_shift_output[:, note_index],
                frame_output[:, note_index])
            
            midi_note = note_index + MIN_MIDI
            for start_frame, end_frame in note_intervals:
                start_time = start_frame / FRAMES_PER_SECOND
                end_time = end_frame / FRAMES_PER_SECOND

                note_event = create_note_event(
                    midi_note,
                    start_time,
                    end_time)
                note_events.append(note_event)
        
        self.recompute_velocities(x, note_events)

        return note_events
