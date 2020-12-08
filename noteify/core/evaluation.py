# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 14:11:29 2020

@author: Shahir
"""

import math
import collections

import numpy as np
from sklearn import metrics
import torch
from tqdm import tqdm

from noteify.core.config import (NUM_NOTES, SEGMENT_SAMPLES,
                                 FRAMES_PER_SECOND, MIN_MIDI)
from noteify.core.datasets import create_note_event

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
                          apply_sigmoid=False):
    
    outputs = collections.defaultdict(list)

    index = 0
    while index <= len(inputs):
        batch_inputs = inputs[index:index+batch_size].to(device)

        with torch.set_grad_enabled(False):
            model.eval()
            batch_outputs = model(batch_inputs, apply_sigmoid=True)
        
        for key, value in batch_outputs.items():
            outputs[key].append(value.detach().cpu().numpy())
    
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

def get_evaluation_stats(model, dataloader, device):
    stats = {}
    
    result = get_model_outputs(model,
                               dataloader,
                               device,
                               return_targets=True,
                               apply_sigmoid=True,
                               max_iter=10)
    targets = result['targets']
    outputs = result['outputs']

    # Frame, onset and offset evaluation
    if 'frame_output' in outputs.keys():
        stats['frame_avg_precision'] = metrics.average_precision_score(
            targets['frame_roll'].flatten(), 
            outputs['frame_output'].flatten(),
            average='macro')
    
    if 'onset_output' in outputs.keys():
        stats['onset_macro_avg_precision'] = metrics.average_precision_score(
            targets['onset_roll'].flatten(), 
            outputs['onset_output'].flatten(),
            average='macro')

    if 'offset_output' in outputs.keys():
        stats['offset_avg_precision'] = metrics.average_precision_score(
            targets['offset_roll'].flatten(), 
            outputs['offset_output'].flatten(),
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
    def __init__(self, model, device):
        self.model = model
        self.device = device

        self.onset_threshold = 0.3
        self.offset_threshold = 0.3
        self.frame_threshold = 0.1
    
    def generate_segments(self, x):
        assert len(x) % SEGMENT_SAMPLES == 0
        segments = []
        
        start_index = 0
        while True:
            end_index = start_index + SEGMENT_SAMPLES
            if end_index >= len(x):
                break
            segments.append(x[start_index:end_index])
            start_index += SEGMENT_SAMPLES // 2
        
        segments = np.stack(segments, axis=0)
        return segments

    def combine_segment_predictions(self, segment_outputs):
        if len(segment_outputs) == 1:
            return segment_outputs[0]
        
        # segment_outputs is (num_segments, num_frames, num_notes)
        x = segment_outputs[:, :-1] # remove last frame
        num_segments, num_frames, num_notes = x.shape
        
        segment_predictions = []
        start_frame = int(num_frames * 0.25)
        end_frame = num_frames - start_frame
        segment_predictions.append(segment_outputs[0, :end_frame])
        for n in range(1, num_frames - 1):
            segment_predictions.append(segment_outputs[n, start_frame:end_frame])
        segment_predictions.append(segment_outputs[-1, start_frame:])

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
                onset_shift = onset_shift_output[curr_start_index]
                onset_frame = curr_start_index + onset_shift

                if frame_output[i] <= self.frame_threshold:
                    # frame_output "disappeared"
                    frame_disappear_index = i
                
                if offset_output_bin[i] and curr_end_index is None:
                    # offset detected
                    curr_end_index = i
                
                if frame_disappear_index is not None:
                    # both an offset and a frame disappearance is required to end a note
                    if curr_end_index is not None and (curr_end_index - curr_start_index) > (frame_disappear_index - curr_end_index):
                        offset_frame = curr_end_index
                    else:
                        offset_frame = frame_disappear_index
                    offset_shift = offset_shift_output[offset_frame]
                    offset_frame += offset_shift

                    note_frame_intervals.append((onset_frame, offset_frame))
                    curr_start_index = None
                    frame_disappear_index = None
                    curr_end_index = None
            
            if curr_start_index is not None or i == num_frames - 1:
                # offset not detected
                offset_shift = offset_shift_output[i]
                offset_frame = i + offset_shift
                
                note_frame_intervals.append((onset_frame, offset_frame))
                curr_start_index = None
                frame_disappear_index = None
                curr_start_index = None
        
        note_frame_intervals.sort(key=lambda n: n[0])

        return note_frame_intervals

    def transcribe(self, x):
        num_segments = math.ceil(len(x)/SEGMENT_SAMPLES)
        pad_length = num_segments*SEGMENT_SAMPLES - len(x)
        x = np.concatenate((x, np.zeros(pad_length)), axis=0)

        segments = torch.tensor(self.generate_segments(x))
        outputs = get_model_predictions(self.model, segments, self.device,
                                        batch_size=32, apply_sigmoid=True)
        for key, value in outputs.items():
            outputs[key] = self.combine_segment_predictions(value)
        
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
                onset_output_bin[note_index],
                onset_shift_output[note_index],
                offset_output_bin[note_index],
                offset_shift_output[note_index],
                frame_output[note_index])
            
            midi_note = note_index + MIN_MIDI
            for start_frame, end_frame in note_intervals:
                start_time = start_frame / FRAMES_PER_SECOND
                end_time = end_frame / FRAMES_PER_SECOND

                note_event = create_note_event(
                    midi_note,
                    start_time,
                    end_time)
                note_events.append(note_event)
        
        return note_events

