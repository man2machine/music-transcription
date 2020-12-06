# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 22:40:18 2020

@author: skarnik
"""

import numpy as np
from sklearn import metrics
import torch

def move_data_to_device(x, device):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        return x

    return x.to(device)


def append_to_dict(dict, key, value):
    
    if key in dict.keys():
        dict[key].append(value)
    else:
        dict[key] = [value]
        
 
def forward_dataloader(model, dataloader, batch_size, return_target=True):
    """Forward data generated from dataloader to model.
    Args:
      model: object
      dataloader: object, used to generate mini-batches for evaluation.
      batch_size: int
      return_target: bool
    Returns:
      output_dict: dict, e.g. {
        'frame_output': (segments_num, frames_num, classes_num),
        'onset_output': (segments_num, frames_num, classes_num),
        'frame_roll': (segments_num, frames_num, classes_num),
        'onset_roll': (segments_num, frames_num, classes_num),
        ...}
    """

    output_dict = {}
    device = next(model.parameters()).device

    for n, batch_data_dict in enumerate(dataloader):
        
        batch_waveform = move_data_to_device(batch_data_dict['waveform'], device)

        with torch.no_grad():
            model.eval()
            batch_output_dict = model(batch_waveform)

        for key in batch_output_dict.keys():
            if '_list' not in key:
                append_to_dict(output_dict, key, 
                    batch_output_dict[key].data.cpu().numpy())

        if return_target:
            for target_type in batch_data_dict.keys():
                if 'roll' in target_type or 'reg_distance' in target_type or \
                    'reg_tail' in target_type:
                    append_to_dict(output_dict, target_type, 
                        batch_data_dict[target_type])

    for key in output_dict.keys():
        output_dict[key] = np.concatenate(output_dict[key], axis=0)
    
    return output_dict


def mae(target, output, mask):
    """
    Mask just indicates if we want to evaluate at only the places where an onset, offset, etc exists

    """
    if mask is None:
        return np.mean(np.abs(target - output))
    else:
        target *= mask
        output *= mask
        return np.sum(np.abs(target - output)) / np.clip(np.sum(mask), 1e-8, np.inf)
    

class SegmentEvaluator(object):
    def __init__(self, model, batch_size):
        """Evaluate segment-wise metrics.
        Args:
          model: object
          batch_size: int
        """
        self.model = model
        self.batch_size = batch_size

    def evaluate(self, dataloader):
        """Evaluate over a few mini-batches.
        Args:
          output_dict
        Returns:
          statistics: dict, e.g. {
            'frame_f1': 0.800, 
            (if exist) 'onset_f1': 0.500, 
            (if exist) 'offset_f1': 0.300, 
            ...}
          
        """

        statistics = {}
        
        output_dict = forward_dataloader(self.model, dataloader, self.batch_size)
        
        # Frame and onset evaluation
        if 'frame_output' in output_dict.keys():
            statistics['frame_avg_precision'] = metrics.average_precision_score(
                output_dict['frame_roll'].flatten(), 
                output_dict['frame_output'].flatten(), average='macro')
        
        if 'onset_output' in output_dict.keys():
            statistics['onset_macro_avg_precision'] = metrics.average_precision_score(
                output_dict['onset_roll'].flatten(), 
                output_dict['onset_output'].flatten(), average='macro')

        if 'offset_output' in output_dict.keys():
            statistics['offset_avg_precision'] = metrics.average_precision_score(
                output_dict['offset_roll'].flatten(), 
                output_dict['offset_output'].flatten(), average='macro')

        if 'reg_onset_output' in output_dict.keys():
            """Mask indictes only evaluate where either prediction or ground truth exists"""
            """Each element in reg_onset_output and reg_onset_roll is binary"""
            mask = (np.sign(output_dict['reg_onset_output'] + output_dict['reg_onset_roll'] - 0.01) + 1) / 2
            statistics['reg_onset_mae'] = mae(output_dict['reg_onset_output'], 
                output_dict['reg_onset_roll'], mask)

        if 'reg_offset_output' in output_dict.keys():
            """Mask indictes only evaluate where either prediction or ground truth exists"""
            """Each element in reg_onset_output and reg_onset_roll is binary"""
            mask = (np.sign(output_dict['reg_offset_output'] + output_dict['reg_offset_roll'] - 0.01) + 1) / 2
            statistics['reg_offset_mae'] = mae(output_dict['reg_offset_output'], 
                output_dict['reg_offset_roll'], mask)


        for key in statistics.keys():
            statistics[key] = np.around(statistics[key], decimals=4)

        return statistics