# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 14:11:29 2020

@author: Shahir
"""

import collections

import numpy as np
from sklearn import metrics
import torch
 
def get_model_outputs(model, dataloader, device, return_inputs=False, return_targets=False):
    inputs = []
    targets = collections.defaultdict(list)
    outputs = collections.defaultdict(list)

    for batch_inputs, batch_targets in dataloader:
        if return_inputs:
            inputs.append(batch_inputs)
        batch_inputs = batch_inputs.to(device)

        if return_targets:
            for key, value in batch_targets.keys():
                targets[key].append(value.detach().cpu().numpy())
        
        with torch.set_grad_enabled(False):
            model.eval()
            batch_output = model(batch_inputs)

        for key, value in batch_output.keys():
            outputs[key].append(value.detach().cpu().numpy())
    
    if return_inputs:
        inputs = np.concatenate(inputs, axis=0)
    if return_targets:
        for key in targets:
            targets[key] = np.concatenate(targets[key], axis=0)
    for key in outputs:
        outputs[key] = np.concatenate(outputs[key], axis=0)
    
    result = {
        'inputs': inputs,
        'targets': targets,
        'outputs': outputs
    }
    
    return result

def masked_average_error(target, output, mask):
    """
    Calculate average error between target and output, only at locations where mask is nonzero
    Inputs are numpy arrays all of same shape
    """
    if mask is None:
        return np.mean(np.abs(target - output))
    else:
        target *= mask
        output *= mask
        return np.sum(np.abs(target - output)) / np.clip(np.sum(mask), 1e-8, np.inf)

def get_evaluation_stats(model, dataloader, device):
    stats = {}
    
    result = get_model_outputs(model, dataloader, device, return_targets=True)
    targets = result['targets']
    outputs = result['outputs']

    # Frame and onset evaluation
    if 'frame_output' in outputs.keys():
        stats['frame_avg_precision'] = metrics.average_precision_score(
            targets['frame_roll'].flatten(), 
            outputs['frame_output'].flatten(), average='macro')
    
    if 'onset_output' in outputs.keys():
        stats['onset_macro_avg_precision'] = metrics.average_precision_score(
            targets['onset_roll'].flatten(), 
            outputs['onset_output'].flatten(), average='macro')

    if 'offset_output' in outputs.keys():
        stats['offset_avg_precision'] = metrics.average_precision_score(
            targets['offset_roll'].flatten(), 
            outputs['offset_output'].flatten(), average='macro')
    
    # we use masked error calculation in order to only evaluate locations
    # where either the prediction or ground truth actually exists

    if 'reg_onset_output' in outputs.keys():
        mask = (np.sign(outputs['reg_onset_output'] + outputs['reg_onset_roll'] - 0.01) + 1) / 2
        stats['reg_onset_mae'] = masked_average_error(outputs['reg_onset_output'], 
            targets['reg_onset_roll'], mask)

    if 'reg_offset_output' in outputs.keys():
        mask = (np.sign(outputs['reg_offset_output'] + outputs['reg_offset_roll'] - 0.01) + 1) / 2
        stats['reg_offset_mae'] = masked_average_error(outputs['reg_offset_output'], 
            targets['reg_offset_roll'], mask)
    
    return stats
