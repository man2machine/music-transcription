# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 14:08:52 2020

@author: Shahir
"""

import torch
import torch.nn.functional as F

def masked_bce_with_logits(logits, targets, mask):
    loss = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        weight=mask,
        reduction='sum')
    loss = loss/torch.sum(mask)
    return loss

def compute_transcription_losses(outputs, targets):
    frame_loss = masked_bce_with_logits(
        outputs['frame_output'],
        targets['frame_roll'],
        targets['mask_roll'])
    onset_loss = masked_bce_with_logits(
        outputs['reg_onset_output'],
        targets['reg_onset_roll'],
        targets['mask_roll'])
    offset_loss = masked_bce_with_logits(
        outputs['reg_offset_output'],
        targets['reg_offset_roll'],
        targets['mask_roll'])
    
    return [frame_loss, onset_loss, offset_loss]

