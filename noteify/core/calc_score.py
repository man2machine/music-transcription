# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 00:12:14 2020

@author: skarnik
"""

import numpy as np
import mir_eval
import librosa
from sklearn import metrics
from evaluation import get_model_outputs

FRAME_THRESHOLD = 0.3
OFFSET_RATIO = 0.2
OFFSET_MIN_TOLERANCE = 0.05
ONSET_TOLERANCE = 0.05

def get_note_events():
    return None

def get_est_on_offs():
    return None

def get_est_midi_notes():
    return None

def calculate_average_stats(list_stats):
    if len(list_stats) == 0:
        return {}
    
    average_stats = {}
    average_stats['frame_precision'] = np.average([stat['frame_precision'] for stat in list_stats])
    average_stats['frame_recall']    = np.average([stat['frame_recall'] for stat in list_stats])
    average_stats['frame_f1']        = np.average([stat['frame_f1'] for stat in list_stats])
    
    if 'note_precision' in list_stats[0].keys():
        average_stats['note_precision']  = np.average([stat['note_precision'] for stat in list_stats])
        average_stats['note_recall']     = np.average([stat['note_recall'] for stat in list_stats])
        average_stats['note_f1']         = np.average([stat['note_f1'] for stat in list_stats])
    
    if 'reg_note_precision' in list_stats[0].keys():
        average_stats['reg_note_precision']  = np.average([stat['reg_note_precision'] for stat in list_stats])
        average_stats['reg_note_recall']     = np.average([stat['reg_note_recall'] for stat in list_stats])
        average_stats['reg_note_f1']         = np.average([stat['reg_note_f1'] for stat in list_stats])
    
    return average_stats

def calculate_score_per_song(model, dataloader, device):
    """
    Calculate score from paper per song.
    """
    
    stats = {}
    
    result = get_model_outputs(model, dataloader, device, return_targets=True)
    targets = result['targets']
    outputs = result['outputs']
    
    # Calculate frame metric
    if 'frame_output' in outputs.keys():
        frame_threshold = FRAME_THRESHOLD
        y_pred = (np.sign(outputs['frame_output'] - frame_threshold) + 1) / 2
        y_pred[np.where(y_pred==0.5)] = 0
        y_true = targets['frame_roll']
        y_pred = y_pred[0 : y_true.shape[0], :]
        y_true = y_true[0 : y_pred.shape[0], :]

        tmp = metrics.precision_recall_fscore_support(y_true.flatten(), y_pred.flatten())
        stats['frame_precision'] = tmp[0][1]
        stats['frame_recall'] = tmp[1][1]
        stats['frame_f1'] = tmp[2][1]
        
    
    if 'onset_output' in outputs.keys() and 'offset_output' in outputs.keys():
        ref_on_off_pairs = np.array([[event['onset_time'], event['offset_time']] for event in get_note_events()])
        ref_midi_notes = np.array([event['midi_note'] for event in get_note_events()])
        
        note_precision, note_recall, note_f1, _ = \
                    mir_eval.transcription.precision_recall_f1_overlap(
                        ref_intervals=ref_on_off_pairs, 
                        ref_pitches=librosa.note_to_hz(ref_midi_notes), 
                        est_intervals=get_est_on_offs(), 
                        est_pitches=librosa.note_to_hz(get_est_midi_notes()), 
                        onset_tolerance=ONSET_TOLERANCE, 
                        offset_ratio=OFFSET_RATIO, 
                        offset_min_tolerance=OFFSET_MIN_TOLERANCE)
        
        stats['note_precision'] = note_precision
        stats['note_recall'] = note_recall
        stats['note_f1'] = note_f1
        
    if 'reg_onset_output' in outputs.keys() and 'reg_offset_output' in outputs.keys():
        ref_on_off_pairs = np.array([[event['onset_time'], event['offset_time']] for event in get_note_events()])
        ref_midi_notes = np.array([event['midi_note'] for event in get_note_events()])
        
        note_precision, note_recall, note_f1, _ = \
                    mir_eval.transcription.precision_recall_f1_overlap(
                        ref_intervals=ref_on_off_pairs, 
                        ref_pitches=librosa.note_to_hz(ref_midi_notes), 
                        est_intervals=get_est_on_offs(), 
                        est_pitches=librosa.note_to_hz(get_est_midi_notes()), 
                        onset_tolerance=ONSET_TOLERANCE, 
                        offset_ratio=OFFSET_RATIO, 
                        offset_min_tolerance=OFFSET_MIN_TOLERANCE)
        
        stats['reg_note_precision'] = note_precision
        stats['reg_note_recall'] = note_recall
        stats['reg_note_f1'] = note_f1
        
    return stats