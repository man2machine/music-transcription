# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 01:21:39 2020

@author: Shahir
"""

import os
import time
import datetime
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from noteify.core.evaluation import targets_to_device
from noteify.core.losses import compute_transcription_losses
from noteify.core.evaluation import get_evaluation_stats

def make_optimizer(model, lr=0.001, verbose=False):
    # Get all the parameters
    params_to_update = model.parameters()
    
    if verbose:
        print("Params to learn:")
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                print("\t", name)
    
    optimizer = optim.Adam(params_to_update, lr=lr,
        betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True)
    nn.utils.clip_grad_norm_(params_to_update, 3.0)
    
    return optimizer

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def set_optimizer_lr(optimizer,
                     lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def make_scheduler(optimizer, epoch_steps, gamma):
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, epoch_steps, gamma=gamma)
    return scheduler

def get_timestamp():
    return datetime.datetime.now().strftime("%m-%d-%Y %I-%M%p")

class ModelTracker:
    def __init__(self, root_dir): 
        experiment_dir = "Experiment {}".format(get_timestamp())
        self.save_dir = os.path.join(root_dir, experiment_dir)
        self.best_model_metric = float('-inf')
        self.record_per_epoch = {}
        os.makedirs(self.save_dir, exist_ok=True)
    
    def update_info_history(self,
                            epoch,
                            info):
        self.record_per_epoch[epoch] = info
        fname = "Experiment Epoch Info History.pckl"
        with open(os.path.join(self.save_dir, fname), 'wb') as f:
            pickle.dump(self.record_per_epoch, f)
    
    def update_model_weights(self,
                             epoch,
                             model_state_dict,
                             metric=None,
                             save_best=True,
                             save_current=True):
        update_best = metric is None or metric > self.best_model_metric
        
        if save_best and update_best:
            torch.save(model_state_dict, os.path.join(self.save_dir,
                "Weights Best.pckl"))
        if save_current:
                torch.save(model_state_dict, os.path.join(self.save_dir,
                    "Weights Epoch {} {}.pckl".format(epoch, get_timestamp())))
            
def train_model(
    device,
    model,
    dataloaders,
    optimizer,
    save_dir,
    lr_scheduler=None,
    save_model=False,
    save_best=False,
    save_all=False,
    save_log=False,
    num_epochs=1,
    test_batch_interval=None,
    train_batch_multiplier=1):
    
    start_time = time.time()
    
    tracker = ModelTracker(save_dir)
    
    train_batch_multiplier = int(train_batch_multiplier)

    epoch_phases = ['train', 'test']
    
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)
        
        train_loss_info = {}
        
        # Each epoch has a training and validation phase
        for epoch_phase in epoch_phases:
            if epoch_phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            
            if epoch_phase == 'train':
                running_loss = 0.0   
                running_count = 0
                batch_step_count = 0
                batch_num = 0
                
                train_loss_record = []
                pbar = tqdm(dataloaders['train'])
                for inputs, targets in pbar:
                    inputs = inputs.to(device)
                    targets_to_device(targets, device)
                    
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    
                    # forward
                    with torch.set_grad_enabled(True):
                        # Get model outputs and calculate loss
                        outputs = model(inputs)
                        loss = compute_transcription_losses(outputs, targets)
                        
                        # loss parts are for debugging purposes
                        loss_parts = loss
                        try:
                            iter(loss_parts)
                        except TypeError:
                            loss_parts = [loss_parts]
                        
                        loss = sum(loss_parts)
                        train_loss_record.append([n.detach().item() for n in loss_parts])
                        
                        loss.backward()
                        if batch_step_count == 0:
                            optimizer.step()
                            batch_step_count = train_batch_multiplier
                    
                        batch_step_count -= 1
                        
                    running_loss += loss.detach().item() * inputs.size(0)
                    running_count += inputs.size(0)
                    epoch_loss = running_loss / running_count
                    
                    loss_fmt = "{:.4f}"
                    desc = "Avg. Loss: {}, Total Loss: {}, Loss Parts: [{}]"
                    desc = desc.format(loss_fmt.format(epoch_loss),
                                       loss_fmt.format(sum(loss_parts)),
                                       ", ".join(loss_fmt.format(n.item()) for n in loss_parts))
                    pbar.set_description(desc)
                    
                    del loss, loss_parts

                    batch_num += 1
                    if test_batch_interval and ((batch_num % test_batch_interval) == 0):
                        stats = get_evaluation_stats(model, dataloaders['test'], device)
                        print("Test statistics:", stats)
                pbar.close()

                print("Training Loss: {:.4f}".format(epoch_loss))
                train_loss_info['loss'] = train_loss_record
            
            elif epoch_phase == 'test' and test_batch_interval is None:
                stats = get_evaluation_stats(model, dataloaders['test'], device)
                print("Test statistics:", stats)
            
            torch.cuda.empty_cache()
        
        if save_model:
            model_weights = model.state_dict()
            tracker.update_model_weights(epoch,
                                         model_weights,
                                         save_best=save_best,
                                         save_current=save_all)
            info = {'train_loss_history': train_loss_info}
        
        if save_log:
            tracker.update_info_history(epoch, info)
        
        print()
        
        if lr_scheduler:
            lr_scheduler.step()

    time_elapsed = time.time() - start_time
    print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    
    return tracker
