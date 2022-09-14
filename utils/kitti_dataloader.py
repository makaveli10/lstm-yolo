import os
import sys

import torch
from torch.utils.data import DataLoader
import numpy as np

from utils.kitti_dataset import KittiTrackingDataset


def create_train_dataloader(configs):
    """Create dataloader for training"""
    train_dataset = KittiTrackingDataset(configs, mode='train')
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=configs.batch_size,
        shuffle=False,
        pin_memory=configs.pin_memory,
        num_workers=configs.num_workers,
        drop_last=True)

    return train_dataloader


def create_val_dataloader(configs):
    """Create dataloader for validation"""
    val_dataset = KittiTrackingDataset(configs, mode='val')
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=configs.batch_size,
        shuffle=False,
        pin_memory=configs.pin_memory,
        num_workers=configs.num_workers,
        drop_last=True)

    return val_dataloader

def create_val_dataloader_with_images(configs):
    """Create dataloader for validation"""
    val_dataset = KittiTrackingDataset(configs, mode='val', images=True)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=configs.batch_size,
        shuffle=False,
        pin_memory=configs.pin_memory,
        num_workers=configs.num_workers,
        drop_last=True)

    return val_dataloader