import os
import numpy as np
import re
import torch
from torch.utils.data import Dataset

from utils.kitti_base import read_labels, read_image


class KittiTrackingDataset(Dataset):
    def __init__(self, config, mode='train', images=False):
        seqs = config.train_seqs if mode=='train' else config.val_seqs
        self.tracking_seqs = [str(seq_id).zfill(4) for seq_id in seqs]
        self.input_size = config.input_size
        self.root_path = config.root_path
        self.classes = config.tracking_type
        self.samples = None
        self.targets = None
        self.read_all_object_labels(config)
        self.index = 0
        self.read_images = images

    def read_all_object_labels(self, config, max_seq_len=10):
        samples, targets = [], []
        images = []
        
        for seq_name in self.tracking_seqs:
            labels, image_paths = read_labels(self.root_path, seq_name, self.classes)

            for k, v in labels.items():
                if len(v) < 10:
                    sample = []
                    padding = max_seq_len - len(v) + 1
                    pad = [[0.,0.,0.,0.] for i in range(padding)]
                    sample.extend(pad)
                    
                    for i in range(len(v)-1):
                        sample.append(v[i])
                    samples.append(sample)
                    targets.append(v[-1])
                    images.append(read_image(image_paths[k][-1]))  
                else:
                    rolling_v = np.lib.stride_tricks.sliding_window_view(
                        np.array(v), window_shape=(max_seq_len, self.input_size))
                    rolling_v = rolling_v.reshape(rolling_v.shape[0], max_seq_len, self.input_size)
                    target_index = max_seq_len
                    for i in range(len(rolling_v) - 1):
                        samples.append(rolling_v[i])
                        targets.append(v[target_index])
                        images.append(image_paths[k][target_index])
                        target_index += 1

        self.samples = np.array(samples)
        self.targets = np.array(targets)
        self.images = images

    def __len__(self):
        return len(self.samples)-1
    
    def __getitem__(self, index):
        sample = torch.from_numpy(self.samples[index])
        target = torch.from_numpy(self.targets[index])
        if self.read_images:
            image = self.images[index]
            return sample.float(), target.float(), image
        return sample.float(), target.float()
        

# if __name__=="__main__":
#     root_path = '/opt/vineet-workspace/datasets/kitti_tracking/training'
#     seq_id = 0
#     dataset = KittiTrackingDataset(root_path, seq_id)