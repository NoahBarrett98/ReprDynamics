import torch
import os
import numpy as np
import torch.utils.data as td
import torchvision

class ReprDataset(td.Dataset):
    def __init__(self, data, labels=None, transform=None):
        
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)
    def save(self, fpath):
        if not os.path.isdir(os.path.join(fpath, "data")):
            os.mkdir(os.path.join(fpath, "data"))
        np.save(os.path.join(os.path.join(fpath, "data"), "data"), self.data)
    
        if self.labels:
            np.save(os.path.join(os.path.join(fpath, "data"), "labels"), self.labels)

    def __getitem__(self, index):
        X = self.data[index]
        if self.labels:
            y = self.labels[index]
        else:
            y = None
        sample = {
                        "X": X,
                        "y": y,
                        "index": index
                    }
        if self.transform is not None:
            sample["X"] = self.transform(sample["X"])
        return sample