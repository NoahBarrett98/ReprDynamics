import torch

class UNNTFDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform


    def __len__(self):
        return len(self.labels)


    def __getitem__(self, index):
        y = self.labels[index]
        X = self.data[index]
        if self.transform is not None:
            X = self.transform(X)

        data = {
            "X": X,
            "y": y,
            "index": index
        }
        return data