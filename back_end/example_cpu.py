from UNNTF import UNNTF
from models import Net
from dataset import UNNTFDataset

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os


num_epochs = 20
num_snapshots = 20
batch_size = 8
save_dir = "save_dir"


## DL EXAMPLE ##
transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                         download=True, transform=transform)
# train_x = trainset.data
# train_y = trainset.targets
# trainset = UNNTFDataset(train_x, train_y, transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
#                                           shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
test_x = testset.data[:16]
test_y = testset.targets[:16]
testset = UNNTFDataset(test_x, test_y, transform=transform)



testloader = torch.utils.data.DataLoader(testset,
                                            batch_size=batch_size,
                                         shuffle=False)

# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# save dataset for inspection later #
data_size = len(testset)
testset.save(save_dir)

# model intialization
net = Net() 
criterion = nn.CrossEntropyLoss() 
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# INIT UNNTF
unntf = UNNTF(
              data_size=data_size,
              num_epochs=num_epochs,
              num_snapshots=num_snapshots,
              save_path=save_dir
              )
accuracy = []
loss = []
for epoch in range(num_epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(testloader):
        inputs, labels = data["X"], data["y"] 
        ids = data["index"]
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        if not epoch % unntf.step_size:
            unntf.store_batch(representation=outputs, ids=ids, epoch=epoch)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

print('Finished Training')

neighbours = unntf.compute_neighbours(max_neighbours=15,
                            knn_algo="ball_tree",
                            save_path=os.path.join(save_dir, "neighbours.json"))

mat = unntf.distance_matrix(neighbours=os.path.join(save_dir, "neighbours.json"), 
                    sample_ids=[1,2,3,4,5,6,7,8],
                    num_neighbours=10,
                    distance_metric="euclidean")

neigh = unntf.get_neighbours(neighbours=os.path.join(save_dir, "neighbours.json"), 
                    sample_ids=[1,2,3,4,5,6,7,8],
                    num_neighbours=10)

import pdb;pdb.set_trace()





