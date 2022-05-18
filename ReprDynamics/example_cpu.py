from ReprD import ReprD
from dataset import ReprDataset

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os


num_epochs = 20
num_snapshots = 20
batch_size = 8
save_dir = "save_dir"


# sample pytorch model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


## DL EXAMPLE ##
transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
train_x = trainset.data
train_y = trainset.targets
trainset = ReprDataset(train_x, train_y, transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
test_x = testset.data
test_y = testset.targets
testset = ReprDataset(test_x, test_y, transform=transform)


testloader = torch.utils.data.DataLoader(testset,
                                            batch_size=batch_size,
                                         shuffle=False)



# save dataset for inspection later #
data_size = len(testset)
testset.save(save_dir)

# model intialization
net = Net() 
criterion = nn.CrossEntropyLoss() 
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# INIT ReprD
reprd = ReprD(
              data_size=data_size,
              num_epochs=num_epochs,
              num_snapshots=num_snapshots,
              save_path=save_dir
              )
# train model
accuracy = []
loss = []
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(testloader):
        inputs, labels = data["X"], data["y"] 
        ids = data["index"]
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # store snapshot #
        if not epoch % reprd.step_size:
            reprd.store_batch(representation=outputs, ids=ids, epoch=epoch)

print('Finished Training')

# compute neighbours for entire dataset #
neighbours = reprd.compute_neighbours(max_neighbours=15,
                            sample_ids=[1,2,3,4,5,6,7,8],
                            knn_algo="ball_tree",
                            save_path=os.path.join(save_dir, "neighbours.json")
                            )
# compute distance matrix for sample #
mat = reprd.distance_matrix(neighbours=os.path.join(save_dir, "neighbours.json"), 
                    sample_ids=[1,2,3,4,5,6,7,8],
                    num_neighbours=10,
                    distance_metric="euclidean")
# compute neighbours for sample #
neigh = reprd.get_neighbours(neighbours=os.path.join(save_dir, "neighbours.json"), 
                    sample_ids=[1,2,3,4,5,6,7,8],
                    num_neighbours=10)






