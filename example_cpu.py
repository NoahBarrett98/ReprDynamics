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
import sklearn

net = Net()
## DL EXAMPLE ##
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
batch_size = 2
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
train_x = trainset.data
train_y = trainset.targets
trainset = UNNTFDataset(train_x, train_y, transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
test_x = testset.data
test_y = testset.targets
testset = UNNTFDataset(test_x, test_y, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


num_epochs = 10
data_size = len(testset)
# INIT UNNTF
unntf = UNNTF(
              data_size=data_size,
              num_epochs=num_epochs,
              num_snapshots=10,
              )
accuracy = []
loss = []
for epoch in range(num_epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(testloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data["X"], data["y"]
        ids = data["index"]
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        if not epoch % unntf.step_size:
            unntf.store_batch(representation= outputs, ids=ids, epoch=epoch)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # if not epoch % unntf.step_size:
    #     all_outputs = torch.tensor([])
    #     for i, data in enumerate(testloader, 0):
    #         # get the inputs; data is a list of [inputs, labels]
    #         inputs, labels = data["X"], data["y"]
    #         ids = data["index"]
    #         outputs = net(inputs)
    #         torch.concat([all_outputs, outputs], dim=0)

        # sklearn.metrics.accuracy_score()

print('Finished Training')
# global view #
d_mat = unntf.global_distance_matrix(subset_size=0.5,
                                     num_neighbours=5)
sns.heatmap(d_mat)
plt.show()
plt.clf()


# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    plt.clf()

# get some random training images
dataiter = iter(testloader)
data = dataiter.next()
images, labels, ids = data["X"], data["y"], data["index"]
# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))


# local view #
d_mat = unntf.local_distance_matrix(sample_set=ids,
                                     num_neighbours=2)
sns.heatmap(d_mat)
plt.show()
plt.clf()

unntf.save_snapshots("./snapshots")
