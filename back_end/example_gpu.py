from UNNTF import UNNTF
from models import Net
from dataset import UNNTFDataset
from evaluation import evaluate_classification

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import time


exp_dir = "/home/noah/UNNTF/plots"
## DL EXAMPLE ##

batch_size = 16
def get_data(root, batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root=root, train=True,
                                            download=True, transform=transform)
    train_x = trainset.data
    train_y = trainset.targets
    trainset = UNNTFDataset(train_x, train_y, transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True)

    testset = torchvision.datasets.CIFAR10(root=root, train=False,
                                           download=True, transform=transform)
    test_x = testset.data
    test_y = testset.targets
    testset = UNNTFDataset(test_x, test_y, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    data_size = len(testset)
    return trainloader, testloader, data_size, classes

trainloader, testloader, data_size, classes = get_data(root="./data", batch_size=batch_size)

# INIT UNNTF
def train(trainloader, testloader, data_size, num_epochs, num_snapshots, exp_dir):
    unntf = UNNTF(
                  data_size=data_size,
                  num_epochs=num_epochs,
                  num_snapshots=num_snapshots,
                  )
    net = Net().cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    accuracy = []
    auc = []
    losses = []
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(testloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data["X"].cuda(), data["y"].cuda()
            ids = data["index"]
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            if not epoch % unntf.step_size:
                unntf.store_batch(representation=outputs.cpu(), ids=ids, epoch=epoch)
            loss = criterion(outputs, labels)
            running_loss+=loss.cpu().item()
            loss.backward()
            optimizer.step()
        if not epoch % unntf.step_size:
            report = evaluate_classification(net, testloader)
            accuracy.append(report["accuracy"])
            auc.append(report["auc"])
            losses.append(running_loss/len(data))
           

    print('Finished Training')
    plt.plot(accuracy)
    plt.savefig(os.path.join(exp_dir, "acc.png"))
    plt.clf()
    plt.plot(auc)
    plt.savefig(os.path.join(exp_dir, "auc.png"))
    plt.clf()
    plt.plot(losses)
    plt.savefig(os.path.join(exp_dir, "loss.png"))
    plt.clf()

    return unntf, accuracy, auc, loss




num_epochs = 100
time_outcomes = []
for num_snaps in [10, 50, 100]:
    exp_dir_i = os.path.join(exp_dir, f"{num_snaps}_snapshots")
    if not os.path.isdir(exp_dir_i):
        os.mkdir(exp_dir_i)
    unntf, accuracy, auc, loss = train(trainloader,
                                       testloader,
                                       data_size,
                                       num_epochs,
                                       num_snaps,
                                       exp_dir_i)


    # functions to show an image
    def imshow(img, fpath):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.savefig(fpath)
        plt.clf()


    # get some random training images
    dataiter = iter(testloader)
    data = dataiter.next()
    images, labels, ids = data["X"], data["y"], data["index"]
    # show images
    imshow(torchvision.utils.make_grid(images), os.path.join(exp_dir_i, "local_samples.png"))
    with open(os.path.join(exp_dir_i, "classes.txt"), "w") as f:
        f.write(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
    for num_neighbours in [2, 5, 10]:
        for subset_size in [0.01, 0.1, 0.5]:
            # prepare unntf
            prep_time = time.time()
            unntf.prepare(num_neighbours=num_neighbours,
                          knn_algo="ball_tree",
                          subset_size=subset_size)
            prep_time = time.time()-prep_time
            # global view #
            global_time = time.time()
            d_mat = unntf.global_distance_matrix(subset_size=1.0,
                                                 num_neighbours=num_neighbours)
            global_time = time.time() - global_time
            sns.heatmap(d_mat)
            plt.savefig(os.path.join(exp_dir_i, f"global_{num_neighbours}_{subset_size}.png"))
            plt.clf()
            # local view #
            local_time = time.time()
            d_mat = unntf.local_distance_matrix(sample_set=ids,
                                                 num_neighbours=num_neighbours)
            local_time = time.time() - local_time
            sns.heatmap(d_mat)
            plt.savefig(os.path.join(exp_dir_i, f"local_{num_neighbours}_{subset_size}.png"))
            plt.clf()

            time_outcomes.append([num_snaps, num_neighbours, subset_size,
                                  prep_time, global_time, local_time])

    stats = pd.DataFrame(time_outcomes, columns=["num_snaps",
                                                 "num_neighbours",
                                                 "subset_size",
                                                 "prep_time",
                                                 "global_time",
                                                 "local_time"])
    stats.to_csv(os.path.join(exp_dir, "time_outcomes.csv"))
    # save unntf snapshots #
    unntf.save_snapshots(os.path.join(exp_dir_i, "snapshots"))
