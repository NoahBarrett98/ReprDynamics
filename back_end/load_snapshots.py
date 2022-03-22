from UNNTF import UNNTF
from dataset import UNNTFDataset

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


## Load data as if training #
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
batch_size = 2
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                         download=True, transform=transform)
# train_x = trainset.data
# train_y = trainset.targets
# trainset = UNNTFDataset(train_x, train_y, transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
#                                           shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
test_x = testset.data
test_y = testset.targets
testset = UNNTFDataset(test_x, test_y, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False)

data_size = len(testset)
load_path = r"C:\Users\NoahB\Desktop\School\first_year_MCSC_(2021-2022)\CS6406\project\code\ReprDynamics\save_dir\snapshots.json"
save_path = r"C:\Users\NoahB\Desktop\School\first_year_MCSC_(2021-2022)\CS6406\project\code\ReprDynamics\save_dir"
# INIT UNNTF
unntf = UNNTF(
              save_path=save_path,
              load_path=load_path
              )

# unntf.compute_neighbours(max_neighbours=15, 
#                             knn_algo="ball_tree", 
#                             save_path=r"C:\Users\NoahB\Desktop\School\first year MCSC (2021-2022)\CS6406\project\code\ReprDynamics\save_dir\neighbours.json")


mat = unntf.distance_matrix(neighbours=r"C:\Users\NoahB\Desktop\School\first_year_MCSC_(2021-2022)\CS6406\project\code\ReprDynamics\save_dir\neighbours.json", 
                sample_ids=[0,1,2,3,4,5],
                num_neighbours=10,
                distance_metric="euclidean")

import pdb;pdb.set_trace()