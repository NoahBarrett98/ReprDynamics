import numpy as np
import random
import torch

from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial import distance
from sklearn import metrics

class UNNTF(object):
    def __init__(self, data_size,
                    num_epochs,
                    num_snapshots,
                    ):
        self.data_size = data_size
        self.num_epochs = num_epochs
        self.num_snapshots = num_snapshots

        self.snapshots = []
        self.knns = None
        self.step_size = int(self.num_epochs / self.num_snapshots)



    def store_batch(self, representation, ids, epoch):
        """
        store representations, it is expected that the outputs will be in batches
        :param representation:
        :param batch:
        :return:
        """
        index = int(epoch / self.step_size)
        with torch.no_grad():
            # add first entry otherwise concatonate to tensor
            if len(self.snapshots)-1 < index:
                self.snapshots.append(torch.zeros((self.data_size, representation.shape[-1])))
                self.snapshots[index][ids] = representation
            else:
                self.snapshots[index][ids] = representation


    def build_knn(self, snapshot, num_neighbours, knn_algo):
        knn = KNeighborsClassifier(n_neighbors=num_neighbours,
                                   algorithm=knn_algo)
        # fit but disregard labels #
        knn.fit(X=snapshot, y=np.zeros((len(snapshot))))
        return knn

    def prepare(self, num_neighbours, knn_algo, subset_size):
        subset = random.sample(range(self.data_size), int(self.data_size * subset_size))
        sampled_snaps = [x[subset, :] for x in self.snapshots]
        self.knns = []
        for snap in sampled_snaps:
            self.knns.append(self.build_knn(snap, num_neighbours, knn_algo))

    def hamming_distance(self, a, b):
        a_i = np.zeros((self.data_size, self.data_size))
        b_i = np.zeros((self.data_size, self.data_size))
        for i, (a_h, b_h) in enumerate(zip(a, b)):
            a_i[i, a_h] = 1
            b_i[i, b_h] = 1
        return metrics.hamming_loss(a_i, b_i)

    @staticmethod
    def get_distance_metric(distance_metric):
        if isinstance(distance_metric, str):
            if distance_metric == "euclidean":
                distance_metric = lambda x, y: distance.euclidean(x, y)
            else:
                raise NotImplementedError
        return distance_metric

    def distance_matrix(self, sampled_snaps,
                        num_neighbours,
                        distance_metric):
        neighbours = np.empty((len(sampled_snaps), len(sampled_snaps[0]), num_neighbours), int)
        for i, sample in enumerate(sampled_snaps):
            # build knn on samples
            neighbours_i = np.empty((len(sample), num_neighbours))
            if not self.knns:
                raise Exception('must prepare unntf before computing distance')
            knn = self.knns[i]
            for j, snapshot_i in enumerate(sample):
                neighbours_i[j] = knn.kneighbors(snapshot_i.cpu().detach().numpy().reshape(1, -1),
                                                 num_neighbours,
                                                 False).squeeze()
            neighbours[i] = neighbours_i.astype(int)

        transitions = np.empty((len(sampled_snaps) - 1, len(sampled_snaps[0]), 1))
        for i in range(len(neighbours) - 1):
            transitions[i] = self.hamming_distance(neighbours[i], neighbours[i + 1])
        # distance matrix
        distance_metric = self.get_distance_metric(distance_metric)
        distance = np.empty((self.num_snapshots - 1, self.num_snapshots - 1))
        for i in range(self.num_snapshots - 1):
            for j in range(self.num_snapshots - 1):
                distance[i, j] = distance_metric(transitions[i], transitions[j])

        return distance

    def global_distance_matrix(self,
                               subset_size=1.0,
                               num_neighbours=5,
                               distance_metric="euclidean"
                               ):

        # convert snaps to tensor
        # reduce size of snapshots
        subset = random.sample(range(self.data_size), int(self.data_size*subset_size))
        sampled_snaps = [x[subset, :] for x in self.snapshots]
        # compute distance
        distance = self.distance_matrix(
                                        sampled_snaps,
                                        num_neighbours,
                                        distance_metric)

        return distance

    def local_distance_matrix(self,
                               sample_set,
                               num_neighbours=5,
                               distance_metric="euclidean"
                               ):
        # get subset
        sampled_snaps = [x[sample_set, :] for x in self.snapshots]
        # compute distance
        distance = self.distance_matrix(sampled_snaps,
                                        num_neighbours,
                                        distance_metric)

        return distance

    def save_snapshots(self, fpath):
        np.save(fpath, self.snapshots)

    def load_snapshots(self, fpath):
        self.snapshots = np.load(fpath)








        