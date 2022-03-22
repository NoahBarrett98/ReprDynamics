import numpy as np
import random
import torch
import os
import json

from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial import distance
from sklearn import metrics

class UNNTF(object):
    def __init__(self, data_size=None,
                    num_epochs=None,
                    num_snapshots=None,
                    save_path=None,
                    load_path=None
                    ):
        self.data_size = data_size
        self.num_epochs = num_epochs
        self.num_snapshots = num_snapshots
        
        if save_path:
            self.save_path = os.path.join(save_path, "snapshots")
        else:
            self.save_path = save_path
        if os.path.isdir(self.save_path):
            os.mkdir(self.save_path)
        self.load_path = load_path
        if self.load_path:
            with open(self.load_path, "r") as f:
                self.snapshots = json.load(f)
            self.store_batch_conf = False
            if not self.data_size:
                self.data_size = len(self.snapshots[0])
        else:
            self.snapshots = []
            # compute step size
            self.step_size = int(self.num_epochs / self.num_snapshots)
            self.store_batch_conf = True

    def store_batch(self, representation, ids, epoch):
        """
        store representations, it is expected that the outputs will be in batches
        :param representation: 
        :param batch:
        :return:
        """
        if self.store_batch_conf:
            index = int(epoch / self.step_size)
            with torch.no_grad():
                # add first entry otherwise concatonate to tensor
                if len(self.snapshots)-1 < index:
                    self.snapshots.append(np.zeros((self.data_size, representation.shape[-1])))
                    self.snapshots[index][ids] = representation.detach().cpu().numpy()
                else:
                    self.snapshots[index][ids] = representation.detach().cpu().numpy()

            if self.save_path:
                self._save_snapshots()
        else:
            raise Exception('Not configured for store_batch')


    
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


    def _build_knn(self, snapshot, num_neighbours, knn_algo):
        knn = KNeighborsClassifier(n_neighbors=num_neighbours,
                                   algorithm=knn_algo)
        # fit but disregard labels #
        knn.fit(X=snapshot, y=np.zeros((len(snapshot))))
        return knn

    def _prepare_knns(self, num_neighbours, knn_algo):
        knns = []
        for snap in self.snapshots:
            knns.append(self._build_knn(snap, num_neighbours, knn_algo))
        return knns


    def compute_neighbours(self, max_neighbours, knn_algo, save_path=None):
        """
        compute neighbours for snapshot
        """
        # prepare knns 
        knns = self._prepare_knns(max_neighbours, knn_algo)
        # compute neighbours #
        neighbours = np.empty((len(self.snapshots), len(self.snapshots[0]), max_neighbours), int)
        neighbours = []
        for i, sample in enumerate(self.snapshots):
            neighbours_i = {}
            knn = knns[i]
            for j, snapshot_i in enumerate(sample):
                knn_i = knn.kneighbors(np.array(snapshot_i).reshape(1, -1),
                                                 max_neighbours,
                                                 True)
                
                neighbours_i[str(j)] = {
                    "distances":knn_i[0].squeeze().tolist(),
                    "neighbours":knn_i[1].squeeze().astype(int).tolist(),
                    "representation":list(snapshot_i)
                }
            neighbours.append(neighbours_i)
        
        if save_path:
            with open(save_path, "w") as f:
                json.dump({"knn_algorithm":knn_algo,
                            "max_neighbours":max_neighbours,
                            "neighbours":neighbours}, f)

        return neighbours

    def distance_matrix(self, 
                        neighbours, 
                        sample_ids,
                        num_neighbours,
                        distance_metric="euclidean"):
        if isinstance(neighbours, str):
            with open(neighbours, "r") as f:
                neighbours = json.load(f)
        if num_neighbours > len(neighbours["neighbours"][0]["0"]["neighbours"]) -1:
            raise Exception(f"num neighbours ({num_neighbours}) > saved neighbours ({len(neighbours[0]['0']['neighbours'])})")
        
        neighbour_ids = []
        for neighbour_i in neighbours["neighbours"]:
            neighbour_ids.append([neighbour_i[str(id)]["neighbours"][:num_neighbours] for id in sample_ids])

        transitions = np.empty((len(sample_ids) - 1, len(neighbour_ids[0]), 1))
        for i in range(len(neighbours) - 1):
            transitions[i] = self.hamming_distance(neighbour_ids[i], neighbour_ids[i + 1])
        # distance matrix
        distance_metric = self.get_distance_metric(distance_metric)
        distance = np.empty((len(neighbours) - 1,len(neighbours) - 1))
        for i in range(len(neighbours) - 1):
            for j in range(len(neighbours) - 1):
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

    def _save_snapshots(self):
        with open(self.save_path+".json" if ".json" not in self.save_path else self.save_path, "w") as f:
            json.dump([s.tolist() for s  in self.snapshots], f)
        

    def load_snapshots(self, fpath):
        self.snapshots = np.load(fpath)








        