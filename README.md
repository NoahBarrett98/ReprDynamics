# ReprDynamics

# NNTF Extension to Unsupervised domain

The aim of this project will be to extend the concepts behind the NNTF to the unsupervised domain. In order to this some modifications to the approach will be made.

## Approach 1, KNN classifier approach

```
K <- number of snapshots
N <- number of neighbours
KNN <- KNN algo 
J <- size of subset to track
Epochs <-number of training epochs
Optimizer <- optimizer for SGD
Data <- dataset for problem
Loss <- loss function for optimizer
f(x) <- neural network
z <- |representation_vector|
--------------------------------------------

step := epochs/k
snapshots := array[k, J, z]
// train and collect representations
for I in range(epochs):
	snapshot = array[len(data), z]
	for x in data:
		x_hat = f(x)
		loss = loss(x, x_hat)
		backprop(optimizer(loss))
		if (i%step):
			z = f(x).representation_layer // must specify which layer to visualize
			snapshot.append(z)

subset := sample(range(data), J)
snapshot = snapshot[:, subset, :] //get snapshots of interest for each iteration
neighbors = array[K, J]
for I, s in enumerate(snapshot):
	KNN = knn_algo(s) // build knn for snapshot, returns N neighbors
    for j, x in enumerate(subset):
        neighbors[I, j] = knn_algo(x)
	
// compute neighbors
distance := array[K,K]
For i in range(K):
	For j in range(K):
		Distance(neighbours[i], neighbours[j])
```

This approach will measure the relative transitions between datapoints within the latent space. For each time step a KNN classifier will be used to quantify the changes in neighbours between a subset of data points. This takes inspiration from the use of KNN as a measure of goodness in learned representations in SSL. 

### problems with approach

A major problem with this approach is that the KNN classifer will be costly, and so we may have to limit the number of samples used when constructing the KNN and utilizing it for inference. 

### benefits of approach

Unlike using clustering techniques, the use of a KNN alleviates the concern of hyper parameter bias in the calculation of similarity between the points. The choice of number of neighbours will only change the resolution of changes to a certain extent and then degrade when distant clusters are reperesented in the transitions. 

### Plan for approach

#### 3 main functions

1. INIT - initialize the UNNTF
2. Store - store representation snapshot during training
3. Visualize - given the stored representations, compute the metrics and visualize matrixs



