# ReprDynamics

![alt text](img/dashboard.png)

This is a tool for visualizing the training dynamics of a neural network. It is inspired by the Neural Network Training Finger Print https://link.springer.com/article/10.1007/s12650-021-00809-4

## Installing ReprDynamics

To install ReprDynamics you must clone this repo and install all dependencies.

```
git clone https://github.com/NoahBarrett98/ReprDynamics
cd ./path/to/ReprDynamics
conda create --name ReprDynamics python=3.9
conda activate ReprDynamics
pip install -e .

```


## Run sample visualization

A sample training session is provided in ./sample. 16 images from cifar-10 were used to train a simple neural network. Once ReprDynamics is installed, you can visualize this training session (flask will host the visualization on port 5000 by default):

```
cd /path/to/ReprDynamics
ReprDynamics --save_dir ./sample
```


## Using ReprDynamics to store training snapshots and computing transition heat maps

This repo is tested using pytorch, however is easily extendable to Tensorflow and other deep learning frameworks by overriding the store_batch() function found in back_end/ReprD.py. There are four steps to incorporating ReprDynamics into a pre existing workflow. 


1. Wrap dataset with ReprDataset and save dataset for future analysis
```
dataset = ReprDataset(data_x, data_y, transform=transform)
dataset.save(save_dir)
```

2. Initialize ReprD
```
reprd = ReprD(
              data_size=len(dataset),
              num_epochs=num_epochs,
              num_snapshots=num_snapshots,
              save_path=save_dir
              )
```

3. Store Snapshots during training
```
for epoch in range(num_epochs):
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
```

4. compute neighbours and associated distance matrix 

```
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

```


An example for both the CPU and GPU can be found in ./back_end. 