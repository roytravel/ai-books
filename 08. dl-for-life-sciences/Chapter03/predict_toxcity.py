import deepchem as dc
import numpy as np

# create simple random numpy array
x = np.random.random((4, 5))
y = np.random.random((4, 1))

# save array as NumpyDataset object
dataset = dc.data.NumpyDataset(x, y)
print (dataset.X, dataset.y)

# check equality
print (np.array_equal(x, dataset.X))
print (np.array_equal(y, dataset.y))

# create toxcity molecule prediction model
## load dataset related to toxcity
tox21_tasks, tox21_datasets, transformers = dc.molnet.load_tox21()
print(tox21_tasks)

## check shape of dataset
train_dataset, valid_dataset, test_dataset = tox21_datasets
print (test_dataset)
print(train_dataset.X.shape, valid_dataset.X.shape, test_dataset.X.shape)
print(np.shape(train_dataset.y), np.shape(valid_dataset.y), np.shape(test_dataset.y))

## check missing value
print (train_dataset.w.shape)
print (np.count_nonzero(train_dataset.w))
print (np.count_nonzero(train_dataset.w == 0))

## check what tool is used to modify original data: BalancingTransformer
print (transformers)

## create model
model = dc.models.MultitaskClassifier(n_tasks=12, n_features=1024, layer_size=[1000]) # layer_size = [layer width, layer depth]. 1000 means 1 hidden layer with 1024 neuron
model.fit(train_dataset, nb_epoch=2)

## metric: roc_auc
metric = dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean)

# evaluate model
train_scores = model.evaluate(train_dataset, [metric], transformers)
test_scores = model.evaluate(test_dataset, [metric], transformers)
print (train_scores, test_scores)