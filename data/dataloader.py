import torch
import numpy as np
import scipy.io as sio


def load_data(dataset, root):
    """
    Load dataset.

    Args
        dataset(str): Dataset name.
        root(str): Path of dataset.
    """
    if dataset == 'cifar10-gist':
        return load_data_gist(root)
    elif dataset == 'cifar-10' or dataset == 'nus-wide-tc21' or dataset == 'imagenet-tc100':
        return _load_data(root)
    else:
        raise ValueError('Invalid dataset name!')


def _load_data(root):
    """
    Load alexnet fc7 features.

    Args
        root(str): Path of dataset.

    Returns
        train_data(torch.Tensor, 5000*4096): Training data.
        train_targets(torch.Tensor, 5000*10): One-hot training targets.
        query_data(torch.Tensor, 1000*4096): Query data.
        query_targets(torch.Tensor, 1000*10): One-hot query targets.
        retrieval_data(torch.Tensor, 59000*4096): Retrieval data.
        retrieval_targets(torch.Tensor, 59000*10): One-hot retrieval targets.
    """
    data = torch.load(root)
    train_data = data['train_features']
    train_targets = data['train_targets']
    query_data = data['query_features']
    query_targets = data['query_targets']
    retrieval_data = data['retrieval_features']
    retrieval_targets = data['retrieval_targets']

    # Normalization
    mean = retrieval_data.mean()
    std = retrieval_data.std()
    train_data = (train_data - mean) / std
    query_data = (query_data - mean) / std
    retrieval_data = (retrieval_data - mean) / std

    return train_data, train_targets, query_data, query_targets, retrieval_data, retrieval_targets


def load_data_gist(root):
    """
    Load cifar10-gist dataset.

    Args
        root(str): Path of dataset.

    Returns
        train_data(torch.Tensor, num_train*512): Training data.
        train_targets(torch.Tensor, num_train*10): One-hot training targets.
        query_data(torch.Tensor, num_query*512): Query data.
        query_targets(torch.Tensor, num_query*10): One-hot query targets.
        retrieval_data(torch.Tensor, num_train*512): Retrieval data.
        retrieval_targets(torch.Tensor, num_train*10): One-hot retrieval targets.
    """
    # Load data
    mat_data = sio.loadmat(root)
    query_data = mat_data['testdata']
    query_targets = mat_data['testgnd'].astype(np.int)
    retrieval_data = mat_data['traindata']
    retrieval_targets = mat_data['traingnd'].astype(np.int)

    # One-hot
    query_targets = encode_onehot(query_targets)
    retrieval_targets = encode_onehot(retrieval_targets)

    # Normalization
    data = np.concatenate((query_data, retrieval_data), axis=0)
    data = (data - data.mean()) / data.std()
    query_data = data[:query_data.shape[0], :]
    retrieval_data = data[query_data.shape[0]:, :]

    # Sample training data
    num_train = 5000
    train_index = np.random.permutation(len(retrieval_data))[:num_train]
    train_data = retrieval_data[train_index, :]
    train_targets = retrieval_targets[train_index, :]

    train_data = torch.from_numpy(train_data).float()
    train_targets = torch.from_numpy(train_targets).float()
    query_data = torch.from_numpy(query_data).float()
    query_targets = torch.from_numpy(query_targets).float()
    retrieval_data = torch.from_numpy(retrieval_data).float()
    retrieval_targets = torch.from_numpy(retrieval_targets).float()


    return train_data, train_targets, query_data, query_targets, train_data, train_targets


def encode_onehot(labels, num_classes=10):
    """
    One-hot labels.

    Args:
        labels (numpy.ndarray): labels.
        num_classes (int): Number of classes.

    Returns:
        onehot_labels (numpy.ndarray): one-hot labels.
    """
    onehot_labels = np.zeros((len(labels), num_classes))

    for i in range(len(labels)):
        onehot_labels[i, labels[i]] = 1

    return onehot_labels
