import data.cifar10 as cifar10
import data.nus_wide as nuswide
import data.flickr25k as flickr25k


def load_data(dataset, root, num_query, num_train, batch_size, num_workers):
    """
    Load dataset.

    Args
        dataset(str): Dataset name.
        root(str): Path of dataset.
        num_query(int): Number of query data points.
        num_train(int): Number of training data points.
        num_workers(int): Number of loading data threads.

    Returns
        query_dataloader, train_dataloader, retrieval_dataloader(torch.evaluate.data.DataLoader): Data loader.
    """
    if dataset == 'cifar-10':
        query_dataloader, train_dataloader, retrieval_dataloader = cifar10.load_data(root,
                                                                                     num_query,
                                                                                     num_train,
                                                                                     batch_size,
                                                                                     num_workers,
                                                                                     )
    elif dataset == 'nus-wide-tc-10':
        query_dataloader, train_dataloader, retrieval_dataloader = nuswide.load_data(10,
                                                                                     root,
                                                                                     num_query,
                                                                                     num_train,
                                                                                     batch_size,
                                                                                     num_workers,
                                                                                     )
    elif dataset == 'nus-wide-tc-21':
        query_dataloader, train_dataloader, retrieval_dataloader = nuswide.load_data(21,
                                                                                     root,
                                                                                     num_query,
                                                                                     num_train,
                                                                                     batch_size,
                                                                                     num_workers
                                                                                     )
    elif dataset == 'flickr25k':
        query_dataloader, train_dataloader, retrieval_dataloader = flickr25k.load_data(root,
                                                                                       num_query,
                                                                                       num_train,
                                                                                       batch_size,
                                                                                       num_workers,
                                                                                       )
    else:
        raise ValueError("Invalid dataset name!")

    return query_dataloader, train_dataloader, retrieval_dataloader
