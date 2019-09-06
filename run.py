import torch
import numpy as np
import argparse
import os

from loguru import logger

import itq

from data.data_loader import load_data
from model_loader import load_model
from evaluate import mean_average_precision

multi_labels_dataset = [
    'nus-wide-tc-10',
    'nus-wide-tc-21',
    'flickr25k',
]

num_features = {
    'alexnet': 4096,
    'vgg16': 4096,
}


def run():
    # Load configuration
    args = load_config()
    logger.add(os.path.join('logs', '{time}.log'), rotation="500 MB", level="INFO")
    logger.info(args)

    # Load dataset
    query_dataloader, train_dataloader, retrieval_dataloader = load_data(args.dataset,
                                                                         args.root,
                                                                         args.num_query,
                                                                         args.num_train,
                                                                         args.batch_size,
                                                                         args.num_workers,
                                                                         )

    # Extract features
    model = load_model(args.arch, args.code_length)
    model.to(args.device)
    features = extract_features(model, train_dataloader, num_features[args.arch], args.device)

    # Training
    multi_labels = args.dataset in multi_labels_dataset
    pca, R = itq.train(
        features.numpy(),
        args.code_length,
        args.max_iter,
    )

    # Evaluate
    mAP = evaluate(model,
                   query_dataloader,
                   retrieval_dataloader,
                   pca,
                   R,
                   num_features[args.arch],
                   args.device,
                   args.topk,
                   multi_labels,
                   )
    logger.info('[map:{:.4f}]'.format(mAP))


def extract_features(model, dataloader, num_features, device):
    """
    Extract features.

    Args
        model(torch.nn.Module): CNN model.
        dataloader(torch.evaluate.data.DataLoader): Data loader.
        num_features(int): Number of features.
        device(torch.device): Using GPU or CPU.

    Returns
        features(torch.Tensor): Features.
    """
    model.eval()
    model.set_extract_features(True)
    features = torch.zeros(dataloader.dataset.data.shape[0], num_features)
    with torch.no_grad():
        for i, (data, _, index) in enumerate(dataloader):
            data = data.to(device)
            features[index, :] = model(data).cpu()

    model.set_extract_features(False)
    model.train()

    return features


def evaluate(model, query_dataloader, retrieval_dataloader, pca, R, num_features, device, topk, multi_labels):
    """
    Evaluate.

    Args
        model(torch.nn.Module): CNN model.
        query_dataloader(torch.evaluate.data.DataLoader): Query data loader.
        retrieval_dataloader(torch.evaluate.data.DataLoader): Retrieval data loader.
        pca(Callable): PCA.
        R(np.ndarray): Rotation matrix.
        num_features(int): Number of features.
        device(torch.device): GPU or CPU.
        topk(int): Calculate top k data points map.
        multi_labels(bool): Multi labels.

    Returns
        mAP(float): Mean average precision.
    """
    model.eval()

    # Extract features
    query_features = extract_features(model, query_dataloader, num_features, device)
    retrieval_features = extract_features(model, retrieval_dataloader, num_features, device)

    # Generate hash code
    query_code = torch.from_numpy(generate_code(query_features.numpy(), pca, R)).float().to(device)
    retrieval_code = torch.from_numpy(generate_code(retrieval_features.numpy(), pca, R)).float().to(device)

    # One-hot encode targets
    if multi_labels:
        onehot_query_targets = query_dataloader.dataset.get_targets().to(device)
        onehot_retrieval_targets = retrieval_dataloader.dataset.get_targets().to(device)
    else:
        onehot_query_targets = query_dataloader.dataset.get_onehot_targets().to(device)
        onehot_retrieval_targets = retrieval_dataloader.dataset.get_onehot_targets().to(device)

    # Calculate mean average precision
    mAP = mean_average_precision(
        query_code,
        retrieval_code,
        onehot_query_targets,
        onehot_retrieval_targets,
        device,
        topk,
    )
    model.train()

    return mAP


def generate_code(data, pca, R):
    """
    Generate hash code.

    Args
        data(np.ndarray): Data.
        pca(callable), R(np.ndarray): Out-of-samples.

    Returns
        code(np.ndarray): Hash code.
    """
    data = pca.fit_transform(data)
    return np.sign(data @ R)


def load_config():
    """
    Load configuration.

    Args
        None

    Returns
        args(argparse.ArgumentParser): Configuration.
    """
    parser = argparse.ArgumentParser(description='ITQ_PyTorch')
    parser.add_argument('-d', '--dataset',
                        help='Dataset name.')
    parser.add_argument('-r', '--root',
                        help='Path of dataset')
    parser.add_argument('-c', '--code-length', default=12, type=int,
                        help='Binary hash code length.(default: 12)')
    parser.add_argument('-T', '--max-iter', default=50, type=int,
                        help='Number of iterations.(default: 50)')
    parser.add_argument('-q', '--num-query', default=1000, type=int,
                        help='Number of query data points.(default: 1000)')
    parser.add_argument('-t', '--num-train', default=5000, type=int,
                        help='Number of training data points.(default: 5000)')
    parser.add_argument('-w', '--num-workers', default=0, type=int,
                        help='Number of loading data threads.(default: 0)')
    parser.add_argument('-b', '--batch-size', default=24, type=int,
                        help='Batch size.(default: 24)')
    parser.add_argument('-a', '--arch', default='vgg16', type=str,
                        help='CNN architecture.(default: vgg16)')
    parser.add_argument('-k', '--topk', default=5000, type=int,
                        help='Calculate map of top k.(default: 5000)')
    parser.add_argument('-g', '--gpu', default=None, type=int,
                        help='Using gpu.(default: False)')

    args = parser.parse_args()

    # GPU
    if args.gpu is None:
        args.device = torch.device("cpu")
    else:
        args.device = torch.device("cuda:%d" % args.gpu)

    return args


if __name__ == '__main__':
    run()
