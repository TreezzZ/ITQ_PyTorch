import torch
import numpy as np

from sklearn.decomposition import PCA
from utils.evaluate import mean_average_precision


def train(
    train_data,
    query_data,
    query_targets,
    retrieval_data,
    retrieval_targets,
    code_length,
    max_iter,
    device,
    topk,
    ):
    """
    Training model.

    Args
        train_data(torch.Tensor): Training data.
        query_data(torch.Tensor): Query data.
        query_targets(torch.Tensor): Query targets.
        retrieval_data(torch.Tensor): Retrieval data.
        retrieval_targets(torch.Tensor): Retrieval targets.
        code_length(int): Hash code length.
        max_iter(int): Number of iterations.
        device(torch.device): GPU or CPU.
        topk(int): Calculate top k data points map.

    Returns
        mAP(float): Mean Average Precision.
    """
    # Initialization
    query_data, query_targets, retrieval_data, retrieval_targets = query_data.to(device), query_targets.to(device), retrieval_data.to(device), retrieval_targets.to(device)
    R = torch.randn(code_length, code_length).to(device)
    [U, _, _] = torch.svd(R)
    R = U[:, :code_length]

    # PCA
    pca = PCA(n_components=code_length)
    V = torch.from_numpy(pca.fit_transform(train_data.numpy())).to(device)

    # Training
    for i in range(max_iter):
        V_tilde = V @ R
        B = V_tilde.sign()
        [U, _, VT] = torch.svd(B.t() @ V)
        R = (VT.t() @ U.t())

    # Evaluate
    # Generate query code and retrieval code
    query_code = generate_code(query_data.cpu(), code_length, R, pca)
    retrieval_code = generate_code(retrieval_data.cpu(), code_length, R, pca)

    # Compute map
    mAP = mean_average_precision(
        query_code,
        retrieval_code,
        query_targets,
        retrieval_targets,
        device,
        topk,
    )

    return mAP


def generate_code(data, code_length, R, pca):
    """
    Generate hashing code.

    Args
        data(torch.Tensor): Data.
        code_length(int): Hashing code length.
        R(torch.Tensor): Rotration matrix.
        pca(callable): PCA function.

    Returns
        pca_data(torch.Tensor): PCA data.
    """
    return (torch.from_numpy(pca.transform(data.numpy())).to(R.device) @ R).sign()

