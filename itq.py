import numpy as np

from sklearn.decomposition import PCA
from numpy.linalg import svd


def train(data,
          code_length,
          max_iter,
          ):
    """
    Training model.

    Args
        data(np.ndarray): Data.
        multi_labels(bool): True, if dataset is multi-labels.
        code_length(int): Hash code length.
        num_features(int): Number of features.
        max_iter(int): Number of iterations.
        arch(str): Model name.
        device(torch.device): GPU or CPU.
        topk(int): Calculate top k data points map.

    Returns
        pca(Callable): PCA.
        R(np.ndarray): Rotation matrix.
    """
    # PCA
    pca = PCA(n_components=code_length).fit(data)
    V = pca.fit_transform(data)

    # Initialization
    R = np.random.randn(code_length, code_length)
    [u, _, _] = svd(R)
    R = u[:, :code_length]

    # Training
    for i in range(max_iter):
        V_tilde = V @ R
        B = np.sign(V_tilde)
        [ua, _, vta] = svd(B.T @ V)
        R = (vta.T @ ua.T)

    return pca, R
