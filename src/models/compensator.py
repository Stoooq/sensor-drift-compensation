import numpy as np
from scipy.linalg import sqrtm
from sklearn.decomposition import PCA


class CORALCompensator:
    def __init__(self):
        self.source_cov_sqrt = None
        self.source_mean = None

    def fit(self, X_source: np.ndarray) -> None:
        self.source_mean = X_source.mean(axis=0)

        source_cov = np.cov(X_source.T) + 1e-5 * np.eye(X_source.shape[1])
        self.source_cov_sqrt = sqrtm(source_cov).real

    def transform(self, X_target: np.ndarray) -> np.ndarray:
        mu_T = X_target.mean(axis=0)
        X_centered = X_target - mu_T

        target_cov = np.cov(X_centered.T) + 1e-5 * np.eye(X_centered.shape[1])
        target_cov_sqrt = sqrtm(target_cov).real

        X_aligned = X_centered @ np.linalg.inv(target_cov_sqrt) @ self.source_cov_sqrt

        return X_aligned + self.source_mean


class PCAWhiteningCompensator:
    def __init__(self):
        self.pca = None

    def fit(self, X_source: np.ndarray) -> None:
        self.pca = PCA(whiten=True, n_components=10)
        self.pca.fit(X_source)

    def transform(self, X_target: np.ndarray) -> np.ndarray:
        X_transformed = self.pca.transform(X_target)

        return X_transformed
