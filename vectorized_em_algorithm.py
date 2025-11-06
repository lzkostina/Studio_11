# vectorized_em_algorithm.py
import numpy as np
from scipy.stats import multivariate_normal
from tqdm import tqdm

class GaussianMixtureModelVectorized:
    """
    Vectorized Gaussian Mixture Model using the EM algorithm.

    Parameters
    ----------
    n_components : int
        Number of mixture components.
    random_state : int or None, optional
        Random seed for reproducibility.
    """

    def __init__(self, n_components=2, random_state=None):
        self.n_components = n_components
        self.random_state = random_state

        self.mu_ = None
        self.cov_ = None
        self.pi_ = None

        self.log_likelihoods_ = []
        self.converged_ = False
        self.n_iter_ = 0

    # Initialization
    def _initialize_parameters(self, X):
        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = X.shape

        indices = rng.choice(n_samples, self.n_components, replace=False)
        self.mu_ = X[indices].copy()
        self.cov_ = np.array([np.eye(n_features) for _ in range(self.n_components)])
        self.pi_ = np.ones(self.n_components) / self.n_components

    # E-step (vectorized)
    def _e_step(self, X):
        n_samples = X.shape[0]
        probs = np.zeros((n_samples, self.n_components))

        # Compute π_k * N(x_i | μ_k, Σ_k) for all k
        for k in range(self.n_components):
            probs[:, k] = self.pi_[k] * multivariate_normal.pdf(X, mean=self.mu_[k], cov=self.cov_[k])

        # Normalize across components to get responsibilities
        row_sums = probs.sum(axis=1, keepdims=True)
        probs = np.divide(probs, row_sums, where=row_sums > 0)
        probs[row_sums.squeeze() == 0] = 1.0 / self.n_components
        return probs

    # M-step (vectorized)
    def _m_step(self, X, responsibilities):
        n_samples, n_features = X.shape
        N_k = responsibilities.sum(axis=0)  # shape (K,)

        # Update means
        self.mu_ = (responsibilities.T @ X) / N_k[:, None]

        # Update covariances (loop over components)
        covs = []
        for k in range(self.n_components):
            diff = X - self.mu_[k]
            cov_k = (responsibilities[:, k][:, None] * diff).T @ diff / N_k[k]
            cov_k += 1e-6 * np.eye(n_features)
            covs.append(cov_k)
        self.cov_ = np.array(covs)

        # Update mixing weights
        self.pi_ = N_k / n_samples

    # Log-likelihood (vectorized)
    def _compute_log_likelihood(self, X):
        n_samples = X.shape[0]
        pdfs = np.zeros((n_samples, self.n_components))
        for k in range(self.n_components):
            pdfs[:, k] = self.pi_[k] * multivariate_normal.pdf(X, mean=self.mu_[k], cov=self.cov_[k])
        return np.sum(np.log(pdfs.sum(axis=1) + 1e-12))

    # Fit EM algorithm
    def fit(self, X, max_iter=500, tol=1e-6, verbose=True, initial_theta=None):
        X = np.asarray(X)

        # Initialize parameters
        if initial_theta is not None:
            self.mu_, self.cov_, self.pi_ = initial_theta
        else:
            self._initialize_parameters(X)

        self.log_likelihoods_ = []

        for iteration in tqdm(range(max_iter), disable=not verbose):
            # E-step
            responsibilities = self._e_step(X)

            # M-step
            self._m_step(X, responsibilities)

            # Compute log-likelihood
            ll = self._compute_log_likelihood(X)
            self.log_likelihoods_.append(ll)

            # Check convergence
            if iteration > 0 and abs(self.log_likelihoods_[-1] - self.log_likelihoods_[-2]) < tol:
                self.converged_ = True
                self.n_iter_ = iteration + 1
                break
        else:
            self.n_iter_ = max_iter

        return self

    # Prediction
    def predict_proba(self, X):
        return self._e_step(np.asarray(X))

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X):
        return self._compute_log_likelihood(np.asarray(X))

    # Sampling
    def sample(self, n_samples=1, random_state=None):
        if self.mu_ is None:
            raise ValueError("Model must be fitted before sampling.")

        rng = np.random.default_rng(random_state)
        n_features = self.mu_.shape[1]
        X = np.zeros((n_samples, n_features))
        labels = np.zeros(n_samples, dtype=int)

        for i in range(n_samples):
            k = rng.choice(self.n_components, p=self.pi_)
            X[i] = rng.multivariate_normal(self.mu_[k], self.cov_[k])
            labels[i] = k

        return X, labels
