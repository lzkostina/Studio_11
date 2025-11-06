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
    def _set_parameters(self, mus, covs, pi):
        """Set model parameters (for compatibility with original class)."""
        self.mu_ = np.array(mus)
        self.cov_ = np.array(covs)
        self.pi_ = np.array(pi)

    # Initialization
    def _initialize_parameters(self, X, theta=None):
        rng = np.random.default_rng(self.random_state)

        if theta is not None:
            self._set_parameters(*theta)
            return

        n_samples, n_features = X.shape
        indices = rng.choice(n_samples, self.n_components, replace=False)
        self.mu_ = X[indices].copy()
        self.cov_ = np.array([np.eye(n_features) for _ in range(self.n_components)])
        self.pi_ = np.ones(self.n_components) / self.n_components


    '''
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
    '''

    
    def _e_step(self, X):
        """E-step: numerically stable computation of responsibilities."""
        n_samples, n_features = X.shape
        log_prob = np.zeros((n_samples, self.n_components))

        # log(π_k * N(x_i | μ_k, Σ_k)) = log π_k + log pdf
        for k in range(self.n_components):
            log_prob[:, k] = (
                np.log(self.pi_[k] + 1e-32)
                + multivariate_normal.logpdf(X, mean=self.mu_[k], cov=self.cov_[k])
            )

        # Use log-sum-exp trick to normalize in log-space
        max_log = np.max(log_prob, axis=1, keepdims=True)
        log_sum_exp = max_log + np.log(
            np.sum(np.exp(log_prob - max_log), axis=1, keepdims=True)
        )

        # Responsibilities: r_ik = exp(log π_k N / log-sum-exp)
        log_resp = log_prob - log_sum_exp
        responsibilities = np.exp(log_resp)

        return responsibilities


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
    '''
        # Log-likelihood (vectorized)
    def _compute_log_likelihood(self, X):
        n_samples = X.shape[0]
        pdfs = np.zeros((n_samples, self.n_components))
        for k in range(self.n_components):
            pdfs[:, k] = self.pi_[k] * multivariate_normal.pdf(X, mean=self.mu_[k], cov=self.cov_[k])
        return np.sum(np.log(pdfs.sum(axis=1) + 1e-12))
    '''

    
    def _compute_log_likelihood(self, X):
        """Compute total log-likelihood using log-sum-exp for stability."""
        n_samples = X.shape[0]
        log_prob = np.zeros((n_samples, self.n_components))

        for k in range(self.n_components):
            log_prob[:, k] = (
                np.log(self.pi_[k] + 1e-32)
                + multivariate_normal.logpdf(X, mean=self.mu_[k], cov=self.cov_[k])
            )

        # Apply log-sum-exp across components
        max_log = np.max(log_prob, axis=1, keepdims=True)
        log_sum_exp = max_log + np.log(np.sum(np.exp(log_prob - max_log), axis=1, keepdims=True))

        return np.sum(log_sum_exp)

    # Fit EM algorithm
    def fit(self, X, max_iter=500, tol=1e-6, initial_theta=None, verbose=True):
        X = np.asarray(X)

        # Initialize parameters
        if initial_theta is not None:
            self._set_parameters(*initial_theta)
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
