import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
from scipy.linalg import cholesky, solve_triangular
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

plt.rcParams.update({
    'lines.linewidth': 1.,
    'lines.markersize': 5,
    'font.size': 9,
    "text.usetex": True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
    'text.latex.preamble': r'\usepackage{amsmath,amsfonts}',
    'axes.linewidth': .75})


def _plot_gaussian_ellipse(mu, cov, ax, color, alpha=0.3, label=None):
    """Plot an ellipse representing a 2D Gaussian"""
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2 * np.sqrt(eigenvalues) * 2  # 2 standard deviations

    ellipse = Ellipse(mu, width, height, angle=angle,
                      facecolor=color, alpha=alpha, edgecolor=color, linewidth=2, label=label)
    ax.add_patch(ellipse)


class GaussianMixtureModelVectorized:
    """
    Gaussian Mixture Model fitted using the EM algorithm.

    Parameters
    ----------
    n_components : int, default=2
        Number of mixture components
    random_state : int or None, default=None
        Random seed for reproducibility
    """

    def __init__(self, n_components=2, random_state=None):
        self.n_components = n_components
        self.random_state = random_state

        # Parameters (set after fitting)
        self.mu_ = None
        self.cov_ = None
        self.pi_ = None

        # Diagnostics (set after fitting)
        self.log_likelihoods_ = None
        self.n_iter_ = None
        self.converged_ = False
        self.reg_ = 1e-6

    def fit(self, X, max_iter=500, tol=1e-6, initial_theta=None, verbose=True):
        """
        Fit the GMM to data using the EM algorithm.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data
        max_iter : int, default=100
            Maximum number of EM iterations
        tol : float, default=1e-4
            Convergence tolerance (change in log-likelihood)


        Returns
        -------
        self : object
            Returns self
        """
        self.max_iter = max_iter
        self.tol = tol

        X = np.asarray(X)
        n_samples, n_features = X.shape

        # Initialize parameters
        self._initialize_parameters(X, initial_theta)

        # Track log-likelihood
        self.log_likelihoods_ = []

        # Main EM loop
        for iteration in tqdm(range(self.max_iter), disable=not verbose):
            # E-step
            responsibilities = self._e_step(X)

            # M-step
            self._m_step(X, responsibilities)

            # Compute log-likelihood
            ll = self._compute_log_likelihood(X)
            self.log_likelihoods_.append(ll)

            # Check convergence
            if iteration > 0:
                ll_change = abs(self.log_likelihoods_[-1] - self.log_likelihoods_[-2])
                if ll_change < self.tol:
                    self.converged_ = True
                    self.n_iter_ = iteration + 1
                    break
        else:
            self.n_iter_ = self.max_iter

        return self

    def _set_parameters(self, mus, covs, pi):
        # Convert to numpy arrays for vectorized operations
        self.mu_ = np.asarray(mus)
        self.cov_ = np.asarray(covs)
        self.pi_ = np.asarray(pi)

        # Precompute Cholesky decomposition of precision matrices for log-sum-exp trick
        self._compute_precisions_cholesky()

    def _initialize_parameters(self, X, theta=None):
        """Initialize GMM parameters randomly."""
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples, n_features = X.shape

        if theta is not None:
            self._set_parameters(*theta)
        else:
            # Initialize means by randomly selecting data points
            indices = np.random.choice(n_samples, self.n_components, replace=False)
            self._set_parameters(
                X[indices].copy(),
                [np.eye(n_features) for _ in range(self.n_components)],
                np.ones(self.n_components) / self.n_components
            )

    def _compute_precisions_cholesky(self):
        """
        Compute the Cholesky decomposition of precision matrices.

        For numerical stability in log-sum-exp computations, we precompute
        the Cholesky decomposition L of the precision matrix Σ^(-1) = L L^T.
        This allows us to compute log densities without overflow/underflow.
        """
        n_features = self.mu_.shape[1]
        self.precisions_chol_ = np.empty((self.n_components, n_features, n_features))

        for k in range(self.n_components):
            try:
                # Cholesky decomposition of covariance
                cov_chol = np.linalg.cholesky(self.cov_[k])
                # Precision Cholesky is inverse of covariance Cholesky (transposed)
                # If Σ = L L^T, then Σ^(-1) = (L^(-1))^T L^(-1)
                # So precision_chol = L^(-1)^T
                self.precisions_chol_[k] = np.linalg.solve(cov_chol, np.eye(n_features)).T
            except np.linalg.LinAlgError:
                # If covariance is singular, use pseudoinverse
                self.precisions_chol_[k] = np.linalg.pinv(self.cov_[k])

        # Precompute log determinants for efficiency
        self.log_det_chol_ = np.array([
            np.sum(np.log(np.abs(np.diag(self.precisions_chol_[k]))))
            for k in range(self.n_components)
        ])


    def _log_multivariate_normal_density(self, X):
        """
        Compute log N(X | μ_k, Σ_k) for all components using the log-sum-exp trick.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)

        Returns
        -------
        log_prob : array, shape (n_samples, n_components)
            log N(x_i | μ_k, Σ_k) for each sample i and component k
        """
        n_samples, n_features = X.shape

        # X - μ_k for all k: shape (n_samples, n_components, n_features)
        diff = X[:, None, :] - self.mu_[None, :, :]

        # Compute (X - μ_k)^T Σ_k^(-1) (X - μ_k) efficiently
        # Using Y = (X - μ_k) @ precision_chol, then ||Y||^2
        # diff: (n_samples, n_components, n_features)
        # precisions_chol_: (n_components, n_features, n_features)
        Y = np.einsum('nkd,kde->nke', diff, self.precisions_chol_)

        # Mahalanobis distance squared: ||Y||^2 for each sample and component
        maha_sq = np.einsum('nke,nke->nk', Y, Y)

        # log N(x | μ, Σ) = -0.5 * [d*log(2π) + log|Σ| + (x-μ)^T Σ^(-1) (x-μ)]
        # log|Σ| = -2 * log|precision_chol|
        log_prob = -0.5 * (n_features * np.log(2 * np.pi) + maha_sq)
        log_prob += self.log_det_chol_[None, :]  # Add log determinant term

        return log_prob

    def _e_step(self, X):
        """
        E-step: Compute responsibilities using log-sum-exp trick (vectorized).

        Returns
        -------
        responsibilities : array, shape (n_samples, n_components)
            Posterior probabilities that each point belongs to each component
        """
        n_samples = X.shape[0]

        # Compute log probabilities: log N(x_i | μ_k, Σ_k)
        # Shape: (n_samples, n_components)
        log_prob = self._log_multivariate_normal_density(X)

        # Add log mixing proportions: log π_k + log N(x_i | μ_k, Σ_k)
        # Use small epsilon to prevent log(0)
        log_weighted = log_prob + np.log(self.pi_ + 1e-300)[None, :]

        # Use log-sum-exp for numerical stability
        # log_sum_i = log(Σ_k exp(log_weighted[i,k]))
        log_sum = logsumexp(log_weighted, axis=1, keepdims=True)

        # Responsibilities: exp(log_weighted - log_sum)
        # This is equivalent to: π_k * N(x_i|μ_k,Σ_k) / Σ_j π_j * N(x_i|μ_j,Σ_j)
        log_responsibilities = log_weighted - log_sum
        responsibilities = np.exp(log_responsibilities)

        # Ensure responsibilities are valid probabilities (handle numerical issues)
        responsibilities = np.clip(responsibilities, 0, 1)
        row_sums = responsibilities.sum(axis=1, keepdims=True)
        mask = row_sums.flatten() == 0
        responsibilities = np.where(
            mask[:, np.newaxis],
            1.0 / self.n_components,
            responsibilities / row_sums
        )

        return responsibilities


    def _m_step(self, X, R):
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        K = self.n_components

        Nk = R.sum(axis=0)  # (K,)
        Nk_safe = np.maximum(Nk, 1e-12)

        self.pi_ = Nk / n  # (K,)
        self.mu_ = (R.T @ X) / Nk_safe[:, None]  # (K,d)

        # Σ_k = (∑ r_ik (x_i - μ_k)(x_i - μ_k)^T) / N_k
        diff = X[:, None, :] - self.mu_[None, :, :]  # (n,K,d)
        w = R[:, :, None]  # (n,K,1)
        cov = np.einsum('nkd,nke->kde', w * diff, diff) / Nk_safe[:, None, None]  # (K,d,d)

        # symmetrize + ridge for SPD
        cov = 0.5 * (cov + np.transpose(cov, (0, 2, 1)))
        cov += self.reg_ * np.eye(d)[None, :, :]

        self.cov_ = cov
        self._compute_precisions_cholesky()  # refresh caches

    def _compute_log_likelihood(self, X):
        """
        Compute the log-likelihood of the data using log-sum-exp trick (vectorized).

        Returns
        -------
        log_likelihood : float
        """
        # Compute log probabilities: log N(x_i | μ_k, Σ_k)
        log_prob = self._log_multivariate_normal_density(X)

        # Add log mixing proportions: log π_k + log N(x_i | μ_k, Σ_k)
        log_weighted = log_prob + np.log(self.pi_ + 1e-300)[None, :]

        # Use log-sum-exp to compute log(Σ_k π_k * N(x_i | μ_k, Σ_k))
        # This gives us log p(x_i) for each sample i
        log_point_likelihoods = logsumexp(log_weighted, axis=1)

        # Sum over all samples to get total log-likelihood
        log_likelihood = log_point_likelihoods.sum()

        return log_likelihood

    def predict_proba(self, X):
        """
        Predict posterior probabilities for each component.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        responsibilities : array, shape (n_samples, n_components)
            Posterior probabilities
        """
        X = np.asarray(X)
        return self._e_step(X)

    def predict(self, X):
        """
        Predict the component labels for each point.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        labels : array, shape (n_samples,)
            Component labels (argmax of responsibilities)
        """
        responsibilities = self.predict_proba(X)
        return np.argmax(responsibilities, axis=1)

    def score(self, X):
        """
        Compute the log-likelihood of X under the fitted model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        log_likelihood : float
        """
        X = np.asarray(X)
        return self._compute_log_likelihood(X)


    def sample(self, n_samples=1, random_state=None):
        """
            Generate random samples from the fitted GMM.

            Parameters
            ----------
            n_samples : int, default=1
                Number of samples to generate
            random_state : int or None, default=None
                Random seed for reproducibility

            Returns
            -------
            X : array, shape (n_samples, n_features)
                Generated samples
            labels : array, shape (n_samples,)
                Component labels for each sample
        """
        if self.mu_ is None:
            raise ValueError("Model must be fitted before sampling.")
        rng = self.rng if random_state is None else np.random.default_rng(random_state)

        K, d = self.mu_.shape
        labels = rng.choice(K, size=n_samples, p=(self.pi_ / self.pi_.sum()))
        X = np.empty((n_samples, d), dtype=float)

        # Cholesky of covariances (not precision)
        Ls = []
        for k in range(K):
            S = self.cov_[k].copy()
            S.flat[::d + 1] += self.reg_
            Ls.append(cholesky(S, lower=True, check_finite=False))

        for k in range(K):
            idx = np.where(labels == k)[0]
            if idx.size == 0:
                continue
            Z = rng.standard_normal((idx.size, d))
            X[idx] = Z @ Ls[k].T + self.mu_[k]
        return X, labels

    def plot_2D_model(self, ax, colors, alpha=0.3):
        """
        Visualize the model parameters using ellipses.
        """
        for k in range(self.n_components):
            ax.plot(self.mu_[k][0], self.mu_[k][1], 'x', color=colors[k])
            _plot_gaussian_ellipse(self.mu_[k], self.cov_[k], ax, color=colors[k], alpha=alpha)