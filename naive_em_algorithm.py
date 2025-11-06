import numpy as np
from scipy.stats import multivariate_normal
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
plt.rcParams.update({
    'lines.linewidth' : 1.,
    'lines.markersize' : 5,
    'font.size': 9,
    "text.usetex": True,
    'font.family': 'serif', 
    'font.serif': ['Computer Modern'],
    'text.latex.preamble' : r'\usepackage{amsmath,amsfonts}',
    'axes.linewidth' : .75})

def _plot_gaussian_ellipse(mu, cov, ax, color, alpha=0.3, label=None):
    """Plot an ellipse representing a 2D Gaussian"""
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2 * np.sqrt(eigenvalues) * 2  # 2 standard deviations
    
    ellipse = Ellipse(mu, width, height, angle=angle, 
                     facecolor=color, alpha=alpha, edgecolor=color, linewidth=2, label=label)
    ax.add_patch(ellipse)
        
class GaussianMixtureModel:
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
        self.mu_ = mus
        self.cov_ = covs
        self.pi_ = pi
        
    
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
                [X[i].copy() for i in indices],
                [np.eye(n_features) for _ in range(self.n_components)],
                np.ones(self.n_components) / self.n_components
            )
    
    def _e_step(self, X):
        """
        E-step: Compute responsibilities.
        
        Returns
        -------
        responsibilities : array, shape (n_samples, n_components)
            Posterior probabilities that each point belongs to each component
        """
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.n_components))
        
        # For each data point
        for i in range(n_samples):
            # For each component
            for k in range(self.n_components):
                # Compute π_k * N(x_i | μ_k, Σ_k)
                try:
                    pdf_value = multivariate_normal.pdf(X[i], mean=self.mu_[k], cov=self.cov_[k])
                    responsibilities[i, k] = self.pi_[k] * pdf_value
                except np.linalg.LinAlgError:
                    # Covariance became singular
                    responsibilities[i, k] = 0.0
        
        # Normalize each row to sum to 1
        for i in range(n_samples):
            row_sum = responsibilities[i, :].sum()
            if row_sum > 0:
                responsibilities[i, :] /= row_sum
            else:
                # If all responsibilities are 0, assign uniformly
                responsibilities[i, :] = 1.0 / self.n_components
        
        return responsibilities
    
    def _m_step(self, X, responsibilities):
        """
        M-step: Update parameters using responsibilities.
        
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
        responsibilities : array, shape (n_samples, n_components)
        """
        n_samples, n_features = X.shape
        
        # Initialize new parameters
        mu_new = []
        cov_new = []
        pi_new = np.zeros(self.n_components)
        
        # For each component
        for k in range(self.n_components):
            # Compute N_k = sum of responsibilities for component k
            N_k = 0.0
            for i in range(n_samples):
                N_k += responsibilities[i, k]
            
            # Update mixing proportion
            pi_new[k] = N_k / n_samples
            
            # Update mean
            mu_k = np.zeros(n_features)
            for i in range(n_samples):
                mu_k += responsibilities[i, k] * X[i]
            mu_k /= N_k
            mu_new.append(mu_k)
            
            # Update covariance
            cov_k = np.zeros((n_features, n_features))
            for i in range(n_samples):
                diff = X[i] - mu_k
                cov_k += responsibilities[i, k] * np.outer(diff, diff)
            cov_k /= N_k
            
            # Add small regularization to prevent singularity
            cov_k += 1e-6 * np.eye(n_features)
            cov_new.append(cov_k)
        
        # Update stored parameters
        self._set_parameters(
            mu_new,
            cov_new,
            pi_new
        )
    
    def _compute_log_likelihood(self, X):
        """
        Compute the log-likelihood of the data.
        
        Returns
        -------
        log_likelihood : float
        """
        n_samples = X.shape[0]
        log_likelihood = 0.0
        
        for i in range(n_samples):
            # Compute p(x_i) = Σ_k π_k * N(x_i | μ_k, Σ_k)
            point_likelihood = 0.0
            for k in range(self.n_components):
                try:
                    pdf_value = multivariate_normal.pdf(X[i], mean=self.mu_[k], cov=self.cov_[k])
                    point_likelihood += self.pi_[k] * pdf_value
                except np.linalg.LinAlgError:
                    pass
            
            # Add log(p(x_i)) to total
            log_likelihood += np.log(point_likelihood)
        
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
            raise ValueError("Model must be fitted before sampling")
        
        if random_state is not None:
            np.random.seed(random_state)
        
        n_features = len(self.mu_[0])
        
        # Allocate output arrays
        X = np.zeros((n_samples, n_features))
        labels = np.zeros(n_samples, dtype=int)
        
        # Sample component assignments from the mixing proportions
        for i in range(n_samples):
            # Sample which component to use
            k = np.random.choice(self.n_components, p=self.pi_)
            labels[i] = k
            
            # Sample from that component
            X[i] = np.random.multivariate_normal(self.mu_[k], self.cov_[k])
        
        return X, labels
    
    def plot_2D_model(self, ax, colors, alpha=0.3):
        """
        Visualize the model parameters using ellipses.
        """
        for k in range(self.n_components):
            ax.plot(self.mu_[k][0], self.mu_[k][1], 'x', color=colors[k])
            _plot_gaussian_ellipse(self.mu_[k], self.cov_[k], ax, color=colors[k], alpha=alpha)
