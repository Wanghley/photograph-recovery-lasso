import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import RepeatedKFold
from joblib import Parallel, delayed
from tqdm.notebook import tqdm
import warnings

# Add path to src to sys
sys.path.append(os.path.join(os.getcwd(), '../src'))

from blockfy import Blockfy
from basisgen import BasisGenerator

# ignore warnings
warnings.filterwarnings("ignore")

class LassoAlphaOptimizer:
    def __init__(self, S, M, train_test_ratio, P, Q):
        self.S = S
        self.M = M
        self.train_test_ratio = train_test_ratio
        self.P = P
        self.Q = Q
        self.basis_generator = BasisGenerator(P, Q)
        
        # Define the range of alpha values to test
        self.alpha_values = np.logspace(-10, 1, 20)
        
        # Calculate m (number of splits) based on train_test_ratio
        self.m = int(1 / self.train_test_ratio)
    
    def _process_block(self, block, basis_matrix, alpha_values):
        """
        Process a single block to find the optimal alpha value.
        
        Parameters:
        -----------
        block : numpy.ndarray
            The image block to process
        basis_matrix : numpy.ndarray
            The basis matrix for the block
        alpha_values : numpy.ndarray
            The alpha values to test
            
        Returns:
        --------
        float
            The best alpha value for the block
        """
        # Get the flattened block and identify non-NaN indices
        blk_flat = block.flatten()
        good_idx = np.where(~np.isnan(blk_flat))[0].astype(int)
        
        # If block has too few non-NaN values, return default alpha
        if len(good_idx) < 5:  # Need at least 5 samples for meaningful CV
            return alpha_values[len(alpha_values) // 2]  # Return middle value as default
        
        # Extract the valid values from the block
        X = basis_matrix[good_idx, :]
        y = blk_flat[good_idx]
        
        # Determine appropriate number of splits for cross-validation
        # Must be at least 2 and less than number of samples
        n_splits = min(5, max(2, len(good_idx) // 2))
        n_repeats = max(1, self.M // n_splits)  # Adjust repeats to maintain similar total iterations
        
        # Create K-fold cross-validator
        rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats)
        
        # Initialize variables to track the best alpha
        min_mse = float('inf')
        best_alpha = None
        
        # Test each alpha value using cross-validation
        for alpha in alpha_values:
            mse_values = []
            try:
                for train_index, test_index in rkf.split(X):
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    
                    # Fit Lasso model with current alpha
                    lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=10000)
                    lasso.fit(X_train, y_train)
                    
                    # Evaluate on test set
                    y_pred = lasso.predict(X_test)
                    mse = np.mean((y_test - y_pred)**2)
                    mse_values.append(mse)
                
                # Calculate mean MSE across all folds
                mean_mse = np.mean(mse_values)
                
                # Update best alpha if this one is better
                if mean_mse < min_mse:
                    min_mse = mean_mse
                    best_alpha = alpha
            except Exception as e:
                # If an error occurs, just continue with next alpha
                continue
        
        # If no valid alpha was found, return default
        if best_alpha is None:
            return alpha_values[len(alpha_values) // 2]
            
        return best_alpha

    def optimize_alphas(self, img_input, is_corrupted=False):
        """
        Optimize alpha values for Lasso regression on each image block.
        
        Parameters:
        -----------
        img_input : str or numpy.ndarray
            Path to the image file or corrupted image array
        is_corrupted : bool
            Whether the input is already a corrupted image array
            
        Returns:
        --------
        numpy.ndarray
            Array of optimized alpha values for each block
        """
        # Initialize Blockfy
        blockfy = Blockfy(img_input, (self.P, self.Q), self.P)
        blockfy.generate_blocks()
        
        # Get corrupted blocks
        if is_corrupted:
            corrupted_blocks = blockfy.get_blocks()
        else:
            corrupted_blocks = blockfy.generate_corrupted_blocks(self.S)
        
        # Generate basis matrix
        basis_matrix = self.basis_generator.generate_basis_matrix()
        
        # Process blocks in parallel to find optimal alphas
        best_alphas = Parallel(n_jobs=min(len(corrupted_blocks), os.cpu_count()))(
            delayed(self._process_block)(corrupted_blocks[i], basis_matrix, self.alpha_values)
            for i in tqdm(range(len(corrupted_blocks)), desc="Optimizing alphas", unit="blocks")
        )
        
        return np.array(best_alphas)