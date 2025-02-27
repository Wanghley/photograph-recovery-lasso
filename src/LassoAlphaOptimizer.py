import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RepeatedKFold
from joblib import Parallel, delayed
from blockfy import Blockfy
from basisgen import BasisGenerator
from tqdm import tqdm

class LassoAlphaOptimizer:
    """A class for optimizing alpha parameters for Lasso regression on image blocks.

    This class provides functionality to:
    1. Optimize alpha parameters for Lasso regression on corrupted image blocks
    2. Return the best alpha parameters for each block

    Attributes:
        S (int): Number of sensed pixels per block
        M (int): Number of training samples
        test_train_split (float): Ratio of test to training data (e.g., 0.2 for 80-20 split)
        P (int): Height of the basis chips
        Q (int): Width of the basis chips
        basis_generator (BasisGenerator): Instance of BasisGenerator for generating basis matrices
        blockifier (Blockfy): Instance of Blockfy for processing image blocks
        alpha_values (np.ndarray): Precomputed array of alpha values for Lasso regression
    """

    def __init__(self, S: int, M: int, test_train_split: float = 0.2, P: int = 8, Q: int = 8):
        """Initialize the LassoAlphaOptimizer instance.

        Args:
            S (int): Number of sensed pixels per block
            M (int): Number of training samples
            test_train_split (float): Ratio of test to training data (default: 0.2)
            P (int): Height of the basis chips (default: 8)
            Q (int): Width of the basis chips (default: 8)
        """
        self.S = S
        self.M = M
        self.test_train_split = test_train_split
        self.P = P
        self.Q = Q
        self.basis_generator = BasisGenerator(P, Q, uv_orientation=True)
        self.blockifier = None
        self.alpha_values = np.logspace(-8, 8, 30)  # Precompute alpha values

    def _process_block(self, corrupted_block, basis_matrix, alpha_values, rkf):
        """Process a single block to find the best alpha value.

        Args:
            corrupted_block (np.ndarray): The corrupted image block.
            basis_matrix (np.ndarray): The basis matrix.
            alpha_values (np.ndarray): Array of alpha values to test.
            rkf (RepeatedKFold): RepeatedKFold instance for cross-validation.

        Returns:
            float: The best alpha value for the block.
        """
        chip_flat = corrupted_block.flatten()
        good_pixels_idx = np.where(~np.isnan(chip_flat))[0]
        chip_flat = chip_flat[good_pixels_idx]
        bm_chip = basis_matrix[good_pixels_idx]

        mse_values = np.full(len(alpha_values), np.inf)  # Preallocate with high values

        for j, a in enumerate(alpha_values):
            mse_fold = []
            for train_index, test_index in rkf.split(bm_chip):
                lasso = Lasso(alpha=a, fit_intercept=False, max_iter=5000, tol=1e-4)
                lasso.fit(bm_chip[train_index], chip_flat[train_index])

                recov_chip = np.dot(basis_matrix, lasso.coef_).reshape(self.P, self.Q)

                # Only compare valid pixels
                test_pixels = good_pixels_idx[test_index]
                mse = mean_squared_error(corrupted_block.flat[test_pixels], recov_chip.flat[test_pixels])
                mse_fold.append(mse)

            mse_values[j] = np.nanmean(mse_fold)

        return alpha_values[np.nanargmin(mse_values)]

    def optimize_alphas(self, img_path: str) -> np.ndarray:
        """Optimize alpha parameters for Lasso regression on corrupted image blocks.

        Args:
            img_path (str): Path to the input image file.

        Returns:
            np.ndarray: Array of best alpha parameters for each block.
        """
        self.blockifier = Blockfy(img_path, (self.P, self.Q), self.P)
        self.blockifier.generate_blocks()
        corrupted_blocks = self.blockifier.generate_corrupted_blocks(sensed_pixels=self.S)
        basis_matrix = self.basis_generator.generate_basis_matrix()
        rkf = RepeatedKFold(n_splits=max(2, int(self.S * self.test_train_split)), n_repeats=self.M)

        best_alphas = Parallel(n_jobs=min(len(corrupted_blocks), -1))(
            delayed(self._process_block)(corrupted_blocks[i], basis_matrix, self.alpha_values, rkf)
            for i in tqdm(range(len(corrupted_blocks)), desc="Optimizing alphas", unit="blocks")
        )

        return np.array(best_alphas)
