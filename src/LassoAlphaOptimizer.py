import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RepeatedKFold
from joblib import Parallel, delayed  # For parallel processing
from Blockfy import BlockifyImage # For generating image blocks and corrupted blocks
from BasisGenerator import BasisGenerator # For generating the basis matrix


class LassoAlphaOptimizer:
    """A class for optimizing alpha parameters for Lasso regression on image blocks.
    
    This class finds the optimal regularization parameter (alpha) for each block
    in an image using cross-validation. The optimization is performed by splitting
    the sensed pixels into training and testing sets multiple times and selecting 
    the alpha that minimizes the mean squared error on the test sets.
    
    Attributes:
        S (int): Number of sensed pixels in each block.
        M (int): Number of cross-validation repeats.
        test_train_split (float): Fraction of sensed pixels to use for testing.
        P (int): Block height.
        Q (int): Block width.
        basis_generator (BasisGenerator): Generator for the basis matrix.
        blockifier (BlockifyImage): Handler for image block operations.
    """

    def __init__(self, S: int, M: int, test_train_split: float = 0.2, P: int = 8, Q: int = 8):
        """Initialize the LassoAlphaOptimizer instance.
        
        Args:
            S (int): Number of sensed pixels in each block.
            M (int): Number of cross-validation repeats.
            test_train_split (float, optional): Fraction of sensed pixels to use for testing.
                Defaults to 0.2.
            P (int, optional): Block height. Defaults to 8.
            Q (int, optional): Block width. Defaults to 8.
        """
        self.S = S
        self.M = M
        self.test_train_split = test_train_split
        self.P = P
        self.Q = Q
        self.basis_generator = BasisGenerator(P, Q, uv_orientation=True)
        self.blockifier = None

    def _process_block(self, corrupted_block, basis_matrix, alpha_values, rkf):
        """Process a single block to find the best alpha parameter.
        
        This method performs cross-validation on a corrupted block to determine
        the optimal alpha value that minimizes reconstruction error.
        
        Args:
            corrupted_block (np.ndarray): A block with missing (NaN) values.
            basis_matrix (np.ndarray): The basis matrix for reconstruction.
            alpha_values (np.ndarray): Array of alpha values to test.
            rkf (RepeatedKFold): Cross-validation splitter.
            
        Returns:
            float: The optimal alpha value for the given block.
        """
        chip_flat = corrupted_block.flatten()
        good_pixels_idx = np.where(~np.isnan(chip_flat))[0].astype(int)
        chip_flat = chip_flat[good_pixels_idx]
        bm_chip = basis_matrix[good_pixels_idx]

        mse_values = np.zeros(len(alpha_values))

        for j, a in enumerate(alpha_values):
            mse_fold = []
            for train_index, test_index in rkf.split(bm_chip):
                lasso = Lasso(alpha=a, fit_intercept=False, max_iter=5000, tol=1e-4)
                lasso.fit(bm_chip[train_index], chip_flat[train_index])

                # Reconstruct the chip
                recov_chip = np.dot(basis_matrix, lasso.coef_).reshape(self.P, self.Q)

                # Mask the reconstructed and corrupted chips
                masked_corrupt_chip = np.full_like(corrupted_block, np.nan)
                masked_corrupt_chip.flat[good_pixels_idx[test_index]] = corrupted_block.flat[good_pixels_idx[test_index]]

                masked_reconstruct_chip = np.full_like(corrupted_block, np.nan)
                masked_reconstruct_chip.flat[good_pixels_idx[test_index]] = recov_chip.flat[good_pixels_idx[test_index]]

                # Calculate MSE
                valid_pixels = ~np.isnan(masked_corrupt_chip) & ~np.isnan(masked_reconstruct_chip)
                if np.any(valid_pixels):
                    mse = mean_squared_error(
                        masked_corrupt_chip[valid_pixels], masked_reconstruct_chip[valid_pixels]
                    )
                    mse_fold.append(mse)
                else:
                    mse_fold.append(np.nan)

            # Average MSE across folds
            mse_values[j] = np.nanmean(mse_fold)

        # Find the best alpha
        min_mse_idx = np.nanargmin(mse_values)
        return alpha_values[min_mse_idx]

    def optimize_alphas(self, img_path: str) -> np.ndarray:
        """Optimize alpha parameters for Lasso regression on corrupted image blocks.
        
        This method performs the following steps:
        1. Divides the input image into blocks
        2. Creates corrupted versions of these blocks by randomly selecting pixels
        3. For each block, finds the optimal alpha parameter using cross-validation
        4. Returns an array of optimal alpha values, one for each block
        
        Args:
            img_path (str): Path to the input image file.
            
        Returns:
            np.ndarray: Array of optimal alpha values, one for each block in the image.
                The shape corresponds to the number of blocks (P*Q).
        """
        # Initialize BlockifyImage
        self.blockifier = BlockifyImage(img_path, (self.P, self.Q), self.P)
        self.blockifier.generate_blocks()
        corrupted_blocks = self.blockifier.generate_corrupted_blocks(sensed_pixels=self.S)

        # Generate basis matrix
        basis_matrix = self.basis_generator.generate_basis_matrix()

        # Initialize RepeatedKFold
        m = int(self.S * self.test_train_split)  # Number of test samples
        rkf = RepeatedKFold(n_splits=m, n_repeats=self.M)

        # Precompute alpha values
        alpha_values = np.logspace(-6, 6, 20)

        # Process blocks in parallel
        best_alphas = Parallel(n_jobs=-1)(
            delayed(self._process_block)(corrupted_blocks[i], basis_matrix, alpha_values, rkf)
            for i in range(self.P * self.Q)
        )

        return np.array(best_alphas)