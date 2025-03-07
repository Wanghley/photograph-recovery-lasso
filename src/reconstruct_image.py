import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from tqdm import tqdm
import warnings

# Add path to src to sys
sys.path.append(os.path.join(os.getcwd(), '../src'))

from blockfy import Blockfy
from basisgen import BasisGenerator
from lassoalphaoptimizer import LassoAlphaOptimizer

class ImageReconstructor:
    def __init__(self, P, Q, S, M, train_test_ratio, img_path=None, corrupted_img=None):
        self.img_path = img_path
        self.corrupted_img = corrupted_img
        self.P = P
        self.Q = Q
        self.S = S
        self.M = M
        self.train_test_ratio = train_test_ratio

    def reconstruct_image(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Initialize Blockfy
            if self.img_path:
                blockfy = Blockfy(self.img_path, (self.P, self.Q), self.P)
            else:
                blockfy = Blockfy(self.corrupted_img, (self.P, self.Q), self.P)

            blockfy.generate_blocks()
            corrpt_blocks = self.corrupted_img if self.corrupted_img else blockfy.generate_corrupted_blocks(self.S)
            original_blocks = blockfy.get_blocks()
            blocks_positioning = blockfy.get_blocks_positioning()

            # Optimize alphas
            lassoalphaoptimizer = LassoAlphaOptimizer(self.S, self.M, self.train_test_ratio, self.P, self.Q)
            if self.img_path:
                alphas = lassoalphaoptimizer.optimize_alphas(self.img_path)
            else:
                alphas = lassoalphaoptimizer.optimize_alphas(self.corrupted_img, is_corrupted=True)

            # Generate basis matrix
            basisgen = BasisGenerator(self.P, self.Q)
            basis_matrix = basisgen.generate_basis_matrix()

            # Reconstruct blocks
            recovered_blocks = []
            for i in tqdm(range(min(len(corrpt_blocks), len(alphas)))):
                block = corrpt_blocks[i]
                alpha = alphas[i]
                blk_flat = block.flatten()
                good_idx = np.where(~np.isnan(blk_flat))[0].astype(int)
                corrpt_idx = np.isnan(blk_flat)
                X = basis_matrix[good_idx, :]
                y = blk_flat[good_idx]
                lasso = Lasso(alpha=alpha, fit_intercept=False)
                lasso.fit(X, y)
                recov_chip = np.zeros((self.P, self.Q))
                full_block = corrpt_blocks[i].copy()
                block_flat = full_block.flatten()
                nan_indices = np.where(np.isnan(block_flat))[0]
                X_missing = basis_matrix[nan_indices]
                y_missing = lasso.predict(X_missing)
                block_flat[nan_indices] = y_missing
                recov_chip = block_flat.reshape(self.P, self.Q)
                recovered_blocks.append(recov_chip)

            # Convert recovered blocks to numpy array
            recovered_blocks_array = np.array(recovered_blocks)

            # Reconstruct full images
            original = None if self.corrupted_img else blockfy.block_to_image(original_blocks)
            corrupted = self.corrupted_img if self.corrupted_img else blockfy.block_to_image(corrpt_blocks)
            recovered_img = blockfy.block_to_image(recovered_blocks_array)

            # save images to class
            self.original_img = original
            self.corrupted_img = corrupted
            self.recovered_img = recovered_img

    def plot_images(self):
        plt.figure(figsize=(30, 20))
        cmap = plt.cm.gray
        cmap.set_bad(color='red')
        if self.corrupted_img is not None:
            plt.subplot(1, 3, 1)
            plt.imshow(self.corrupted_img, cmap=cmap)
            plt.title('Corrupted Image')
            plt.axis('off')

        if self.recovered_img is not None:
            plt.subplot(1, 3, 2)
            plt.imshow(self.recovered_img, cmap='gray')
            plt.title('Recovered Image')
            plt.axis('off')

        if self.original_img is not None:
            plt.subplot(1, 3, 3)
            plt.imshow(self.original_img, cmap='gray')
            plt.title('Original Image')
            plt.axis('off')

        plt.tight_layout()
        plt.show()

    def get_images(self):
        return self.original_img, self.corrupted_img, self.recovered_img

# if __name__ == "__main__":
#     img_path = '../assets/nature/nature.bmp'
#     P, Q = 8, 8
#     S = 30
#     M = 20
#     train_test_ratio = 0.2
#     reconstructor = ImageReconstructor(P, Q, S, M, train_test_ratio, img_path=img_path)
#     reconstructor.reconstruct_image()
#     reconstructor.plot_images()
