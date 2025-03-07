import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import random
from skimage.util import view_as_windows
from typing import Tuple, Optional, Union


class Blockfy:
    """A class for processing images by dividing them into blocks and corrupting pixels.

    This class provides functionality to:
    1. Load and divide an image into blocks of specified size
    2. Generate corrupted versions of these blocks by randomly setting pixels to NaN
    3. Save and reconstruct the processed blocks

    Attributes:
        img (numpy.ndarray): The loaded image array
        block_shape (tuple): The shape of blocks (height, width)
        height (int): Height of the original image
        width (int): Width of the original image
        block_step (int): The step size between blocks
        img_blocks (numpy.ndarray): Generated image blocks
        corrupted_blocks (numpy.ndarray): Generated corrupted blocks
    """

    def __init__(self, img_source: Union[str, np.ndarray], block_shape: Tuple[int, int], block_step: int):
        """Initialize the Blockfy instance.

        Args:
            img_source (Union[str, np.ndarray]): Either a path to the input image file or a pre-loaded numpy array
            block_shape (tuple): Shape of blocks to generate (height, width)
            block_step (int): Step size between blocks (stride)
        """
        if isinstance(img_source, str):
            self.img = mpimg.imread(img_source)
        elif isinstance(img_source, np.ndarray):
            self.img = img_source
        else:
            raise TypeError("img_source must be either a string path or a numpy array")
            
        self.block_shape = block_shape
        print(self.img.shape)
        self.height, self.width = self.img.shape[:2]
        self.block_step = block_step
        self.img_blocks: Optional[np.ndarray] = None
        self.corrupted_blocks: Optional[np.ndarray] = None
        

    def generate_blocks(self) -> np.ndarray:
        """Divide the image into blocks of the specified shape and step size.

        Returns:
            numpy.ndarray: Array of image blocks with shape (num_blocks_y, num_blocks_x, block_height, block_width)
        """
        self.img_blocks = view_as_windows(self.img, self.block_shape, self.block_step)
        return self.img_blocks

    def get_blocks(self) -> np.ndarray:
        """Get the generated image blocks.

        Returns:
            numpy.ndarray: Array of image blocks, reshaped to flatten the spatial dimensions.
        """
        if self.img_blocks is None:
            raise ValueError("Blocks have not been generated yet. Call `generate_blocks()` first.")
        
        # Reshape blocks from (X, Y, block_height, block_width) to (X*Y, block_height, block_width)
        num_blocks_y, num_blocks_x = self.img_blocks.shape[:2]
        return self.img_blocks.reshape(num_blocks_y * num_blocks_x, self.block_shape[0], self.block_shape[1])
    
    def get_blocks_positioning(self) -> np.ndarray:
        """Get the positioning of the generated image blocks.

        Returns:
            numpy.ndarray: Shape of the image blocks array (num_blocks_y, num_blocks_x).
        """
        return self.img_blocks.shape[:2]

    def generate_corrupted_blocks(self, sensed_pixels: int) -> np.ndarray:
        """Generate corrupted versions of the image blocks by randomly setting pixels to NaN.

        Args:
            sensed_pixels (int): Number of pixels to keep in each block.

        Returns:
            numpy.ndarray: Array of corrupted blocks with shape (num_blocks_y * num_blocks_x, block_height, block_width)
        """
        if self.img_blocks is None:
            raise ValueError("Blocks have not been generated yet. Call `generate_blocks()` first.")

        corrupted_blocks = []
        for i in range(self.img_blocks.shape[0]):
            for j in range(self.img_blocks.shape[1]):
                corrupted_block = self._generate_sensed_pixels_in_block(self.img_blocks[i, j], sensed_pixels)
                corrupted_blocks.append(corrupted_block)

        self.corrupted_blocks = np.array(corrupted_blocks)
        return self.corrupted_blocks

    def save_corrupted_blocks(self, path: str) -> str:
        """Save the corrupted blocks to a file.

        Args:
            path (str): Path to save the corrupted blocks.

        Returns:
            str: Path where the file was saved.
        """
        if self.corrupted_blocks is None:
            raise ValueError("Corrupted blocks have not been generated yet. Call `generate_corrupted_blocks()` first.")
        np.save(path, self.corrupted_blocks)
        return path

    def block_to_image(self, blocks: np.ndarray) -> np.ndarray:
        """Reconstruct an image from the given blocks.

        Args:
            blocks (numpy.ndarray): Array of blocks to reconstruct into an image.

        Returns:
            numpy.ndarray: Reconstructed image.
        """
        reconstructed_image = np.full_like(self.img, np.nan, dtype=np.float32)
        num_blocks_y = (self.height - self.block_shape[0]) // self.block_step + 1
        num_blocks_x = (self.width - self.block_shape[1]) // self.block_step + 1

        blocks = blocks.reshape(num_blocks_y, num_blocks_x, self.block_shape[0], self.block_shape[1])

        for i in range(num_blocks_y):
            for j in range(num_blocks_x):
                y_start = i * self.block_step
                y_end = y_start + self.block_shape[0]
                x_start = j * self.block_step
                x_end = x_start + self.block_shape[1]
                reconstructed_image[y_start:y_end, x_start:x_end] = blocks[i, j]

        return reconstructed_image

    def save_blocks(self, path: str, img_name: str, sensed_pixels: int) -> None:
        """Save the original and corrupted blocks to a specified path.

        Args:
            path (str): Base directory to save the blocks.
            img_name (str): Name of the image (used for naming files).
            sensed_pixels (int): Number of sensed pixels (used for naming files).
        """
        if self.img_blocks is None or self.corrupted_blocks is None:
            raise ValueError("Blocks or corrupted blocks have not been generated yet.")

        path_joined = os.path.join(path, f"{img_name}_blocks_{sensed_pixels}")
        os.makedirs(path_joined, exist_ok=True)

        np.save(os.path.join(path_joined, f"{img_name}_blocks.npy"), self.img_blocks)
        np.save(os.path.join(path_joined, f"{img_name}_corrupted_blocks.npy"), self.corrupted_blocks)

    def _generate_sensed_pixels_in_block(self, img_block: np.ndarray, sensed_pixels: int) -> np.ndarray:
        """Randomly select a subset of pixels in a block and set their values to NaN.

        Args:
            img_block (numpy.ndarray): A 2D array representing the image block.
            sensed_pixels (int): The number of pixels to keep.

        Returns:
            numpy.ndarray: The modified image block with sensed pixels set to NaN.
        """
        block_height, block_width = img_block.shape
        total_pixels = block_height * block_width

        if sensed_pixels > total_pixels:
            raise ValueError("sensed_pixels cannot be greater than the total number of pixels in the block.")

        corrupted_pixels = random.sample(range(total_pixels), total_pixels - sensed_pixels)
        img_block = img_block.astype(float).copy()

        for pixel in corrupted_pixels:
            row, col = divmod(pixel, block_width)
            img_block[row, col] = np.nan

        return img_block

    def save_corrupted_image(self, path: str, img_name: str, sensed_pixels: int, format: str = 'png') -> None:
        """Save the corrupted image to a specified path in png, txt, or npy format.

        Args:
            path (str): Directory to save the image.
            img_name (str): Name of the image (used for naming files).
            sensed_pixels (int): Number of sensed pixels (used for naming files).
            format (str): Format to save the image ('png', 'txt', or 'npy').
        """
        if self.corrupted_blocks is None:
            raise ValueError("Corrupted blocks have not been generated yet.")

        corrupted_image = self.block_to_image(self.corrupted_blocks)
        file_path = os.path.join(path, f"{img_name}_corrupted_{sensed_pixels}_px.{format}")

        if format == 'png':
            cmap = plt.get_cmap('gray')
            cmap.set_bad(color='red')
            plt.imsave(file_path, corrupted_image, cmap=cmap)
        elif format == 'txt':
            np.savetxt(file_path, corrupted_image)
        elif format == 'npy':
            np.save(file_path, corrupted_image)
        else:
            raise ValueError("Invalid format. Please choose from 'png', 'txt', or 'npy'.")