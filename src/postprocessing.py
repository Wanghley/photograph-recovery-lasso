import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
from skimage.restoration import denoise_nl_means, estimate_sigma
import time
import os
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
import warnings

# Ignore all warnings
warnings.filterwarnings('ignore')

class ImagePostProcessor:
    """
    A class for applying post-processing filters to reconstructed images and analyzing results.
    """
    
    def __init__(self, original=None, corrupted=None, reconstructed=None):
        """
        Initialize the ImagePostProcessor with images.
        
        Parameters:
        -----------
        original : numpy.ndarray
            Original uncorrupted image
        corrupted : numpy.ndarray
            Corrupted/noisy image
        reconstructed : numpy.ndarray
            Reconstructed image before post-processing
        """
        self.original = original
        self.corrupted = corrupted
        self.reconstructed = reconstructed
        self.filtered_images = {}
        self.metrics = {}
    
    def load_images_from_npy(self, original_path, corrupted_path, reconstructed_path):
        """
        Load images from .npy files.
        
        Parameters:
        -----------
        original_path : str
            Path to the original image .npy file
        corrupted_path : str
            Path to the corrupted image .npy file
        reconstructed_path : str
            Path to the reconstructed image .npy file
        """
        self.original = np.load(original_path)
        self.corrupted = np.load(corrupted_path)  # Keep as float with NaN values
        self.reconstructed = np.load(reconstructed_path)
        
        # Ensure original and reconstructed are uint8 for proper display and filtering
        self.original = np.clip(self.original, 0, 255).astype(np.uint8)
        self.reconstructed = np.clip(self.reconstructed, 0, 255).astype(np.uint8)
        # Corrupted image stays as float with NaN values
        
        print(f"Loaded images - Original shape: {self.original.shape}, "
              f"Corrupted shape: {self.corrupted.shape}, "
              f"Reconstructed shape: {self.reconstructed.shape}")
    
    def apply_median_filter(self, image=None, kernel_size=3):
        """
        Apply median filter to an image for noise reduction.
        
        Parameters:
        -----------
        image : numpy.ndarray or None
            Input image (if None, uses self.reconstructed)
        kernel_size : int
            Size of the median filter kernel (must be odd)
            
        Returns:
        --------
        numpy.ndarray
            Filtered image
        """
        if image is None:
            if self.reconstructed is None:
                raise ValueError("No reconstructed image available to filter")
            image = self.reconstructed
        
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure kernel size is odd
            
        start_time = time.time()
        
        if len(image.shape) == 2:  # Grayscale image
            filtered_image = cv2.medianBlur(image.astype(np.uint8), kernel_size)
        else:  # Color image
            filtered_image = cv2.medianBlur(image.astype(np.uint8), kernel_size)
            
        elapsed_time = time.time() - start_time
        
        print(f"Median filter applied with kernel size {kernel_size} in {elapsed_time:.4f} seconds")
        
        filter_name = f"median{kernel_size}"
        self.filtered_images[filter_name] = filtered_image
        
        return filtered_image
    
    def apply_gaussian_filter(self, image=None, kernel_size=3, sigma=0):
        """
        Apply Gaussian filter to an image.
        
        Parameters:
        -----------
        image : numpy.ndarray or None
            Input image (if None, uses self.reconstructed)
        kernel_size : int
            Size of the Gaussian kernel (must be odd)
        sigma : float
            Standard deviation of Gaussian kernel (0 = auto)
            
        Returns:
        --------
        numpy.ndarray
            Filtered image
        """
        if image is None:
            if self.reconstructed is None:
                raise ValueError("No reconstructed image available to filter")
            image = self.reconstructed
        
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure kernel size is odd
            
        start_time = time.time()
        
        filtered_image = cv2.GaussianBlur(image.astype(np.uint8), (kernel_size, kernel_size), sigma)
            
        elapsed_time = time.time() - start_time
        
        print(f"Gaussian filter applied with kernel size {kernel_size} in {elapsed_time:.4f} seconds")
        
        filter_name = f"gaussian{kernel_size}"
        self.filtered_images[filter_name] = filtered_image
        
        return filtered_image
    
    def apply_bilateral_filter(self, image=None, d=9, sigma_color=75, sigma_space=75):
        """
        Apply bilateral filter to an image.
        
        Parameters:
        -----------
        image : numpy.ndarray or None
            Input image (if None, uses self.reconstructed)
        d : int
            Diameter of each pixel neighborhood
        sigma_color : float
            Filter sigma in the color space
        sigma_space : float
            Filter sigma in the coordinate space
            
        Returns:
        --------
        numpy.ndarray
            Filtered image
        """
        if image is None:
            if self.reconstructed is None:
                raise ValueError("No reconstructed image available to filter")
            image = self.reconstructed
            
        start_time = time.time()
        
        filtered_image = cv2.bilateralFilter(
            image.astype(np.uint8), d, sigma_color, sigma_space)
            
        elapsed_time = time.time() - start_time
        
        print(f"Bilateral filter applied with d={d}, sigma_color={sigma_color}, "
              f"sigma_space={sigma_space} in {elapsed_time:.4f} seconds")
        
        self.filtered_images["bilateral"] = filtered_image
        
        return filtered_image
    
    def apply_nlm_filter(self, image=None, h_factor=1.15, patch_size=5, patch_distance=3):
        """
        Apply Non-Local Means filter to an image.
        
        Parameters:
        -----------
        image : numpy.ndarray or None
            Input image (if None, uses self.reconstructed)
        h_factor : float
            Factor to multiply the estimated noise standard deviation
        patch_size : int
            Size of patches used for denoising
        patch_distance : int
            Maximum distance to search for similar patches
            
        Returns:
        --------
        numpy.ndarray
            Filtered image
        """
        if image is None:
            if self.reconstructed is None:
                raise ValueError("No reconstructed image available to filter")
            image = self.reconstructed
            
        start_time = time.time()
        
        if len(image.shape) == 3:  # Color image
            sigma_est = np.mean(estimate_sigma(image, channel_axis=2))
            nl_means = denoise_nl_means(
                image, h=h_factor * sigma_est, fast_mode=True,
                patch_size=patch_size, patch_distance=patch_distance, channel_axis=2)
            filtered_image = (nl_means * 255).astype(np.uint8)
        else:  # Grayscale image
            sigma_est = estimate_sigma(image)
            nl_means = denoise_nl_means(
                image, h=h_factor * sigma_est, fast_mode=True,
                patch_size=patch_size, patch_distance=patch_distance)
            filtered_image = (nl_means * 255).astype(np.uint8)
            
        elapsed_time = time.time() - start_time
        
        print(f"Non-Local Means filter applied with h_factor={h_factor}, "
              f"patch_size={patch_size}, patch_distance={patch_distance} "
              f"in {elapsed_time:.4f} seconds")
        
        self.filtered_images["nl_means"] = filtered_image
        
        return filtered_image
    
    def apply_all_filters(self, median_sizes=[3, 5, 7], gaussian_sizes=[3, 7]):
        """
        Apply all available filters to the reconstructed image.
        
        Parameters:
        -----------
        median_sizes : list
            List of kernel sizes for median filters
        gaussian_sizes : list
            List of kernel sizes for Gaussian filters
            
        Returns:
        --------
        dict
            Dictionary of all filtered images
        """
        if self.reconstructed is None:
            raise ValueError("No reconstructed image available to filter")
        
        # Apply median filters
        for size in median_sizes:
            self.apply_median_filter(kernel_size=size)
        
        # Apply Gaussian filters
        for size in gaussian_sizes:
            self.apply_gaussian_filter(kernel_size=size)
        
        # Apply bilateral filter
        self.apply_bilateral_filter()
        
        # Apply Non-Local Means filter
        self.apply_nlm_filter()
        
        return self.filtered_images
    
    def calculate_metrics(self, filtered_image=None, filter_name=None):
        """
        Calculate MSE, PSNR, and SSIM between filtered image and original.
        
        Parameters:
        -----------
        filtered_image : numpy.ndarray or None
            Filtered image (if None, all stored filtered images are evaluated)
        filter_name : str or None
            Name for the filtered image
            
        Returns:
        --------
        dict
            Dictionary containing metrics
        """
        if self.original is None:
            raise ValueError("Original image is required for metric calculation")
        
        if filtered_image is not None and filter_name != 'corrupted':
            # Calculate metrics for a single filtered image (excluding corrupted)
            mse = mean_squared_error(self.original, filtered_image)
            psnr = peak_signal_noise_ratio(self.original, filtered_image)
            
            if len(self.original.shape) == 3:  # Color image
                # For color image, use multichannel SSIM
                ssim = structural_similarity(
                    self.original, filtered_image, channel_axis=2)
            else:
                # For grayscale
                ssim = structural_similarity(self.original, filtered_image)
            
            metrics = {'mse': mse, 'psnr': psnr, 'ssim': ssim}
            
            if filter_name:
                self.metrics[filter_name] = metrics
            
            return metrics
        elif filter_name == 'corrupted':
            # For corrupted image, just return NaN for metrics
            metrics = {'mse': float('nan'), 'psnr': float('nan'), 'ssim': float('nan')}
            self.metrics['corrupted'] = metrics
            return metrics
        else:
            # Calculate metrics for all stored filtered images
            results = {}
            
            # Calculate metrics for the reconstructed image first
            results['reconstructed'] = self.calculate_metrics(
                self.reconstructed, 'reconstructed')
            
            # Add corrupted image with NaN metrics
            results['corrupted'] = self.calculate_metrics(None, 'corrupted')
            
            # Calculate metrics for each filtered image
            for name, img in self.filtered_images.items():
                results[name] = self.calculate_metrics(img, name)
            
            return results
    
    def plot_metrics_comparison(self, save_path=None):
        """
        Plot comparison of metrics for different filters.
        
        Parameters:
        -----------
        save_path : str or None
            Path to save the plot, if None, the plot is displayed but not saved
        """
        if not self.metrics:
            self.calculate_metrics()
        
        # Extract metrics (exclude corrupted image)
        filtered_metrics = {k: v for k, v in self.metrics.items() 
                            if k != 'corrupted' and not np.isnan(v['mse'])}
        
        names = list(filtered_metrics.keys())
        mse_values = [filtered_metrics[name]['mse'] for name in names]
        psnr_values = [filtered_metrics[name]['psnr'] for name in names]
        ssim_values = [filtered_metrics[name]['ssim'] for name in names]
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # MSE plot (lower is better)
        axes[0].bar(names, mse_values, color='crimson')
        axes[0].set_title('Mean Squared Error (Lower is Better)')
        axes[0].set_ylabel('MSE')
        axes[0].set_xticklabels(names, rotation=45, ha='right')
        
        # PSNR plot (higher is better)
        axes[1].bar(names, psnr_values, color='forestgreen')
        axes[1].set_title('Peak Signal-to-Noise Ratio (Higher is Better)')
        axes[1].set_ylabel('PSNR (dB)')
        axes[1].set_xticklabels(names, rotation=45, ha='right')
        
        # SSIM plot (higher is better)
        axes[2].bar(names, ssim_values, color='royalblue')
        axes[2].set_title('Structural Similarity Index (Higher is Better)')
        axes[2].set_ylabel('SSIM')
        axes[2].set_xticklabels(names, rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Metrics comparison plot saved to {save_path}")
        
        plt.show()
    
    def compare_filters(self, save_path=None):
        """
        Display comparison of different filters.
        
        Parameters:
        -----------
        save_path : str or None
            Path to save the comparison image, if None, the image is displayed but not saved
        """
        if not self.filtered_images:
            self.apply_all_filters()
        
        # Determine number of rows and columns
        num_images = len(self.filtered_images) + 3  # +3 for original, corrupted, and reconstructed
        cols = 4
        rows = (num_images + cols - 1) // cols
        
        # Create figure
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
        # define font size to be 12 for all text
        plt.rcParams.update({'font.size': 12})
        axes = axes.flatten()
        
        # Function to display image properly based on color/grayscale
        def show_image(ax, img, title, metrics=None):
            if title == 'Corrupted':
                # Special handling for corrupted image with NaNs
                if len(img.shape) == 3:
                    # Create a copy with NaNs replaced by zeros for display
                    display_img = np.copy(img)
                    mask = np.isnan(display_img)
                    display_img[mask] = 0
                    ax.imshow(display_img)
                    
                    # Highlight NaN areas in red
                    nan_mask = np.any(mask, axis=2)
                    overlay = np.zeros(img.shape[:2] + (4,))  # RGBA
                    overlay[..., 0] = 1.0  # Red channel
                    overlay[..., 3] = nan_mask * 0.7  # Alpha channel
                    ax.imshow(overlay)
                else:
                    # For grayscale with NaNs
                    display_img = np.copy(img)
                    mask = np.isnan(display_img)
                    display_img[mask] = 0
                    ax.imshow(display_img, cmap='gray')
                    
                    # Highlight NaN areas
                    overlay = np.zeros(img.shape + (4,))  # RGBA
                    overlay[..., 0] = 1.0  # Red channel
                    overlay[..., 3] = mask * 0.7  # Alpha channel
                    ax.imshow(overlay)
            else:
                # Normal display for non-corrupted images
                if len(img.shape) == 3:
                    ax.imshow(img)
                else:
                    ax.imshow(img, cmap='gray')
            
            title_text = title
            if metrics and title != 'Corrupted' and not np.isnan(list(metrics.values())[0]):
                title_text += f"\nMSE: {metrics['mse']:.2f}, PSNR: {metrics['psnr']:.2f}dB, SSIM: {metrics['ssim']:.4f}"
            
            ax.set_title(title_text)
            ax.axis('off')
        
        # Original, corrupted, and reconstructed images
        orig_metrics = self.calculate_metrics(self.original, 'original')
        corr_metrics = {'mse': float('nan'), 'psnr': float('nan'), 'ssim': float('nan')}
        recon_metrics = self.metrics.get('reconstructed', None)
        
        show_image(axes[0], self.original, 'Original', orig_metrics)
        show_image(axes[1], self.corrupted, 'Corrupted', corr_metrics)
        show_image(axes[2], self.reconstructed, 'Reconstructed', recon_metrics)
        
        # Display all filtered images
        for i, (name, img) in enumerate(self.filtered_images.items(), start=3):
            metrics = self.metrics.get(name, None)
            show_image(axes[i], img, f'Filter: {name}', metrics)
        
        # Hide any unused subplots
        for i in range(num_images, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('Image Filtering Comparison', fontsize=16)
        plt.tight_layout(pad=1.5)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Filter comparison image saved to {save_path}")
        
        plt.show()
    
    def get_best_filter(self, metric='psnr'):
        """
        Get the best filtered image based on a specific metric.
        
        Parameters:
        -----------
        metric : str
            Metric to use for comparison ('mse', 'psnr', or 'ssim')
            
        Returns:
        --------
        tuple
            (best_filter_name, best_filtered_image)
        """
        if not self.metrics:
            self.calculate_metrics()
        
        # Create a copy of metrics excluding 'original' and 'corrupted'
        filtered_metrics = {k: v for k, v in self.metrics.items() 
                            if k != 'original' and k != 'corrupted'}
        
        if not filtered_metrics:
            raise ValueError("No filtered images available for comparison")
        
        # Determine the best filter based on the specified metric
        if metric == 'mse':
            # For MSE, lower is better
            best_filter = min(filtered_metrics.items(), key=lambda x: x[1]['mse'])
        elif metric == 'psnr':
            # For PSNR, higher is better
            best_filter = max(filtered_metrics.items(), key=lambda x: x[1]['psnr'])
        elif metric == 'ssim':
            # For SSIM, higher is better
            best_filter = max(filtered_metrics.items(), key=lambda x: x[1]['ssim'])
        else:
            raise ValueError("Invalid metric. Use 'mse', 'psnr', or 'ssim'")
        
        best_name = best_filter[0]
        
        if best_name == 'reconstructed':
            best_image = self.reconstructed
        else:
            best_image = self.filtered_images[best_name]
        
        print(f"Best filter based on {metric}: {best_name} with "
              f"{metric} = {best_filter[1][metric]:.4f}")
        
        return best_name, best_image
    
    def save_best_image(self, output_path, metric='psnr'):
        """
        Save the best filtered image based on a specific metric.
        
        Parameters:
        -----------
        output_path : str
            Path to save the best filtered image
        metric : str
            Metric to use for comparison ('mse', 'psnr', or 'ssim')
        """
        best_name, best_image = self.get_best_filter(metric)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the image
        if len(best_image.shape) == 3:  # Color image
            cv2.imwrite(output_path, cv2.cvtColor(best_image, cv2.COLOR_RGB2BGR))
        else:  # Grayscale
            cv2.imwrite(output_path, best_image)
            
        print(f"Best filtered image ({best_name}) saved to {output_path}")