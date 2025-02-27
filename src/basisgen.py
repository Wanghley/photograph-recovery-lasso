import numpy as np
import matplotlib.pyplot as plt

class BasisGenerator:
    """A class for generating basis chips and basis vector matrix for image reconstruction.
    
    This class provides functionality to:
    1. Generate basis chips with specified dimensions
    2. Create basis vector matrix for LASSO reconstruction
    3. Support both UV and XY orientations
    
    Attributes:
        P (int): Height of the basis chips
        Q (int): Width of the basis chips
        uv_orientation (bool): Whether to use UV orientation (True) or XY orientation (False)
    """
    
    def __init__(self, P, Q, uv_orientation=True):
        """Initialize the BasisGenerator instance.
        
        Args:
            P (int): Height of the basis chips
            Q (int): Width of the basis chips
            uv_orientation (bool, optional): Use UV orientation if True, XY if False. Defaults to True.
        """
        self.P = P
        self.Q = Q
        self.uv_orientation = uv_orientation
        
    def generate_basis_chip(self, u, v):
        """Generate a single basis chip.
        
        Args:
            u (int): Frequency component in u direction
            v (int): Frequency component in v direction
            
        Returns:
            numpy.ndarray: Generated basis chip of shape (P, Q)
        """
        x = np.arange(1, self.P + 1)
        y = np.arange(1, self.Q + 1)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Normalization factors
        alpha_u = np.sqrt(1 / self.P) if u == 1 else np.sqrt(2 / self.P)
        beta_v = np.sqrt(1 / self.Q) if v == 1 else np.sqrt(2 / self.Q)
        
        # Basis chip formula
        chip = (
            alpha_u * beta_v *
            np.cos((np.pi * (2 * X - 1) * (u - 1)) / (2 * self.P)) *
            np.cos((np.pi * (2 * Y - 1) * (v - 1)) / (2 * self.Q))
        )
        
        return chip
    
    def generate_basis_matrix(self):
        """Generate the complete basis vector matrix.
        
        Returns:
            numpy.ndarray: Basis vector matrix of shape (P*Q, P*Q)
        """
        basis_matrix = np.zeros((self.P * self.Q, self.P * self.Q))
        
        if not self.uv_orientation:
            for v in range(1, self.Q + 1):
                for u in range(1, self.P + 1):
                    chip = self.generate_basis_chip(u, v)
                    rasterized_chip = chip.flatten(order='F')
                    basis_matrix[:, (v - 1) * self.P + (u - 1)] = rasterized_chip
        else:
            for u in range(1, self.P + 1):
                for v in range(1, self.Q + 1):
                    chip = self.generate_basis_chip(u, v)
                    rasterized_chip = chip.flatten(order='F')
                    basis_matrix[:, (u - 1) * self.Q + (v - 1)] = rasterized_chip
        
        return basis_matrix
    
    def get_basis_chip(self, index):
        """Get a specific basis chip by its index.
        
        Args:
            index (int): Index of the basis chip
            
        Returns:
            numpy.ndarray: Basis chip of shape (P, Q)
        """
        if self.uv_orientation:
            u = index // self.Q + 1
            v = index % self.Q + 1
        else:
            u = index // self.P + 1
            v = index % self.P + 1
        return self.generate_basis_chip(u, v)
    
    def get_all_basis_chips(self):
        """Get all basis chips in a list.
        
        Returns:
            list: List of basis chips
        """
        basis_chips = []
        for i in range(self.P * self.Q):
            basis_chips.append(self.get_basis_chip(i))
        return basis_chips
    
    def plot_basis_chip(self, u, v):
        """Plot a specific basis chip.
        
        Args:
            u (int): Frequency component in u direction
            v (int): Frequency component in v direction
        """
        chip = self.generate_basis_chip(u, v)
        plt.imshow(chip, cmap='gray')
        plt.colorbar()
        plt.title(f"Basis Chip T(u={u}, v={v})")
        plt.show()
    
    def plot_basis_matrix(self):
        """Plot the complete basis vector matrix."""
        basis_matrix = self.generate_basis_matrix()
        plt.figure(figsize=(8, 8))
        plt.imshow(basis_matrix, cmap='gray')
        plt.axis('off')
        plt.title(f"DCT Basis Matrix ($P={self.P}, Q={self.Q}$)")
        plt.show()