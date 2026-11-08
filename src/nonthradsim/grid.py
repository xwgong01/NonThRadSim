import numpy as np


class Grid:
    def __init__(self, dimension, dx, n_mesh, data_type=float):
        self.dimension = dimension
        self.dx = dx
        self.n_mesh = n_mesh
        self.data_type = data_type
        self.create_grid()
        self.shape = self.get_shape()

    def create_grid(self):
        if self.dimension == 1:
            # In the 1D model x is the radial coordinate.
            self.x = np.arange(self.n_mesh, dtype=self.data_type) * self.dx
            self.r = self.x
            self.edges = np.arange(self.n_mesh + 1, dtype=self.data_type) * self.dx
            self.radius = self.edges[-1]
        elif self.dimension == 2:
            self.x = np.arange(self.n_mesh) * self.dx - 0.5 * (self.n_mesh - 1) * self.dx
            self.y = np.arange(self.n_mesh) * self.dx - 0.5 * (self.n_mesh - 1) * self.dx
        elif self.dimension == 3:
            self.x = np.arange(self.n_mesh) * self.dx - 0.5 * (self.n_mesh - 1) * self.dx
            self.y = np.arange(self.n_mesh) * self.dx - 0.5 * (self.n_mesh - 1) * self.dx
            self.z = np.arange(self.n_mesh) * self.dx - 0.5 * (self.n_mesh - 1) * self.dx
        else:
            raise ValueError("Dimension must be 1, 2, or 3.")

    def get_shape(self):
        if self.dimension == 1:
            return (self.n_mesh,)
        if self.dimension == 2:
            return (self.n_mesh, self.n_mesh)
        if self.dimension == 3:
            return (self.n_mesh, self.n_mesh, self.n_mesh)
        raise ValueError("Dimension must be 1, 2, or 3.")
