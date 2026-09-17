import numpy as np 
import matplotlib.pyplot as plt
class Grid:
    def __init__(self, dimension, dx, n_mesh):
        self.dimension = dimension
        self.dx = dx
        self.n_mesh = n_mesh
        self.create_grid()
        self.shape = self.get_shape()
    
    def create_grid(self):
        if self.dimension == 1:
            self.x = np.arange(self.n_mesh) * self.dx - 0.5 * (self.n_mesh - 1) * self.dx
            # self.grid = self.x
        elif self.dimension == 2:
            self.x = np.arange(self.n_mesh) * self.dx - 0.5 * (self.n_mesh - 1) * self.dx
            self.y = np.arange(self.n_mesh) * self.dx - 0.5 * (self.n_mesh - 1) * self.dx
            # self.grid = np.meshgrid(self.x, self.y)
        elif self.dimension == 3:
            self.x = np.arange(self.n_mesh) * self.dx - 0.5 * (self.n_mesh - 1) * self.dx
            self.y = np.arange(self.n_mesh) * self.dx - 0.5 * (self.n_mesh - 1) * self.dx
            self.z = np.arange(self.n_mesh) * self.dx - 0.5 * (self.n_mesh - 1) * self.dx
            # self.grid = np.meshgrid(self.x, self.y, self.z)
        else:
            raise ValueError("Dimension must be 1, 2, or 3.")

    def get_shape(self):
        if self.dimension == 1:
            return (self.n_mesh,)
        elif self.dimension == 2:
            return (self.n_mesh, self.n_mesh)
        elif self.dimension == 3:
            return (self.n_mesh, self.n_mesh, self.n_mesh)
        else:
            raise ValueError("Dimension must be 1, 2, or 3.")
        
