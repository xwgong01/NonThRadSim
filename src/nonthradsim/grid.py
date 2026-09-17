import numpy as np 
import matplotlib.pyplot as plt
class Grid:
    def __init__(self, dimension, dx, n_mesh, data_type):
        self.dimension = dimension
        self.dx = dx
        self.n_mesh = n_mesh
        self.create_grid()

    def create_grid(self):
        extent = np.arange(self.n_mesh) * self.dx - self.n_mesh * self.dx / 2
        if self.dimension == 1:
            self.mesh = extent
            
        elif self.dimension == 2:
            self.mesh = np.meshgrid(extent, extent)
        elif self.dimension == 3:
            self.mesh = np.meshgrid(extent, extent, extent)
        else:
            raise ValueError("Dimension must be 1, 2 or 3.")
        return
    
    def shape(self):
        if self.dimension == 1:
            return (self.n_mesh,)
        elif self.dimension == 2:
            return (self.n_mesh, self.n_mesh)
        elif self.dimension == 3:
            return (self.n_mesh, self.n_mesh, self.n_mesh)
        else:
            raise ValueError("Dimension must be 1, 2 or 3.")
        
    def coord(self, coord):
        if self.dimension == 3:
            return (self.mesh[0][coord[0], coord[1], coord[2]],
                    self.mesh[1][coord[0], coord[1], coord[2]],
                    self.mesh[2][coord[0], coord[1], coord[2]])
        
