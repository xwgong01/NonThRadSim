import numpy as np 

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
    
class ParticleSpectrum():
    def __init__(self, gm_min, gm_max, shape, dtype = np.float32):
        self.gm_min = gm_min
        self.gm_max = gm_max
        self.shape = shape
        self.gm = np.geomspace(self.gm_min, self.gm_max, self.shape)
        self.spec = np.zeros(self.shape, dtype=dtype)
        
class PhotonSpectrum():
    def __init__(self, E_min, E_max, shape, dtype = np.float32):
        self.E_min = E_min
        self.E_max = E_max
        self.shape = shape
        self.E = np.geomspace(self.E_min, self.E_max, self.shape)
        self.spec = np.zeros(self.shape, dtype=dtype)

class Particle():
    def __init__(self, Grid, m, q, Spectrum_type):
        self.Grid = Grid
        self.mass = m
        self.charge = q
        self.spectrum_type = Spectrum_type
        self.value = np.empty(self.Grid.shape(), dtype=object)
        
class Photon(Particle):
    def __init__(self, Grid, Spectrum_type):
        super().__init__(Grid, 0, 0, Spectrum_type)
        self.value = np.empty(self.Grid.shape(), dtype=object)
        
    



if __name__ == "__main__":
    grid = Grid(dimension=2, dx=0.1, n_mesh=10, data_type=float)
    # print(grid.mesh)
    # print(grid.mesh[0].shape, grid.mesh[1].shape)
    spectrum = ParticleSpectrum(1e-1,1e10,101)
    particle = Particle(grid, 1.0, -1.0, ParticleSpectrum)
    photon = Photon(grid, PhotonSpectrum)
    print(grid.shape())
    print(particle.value.shape)
    print(photon.value.shape)
        