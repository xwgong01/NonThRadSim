import numpy as np 
import matplotlib.pyplot as plt
from .spectrum import Spectrum

class Particle():
    def __init__(self, grid, m, q, gm_min, gm_max, n_spectrum):
        self.grid = grid
        self.mass = m
        self.charge = q
        self.gm_min = gm_min
        self.gm_max = gm_max
        self.n_spectrum = n_spectrum
        self.value = np.array([Spectrum(self.gm_min, self.gm_max, self.n_spectrum) for _ in range(np.prod(self.Grid.shape()))], dtype=object)
        self.value = self.value.reshape(self.Grid.shape())  
    
    def __getitem__(self, key):
        return self.value[key]
        
        
class Photon(Particle):
    def __init__(self, grid, Emin, Emax, n_spectrum):
        super().__init__(grid, 0, 0, Emin, Emax, n_spectrum)
