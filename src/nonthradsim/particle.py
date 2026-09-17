import numpy as np 
import matplotlib.pyplot as plt
from .spectrum import Spectrum

class Particle():
    '''
    Spectrum is defined as dN/d\gamma, where \gamma is the lorentz factor. 
    all units are by default in cgs. 
    '''
    
    def __init__(self, grid, mass, charge, gm_min, gm_max, n_spectrum):
        self.grid = grid
        self.mass = mass
        self.charge = charge
        self.gm_min = gm_min
        self.gm_max = gm_max
        self.n_spectrum = n_spectrum
        self.value = np.array([Spectrum(self.gm_min, self.gm_max, self.n_spectrum) for _ in range(np.prod(self.grid.shape()))], dtype=object)
        self.value = self.value.reshape(self.grid.shape())  
    
    def __getitem__(self, key):
        return self.value[key]
        
        
class Photon(Particle):
    '''
    Default energy is in erg.
    '''
    
    def __init__(self, grid, Emin, Emax, n_spectrum):
        super().__init__(grid, 0, 0, Emin, Emax, n_spectrum)
