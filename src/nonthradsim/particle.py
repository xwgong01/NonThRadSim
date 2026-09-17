import numpy as np 
import matplotlib.pyplot as plt

class Particle():
    '''
    Spectrum is defined as dN/d\gamma, where \gamma is the lorentz factor. 
    all units are by default in cgs. 
    '''
    
    def __init__(self, grid, mass, charge, gm_min, gm_max, n_spectrum):
        self.grid = grid
        # print(grid.shape)
        self.mass = mass
        self.charge = charge
        self.gm_min = gm_min
        self.gm_max = gm_max
        self.gm = np.logspace(np.log10(gm_min), np.log10(gm_max), n_spectrum)
        self.n_spectrum = n_spectrum
        self.value = np.zeros((*grid.shape, n_spectrum))
        # print(self.value.shape)
        
    def set_norm(self, n):
        if np.ndim(n) == 0:
            n = np.full(self.grid.shape, n)
        # print(self.value.shape, n.shape)
        dloggm = np.log(self.gm[1:]/self.gm[:-1]).mean()
        norm = (self.value*dloggm * self.gm[None, None,None, :]).sum(-1) + 1e-50
        # print(n.shape, norm.shape)
        self.value *= n[:,:,:,None]/norm[:,:,:, None]
    
    def __getitem__(self, key):
        return self.value[key]
        
        
class Photon():
    '''
    Default energy is in erg.
    '''
    def __init__(self, grid, E_min, E_max, n_spectrum):
        self.grid = grid
        self.E_min = E_min
        self.E_max = E_max
        self.E = np.logspace(np.log10(E_min), np.log10(E_max), n_spectrum)
        self.n_spectrum = n_spectrum
        self.value = np.zeros((*grid.shape, n_spectrum))
    
    def set_norm(self, n):
            dlogE = np.log(self.E[1:]/self.E[:-1]).mean()
            norm = (self.value[...,:]*dlogE).sum(-1) + 1e-50
            self.value *= n[..., None]/norm[..., None]
    
    def __getitem__(self, key):
        return self.value[key]
            
