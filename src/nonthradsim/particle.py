import numpy as np


class Particle:
    """
    Particle spectrum is dN / (dV dgamma).
    All units are by default in cgs.
    """

    def __init__(self, grid, mass, charge, gm_min, gm_max, n_spectrum):
        self.grid = grid
        self.mass = mass
        self.charge = charge
        self.gm_min = gm_min
        self.gm_max = gm_max
        self.gm = np.logspace(np.log10(gm_min), np.log10(gm_max), n_spectrum)
        self.n_spectrum = n_spectrum
        self.value = np.zeros((*grid.shape, n_spectrum))

    def set_norm(self, n):
        if np.ndim(n) == 0:
            n = np.full(self.grid.shape, n)
        dgm = np.gradient(self.gm)
        norm = (self.value * dgm).sum(-1) + 1e-300
        self.value *= n[..., None] / norm[..., None]

    def __getitem__(self, key):
        return self.value[key]


class Photon:
    """
    Photon spectrum is dN / (dV dE). Energy is in erg.
    """

    def __init__(self, grid, E_min, E_max, n_spectrum):
        self.grid = grid
        self.E_min = E_min
        self.E_max = E_max
        self.E = np.logspace(np.log10(E_min), np.log10(E_max), n_spectrum)
        self.n_spectrum = n_spectrum
        self.value = np.zeros((*grid.shape, n_spectrum))

    def set_norm(self, n):
        if np.ndim(n) == 0:
            n = np.full(self.grid.shape, n)
        dE = np.gradient(self.E)
        norm = (self.value * dE).sum(-1) + 1e-300
        self.value *= n[..., None] / norm[..., None]

    def __getitem__(self, key):
        return self.value[key]
