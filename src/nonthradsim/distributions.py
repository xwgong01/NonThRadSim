import numpy as np
import nonthradsim.Constants as C

def MaxwellBoltzmann(particle, T, n):
    # TODO: Implement the Maxwell-Boltzmann distribution for the particle spectrum
    if np.ndim(T) == 0:
        T = np.full(particle.grid.shape, T)
    if np.ndim(n) == 0:
        n = np.full(particle.grid.shape, n)
    particle.value = np.sqrt(particle.gm[None, None, None, :])*np.exp(-(particle.gm[None, None, None, :]) * C.me * C.c**2 / T[..., None])
    particle.set_norm(n)
    return

def Powerlaw(particle, alpha, gmmin, gmmax, n):
    # TODO: Implement the power-law distribution for the particle spectrum
    if np.ndim(alpha) == 0:
        alpha = np.full(particle.grid.shape, alpha)
    if np.ndim(gmmin) == 0:
        gmmin = np.full(particle.grid.shape, gmmin)
    if np.ndim(gmmax) == 0:
        gmmax = np.full(particle.grid.shape, gmmax)
    if np.ndim(n) == 0:
        n = np.full(particle.grid.shape, n)
    # print(gmmin.shape, gmmax.shape, n.shape)
    # print(particle.grid.shape)
    particle.value = np.where((particle.gm[None, None, None, :] >= gmmin[..., None]) & (particle.gm[None, None, None, :] <= gmmax[..., None]), n[..., None] * particle.gm[None, None, None, :] ** (-alpha[..., None]), 0)
    particle.set_norm(n)
    return
