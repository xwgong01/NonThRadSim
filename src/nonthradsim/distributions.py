import numpy as np
import nonthradsim.Constants as C


def MaxwellBoltzmann(particle, T, n):
    if np.ndim(T) == 0:
        T = np.full(particle.grid.shape, T)
    gm = particle.gm.reshape((1,) * len(particle.grid.shape) + (-1,))
    particle.value = np.sqrt(gm) * np.exp(-gm * C.me * C.c**2 / T[..., None])
    particle.set_norm(n)
    return


def Powerlaw(particle, alpha, gmmin, gmmax, n):
    if np.ndim(alpha) == 0:
        alpha = np.full(particle.grid.shape, alpha)
    if np.ndim(gmmin) == 0:
        gmmin = np.full(particle.grid.shape, gmmin)
    if np.ndim(gmmax) == 0:
        gmmax = np.full(particle.grid.shape, gmmax)
    gm = particle.gm.reshape((1,) * len(particle.grid.shape) + (-1,))
    particle.value = np.where(
        (gm >= gmmin[..., None]) & (gm <= gmmax[..., None]),
        gm ** (-alpha[..., None]),
        0,
    )
    particle.set_norm(n)
    return
