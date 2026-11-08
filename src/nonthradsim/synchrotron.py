import numpy as np
from scipy.special import kv
import nonthradsim.Constants as C


def K(x):
    return kv(5 / 3, x)


def F():
    """Return a table for the synchrotron function F(x)."""
    numx = 1001
    x = np.logspace(-5, 2, numx)
    xx, XX = np.meshgrid(x, x)
    integrand = XX * K(xx)
    integrand = np.where(xx >= XX, integrand, 0)
    fx = ((integrand[:, 1:] + integrand[:, :-1]) / 2 * (x[1:] - x[:-1])).sum(1)
    return x, fx


def synchrotron(photon, particle, magnetic_field, Fx):
    """
    Calculate the synchrotron photon spectrum at every radial grid point.
    Photon and particle spectra are dN / (dV dE) and dN / (dV dgamma).
    """
    E_ph = photon.E
    nu = E_ph / C.h
    gm = particle.gm
    B = magnetic_field
    if np.ndim(B) == 0:
        B = np.full(photon.grid.shape, B)
    dgm = np.gradient(gm)
    syn_spec = np.zeros_like(photon.value)

    for i in range(gm.shape[0]):
        nuc = 3 / (4 * np.pi) * gm[i] ** 2 * C.q * B / C.me / C.c
        x = nu[None, :] / nuc[..., None]
        fx = np.interp(x.ravel(), Fx[0], Fx[1]).reshape(x.shape)
        power = 3**0.5 * C.q**3 * B[..., None] / (2 * C.me * C.c**2) * fx
        syn_spec += power / C.h / E_ph[None, :] * particle.value[..., i, None] * dgm[i]

    photon.value += syn_spec
    return
