import numpy as np

from nonthradsim.distributions import Powerlaw
from nonthradsim.grid import Grid
from nonthradsim.mc import run_mc
from nonthradsim.particle import Particle, Photon
from nonthradsim.synchrotron import F, synchrotron


def test_spherical_synchrotron_shape():
    grid = Grid(1, 1.0, 4)
    particle = Particle(grid, 1, -1, 1, 1e3, 16)
    Powerlaw(particle, 2, 1, 1e3, 1e4)
    photon = Photon(grid, 1e-20, 1e-12, 24)
    synchrotron(photon, particle, 1e-3, F())
    assert photon.value.shape == (4, 24)
    assert np.all(np.isfinite(photon.value))


def test_mc_packets_escape():
    grid = Grid(1, 1.0, 4)
    particle = Particle(grid, 1, -1, 1, 10, 12)
    Powerlaw(particle, 2, 1, 10, 1e24)
    photon = Photon(grid, 1e-12, 1e-9, 16)
    photon.value[:] = 1
    escaped = run_mc(grid, particle, photon, 32, 8, n_jobs=1, seed=2)
    assert escaped["energy"].size == 32
    assert np.all(escaped["mu"] >= -1)
    assert np.all(escaped["mu"] <= 1)
    assert np.all(escaped["weight"] > 0)
