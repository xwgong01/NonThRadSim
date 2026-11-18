import numpy as np

from nonthradsim.distributions import Powerlaw
from nonthradsim.grid import Grid
from nonthradsim.mc import propagate_batch, run_mc
from nonthradsim.particle import Particle, Photon
from nonthradsim.sampling import sample_manual_photons
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


def test_manual_packets_propagate():
    grid = Grid(1, 1.0, 4)
    particle = Particle(grid, 1, -1, 1, 10, 12)
    particle.value[:] = 0
    position = np.zeros((8, 3))
    position[:, 0] = 0.5
    packets = sample_manual_photons(
        position,
        energy=1e-12,
        direction=np.tile([1.0, 0.0, 0.0], (8, 1)),
    )
    escaped = propagate_batch(grid, particle, packets, seed=5)
    assert escaped["energy"].size == 8
    assert np.all(escaped["n_scatter"] == 0)
