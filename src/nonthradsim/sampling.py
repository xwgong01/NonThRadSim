import numpy as np


def _shell_volume(edges):
    return 4 * np.pi / 3 * (edges[1:] ** 3 - edges[:-1] ** 3)


def _isotropic_vectors(rng, size):
    mu = rng.uniform(-1, 1, size)
    phi = rng.uniform(0, 2 * np.pi, size)
    sin_theta = np.sqrt(1 - mu**2)
    vectors = np.column_stack(
        (sin_theta * np.cos(phi), sin_theta * np.sin(phi), mu)
    )
    return vectors


def sample_photons(photon, n_particles, seed=None):
    """
    Sample photon packets from a spherical photon field.

    photon.value is dN / (dV dE). Energy is sampled uniformly in log(E).
    The returned weight makes the sampling unbiased.
    """
    if photon.grid.dimension != 1:
        raise ValueError("sample_photons requires a 1D radial grid")

    rng = np.random.default_rng(seed)
    grid = photon.grid
    E = photon.E
    dE = np.gradient(E)
    volume = _shell_volume(grid.edges)
    shell_number = volume * (photon.value * dE).sum(-1)
    shell_number = np.maximum(shell_number, 0)
    total_number = shell_number.sum()

    if total_number <= 0:
        raise ValueError("The photon field has no positive photons")

    shell_cdf = np.cumsum(shell_number) / total_number
    shell_cdf[-1] = 1.0
    shell = np.searchsorted(shell_cdf, rng.random(n_particles), side="right")
    shell = np.minimum(shell, grid.n_mesh - 1)

    r0 = grid.edges[shell]
    r1 = grid.edges[shell + 1]
    r = (r0**3 + rng.random(n_particles) * (r1**3 - r0**3)) ** (1 / 3)

    logE = rng.uniform(np.log(E[0]), np.log(E[-1]), n_particles)
    energy = np.exp(logE)
    logE_grid = np.log(E)
    local_value = np.array(
        [np.interp(logE[i], logE_grid, photon.value[shell[i]]) for i in range(n_particles)]
    )

    # q_r is the volume distribution inside the selected shell.
    q_shell = shell_number[shell] / total_number
    q_r = 3 * r**2 / (r1**3 - r0**3)
    q_E = 1 / (energy * (np.log(E[-1]) - np.log(E[0])))
    target = 4 * np.pi * r**2 * local_value
    weight = target / (q_shell * q_r * q_E) / n_particles

    position = r[:, None] * _isotropic_vectors(rng, n_particles)
    direction = _isotropic_vectors(rng, n_particles)
    radius = np.linalg.norm(position, axis=1)
    mu = np.sum(position * direction, axis=1) / np.maximum(radius, 1e-300)

    return {
        "position": position,
        "direction": direction,
        "energy": energy,
        "weight": weight,
        "mu": mu,
        "dead": np.zeros(n_particles, dtype=bool),
    }


def sample_manual_photons(position, energy, direction=None, weight=1.0, seed=None):
    """Create photon packets from user-supplied positions and energies."""
    position = np.asarray(position, dtype=float)
    if position.ndim == 1:
        position = position[None, :]
    if position.ndim != 2 or position.shape[1] != 3:
        raise ValueError("position must have shape (n_particles, 3)")

    n_particles = position.shape[0]
    energy = np.asarray(energy, dtype=float)
    if energy.ndim == 0:
        energy = np.full(n_particles, energy)
    if energy.shape != (n_particles,):
        raise ValueError("energy must have shape (n_particles,)")

    rng = np.random.default_rng(seed)
    if direction is None:
        direction = _isotropic_vectors(rng, n_particles)
    else:
        direction = np.asarray(direction, dtype=float)
        if direction.ndim == 1:
            direction = direction[None, :]
        if direction.shape != (n_particles, 3):
            raise ValueError("direction must have shape (n_particles, 3)")
        direction = direction / np.maximum(
            np.linalg.norm(direction, axis=1), 1e-300
        )[:, None]

    weight = np.asarray(weight, dtype=float)
    if weight.ndim == 0:
        weight = np.full(n_particles, weight)
    if weight.shape != (n_particles,):
        raise ValueError("weight must have shape (n_particles,)")

    radius = np.linalg.norm(position, axis=1)
    mu = np.sum(position * direction, axis=1) / np.maximum(radius, 1e-300)
    return {
        "position": position.copy(),
        "direction": direction.copy(),
        "energy": energy.copy(),
        "weight": weight.copy(),
        "mu": mu,
        "dead": np.zeros(n_particles, dtype=bool),
    }
