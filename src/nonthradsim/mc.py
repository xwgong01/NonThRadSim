import numpy as np
from joblib import Parallel, delayed

import nonthradsim.Constants as C
from .sampling import sample_photons


def sigma_kn_total(epsilon):
    """Total Klein-Nishina cross section, epsilon in mec2 units."""
    epsilon = np.asarray(epsilon, dtype=float)
    result = np.empty_like(epsilon)
    small = epsilon < 1e-4
    result[small] = C.sigma_T
    x = np.maximum(epsilon[~small], 1e-12)
    log_term = np.log1p(2 * x)
    result[~small] = 3 * C.sigma_T / 4 * (
        (1 + x) / x**3 * (2 * x * (1 + x) / (1 + 2 * x) - log_term)
        + log_term / (2 * x)
        - (1 + 3 * x) / (1 + 2 * x) ** 2
    )
    return result


def _sphere_distance(position, direction, radius, eps):
    dot = np.sum(position * direction, axis=1)
    r2 = np.sum(position * position, axis=1)
    discriminant = dot**2 + radius**2 - r2
    valid = discriminant >= 0
    root = np.sqrt(np.maximum(discriminant, 0))
    s1 = -dot - root
    s2 = -dot + root
    s1 = np.where(s1 > eps, s1, np.inf)
    s2 = np.where(s2 > eps, s2, np.inf)
    result = np.minimum(s1, s2)
    return np.where(valid, result, np.inf)


def _cell_boundary(position, direction, cell, edges, eps):
    low = edges[cell]
    high = edges[cell + 1]
    s_high = _sphere_distance(position, direction, high, eps)
    s_low = np.full_like(s_high, np.inf)
    nonzero = low > 0
    if np.any(nonzero):
        s_low[nonzero] = _sphere_distance(
            position[nonzero], direction[nonzero], low[nonzero], eps
        )
    use_low = s_low < s_high
    distance = np.minimum(s_low, s_high)
    hit_high = ~use_low
    return distance, hit_high


def _kn_angle(epsilon, rng):
    """Sample the Klein-Nishina scattering angle in the electron frame."""
    size = epsilon.size
    mu = np.empty(size)
    epsilon_out = np.empty(size)
    pending = np.arange(size)

    while pending.size:
        candidate_mu = rng.uniform(-1, 1, pending.size)
        candidate_out = epsilon[pending] / (
            1 + epsilon[pending] * (1 - candidate_mu)
        )
        ratio = candidate_out / epsilon[pending]
        kernel = ratio**2 * (
            1 / ratio + ratio - (1 - candidate_mu**2)
        )
        accepted = rng.random(pending.size) < kernel / 2
        accepted_index = pending[accepted]
        mu[accepted_index] = candidate_mu[accepted]
        epsilon_out[accepted_index] = candidate_out[accepted]
        pending = pending[~accepted]

    return mu, epsilon_out


def _orthogonal_basis(direction):
    reference = np.zeros_like(direction)
    use_z = np.abs(direction[:, 2]) < 0.9
    reference[use_z, 2] = 1
    reference[~use_z, 1] = 1
    e1 = np.cross(reference, direction)
    e1 /= np.linalg.norm(e1, axis=1)[:, None]
    e2 = np.cross(direction, e1)
    return e1, e2


def _scatter_kn(energy, direction, gamma, electron_beta, rng):
    """Scatter photons from isotropic electrons using the KN kernel."""
    beta2 = np.sum(electron_beta * electron_beta, axis=1)
    gamma_energy = energy / (C.me * C.c**2)
    beta_dot_n = np.sum(electron_beta * direction, axis=1)
    epsilon_in = gamma * gamma_energy * (1 - beta_dot_n)

    mu_s, epsilon_out = _kn_angle(epsilon_in, rng)
    phi = rng.uniform(0, 2 * np.pi, energy.size)

    # Incident photon in the electron rest frame.
    p0 = gamma_energy
    p = p0[:, None] * direction
    dot = np.sum(electron_beta * p, axis=1)
    factor = np.zeros_like(dot)
    nonzero = beta2 > 1e-20
    factor[nonzero] = (gamma[nonzero] - 1) * dot[nonzero] / beta2[nonzero] - gamma[nonzero] * p0[nonzero]
    p_prime = p + factor[:, None] * electron_beta
    epsilon_prime = gamma * (p0 - dot)
    n_prime = p_prime / np.maximum(epsilon_prime, 1e-300)[:, None]

    e1, e2 = _orthogonal_basis(n_prime)
    sin_s = np.sqrt(1 - mu_s**2)
    n_out_prime = (
        mu_s[:, None] * n_prime
        + sin_s[:, None] * np.cos(phi)[:, None] * e1
        + sin_s[:, None] * np.sin(phi)[:, None] * e2
    )
    p_out_prime = epsilon_out[:, None] * n_out_prime

    dot_out = np.sum(electron_beta * p_out_prime, axis=1)
    factor_out = np.zeros_like(dot_out)
    factor_out[nonzero] = (
        (gamma[nonzero] - 1) * dot_out[nonzero] / beta2[nonzero]
        + gamma[nonzero] * epsilon_out[nonzero]
    )
    p_out = p_out_prime + factor_out[:, None] * electron_beta
    epsilon_lab = gamma * (epsilon_out + dot_out)
    direction_out = p_out / np.maximum(epsilon_lab, 1e-300)[:, None]
    return epsilon_lab * C.me * C.c**2, direction_out


def _sample_collision(particle, cell, energy, direction, rng):
    """Sample a candidate electron and accept a real KN collision."""
    gm = particle.gm
    dgm = np.gradient(gm)
    spectrum = np.maximum(particle.value[cell], 0)
    cdf = np.cumsum(spectrum * dgm, axis=1)
    norm = cdf[:, -1]
    valid = norm > 0
    cdf[valid] /= norm[valid, None]
    u = rng.random(cell.size)
    gamma_index = (cdf < u[:, None]).sum(axis=1)
    gamma_index = np.minimum(gamma_index, gm.size - 1)
    gamma = gm[gamma_index]
    beta = np.sqrt(np.maximum(1 - 1 / gamma**2, 0))

    mu_e = rng.uniform(-1, 1, cell.size)
    phi_e = rng.uniform(0, 2 * np.pi, cell.size)
    e1, e2 = _orthogonal_basis(direction)
    sin_e = np.sqrt(1 - mu_e**2)
    electron_direction = (
        mu_e[:, None] * direction
        + sin_e[:, None] * np.cos(phi_e)[:, None] * e1
        + sin_e[:, None] * np.sin(phi_e)[:, None] * e2
    )
    electron_beta = beta[:, None] * electron_direction
    epsilon = energy / (C.me * C.c**2)
    epsilon_prime = gamma * epsilon * (1 - beta * mu_e)
    accept = (1 - beta * mu_e) * sigma_kn_total(epsilon_prime) / (2 * C.sigma_T)
    real = valid & (rng.random(cell.size) < np.minimum(accept, 1))
    return real, gamma[real], electron_beta[real]


def run_mc_batch(grid, particle, photon, n_particles, seed=None, max_steps=100000):
    """Run one vectorized batch of spherical MC photons."""
    packets = sample_photons(photon, n_particles, seed=seed)
    rng = np.random.default_rng(seed + 1 if seed is not None else None)
    position = packets["position"]
    direction = packets["direction"]
    energy = packets["energy"]
    weight = packets["weight"]
    dead = packets["dead"]
    n_scatter = np.zeros(n_particles, dtype=int)
    scatter_radius = []
    tau = rng.exponential(1, n_particles)

    dgm = np.gradient(particle.gm)
    electron_density = np.maximum((particle.value * dgm).sum(-1), 0)
    eps = max(grid.dx * 1e-10, 1e-30)

    for _ in range(max_steps):
        active = np.flatnonzero(~dead)
        if active.size == 0:
            break

        pos = position[active]
        direc = direction[active]
        radius = np.linalg.norm(pos, axis=1)
        cell = np.searchsorted(grid.edges[1:], radius, side="right")
        cell = np.clip(cell, 0, grid.n_mesh - 1)
        distance, hit_high = _cell_boundary(pos, direc, cell, grid.edges, eps)
        alpha = 2 * C.sigma_T * electron_density[cell]
        can_scatter = (alpha > 0) & (tau[active] <= alpha * distance)
        step = np.where(can_scatter, tau[active] / np.maximum(alpha, 1e-300), distance)
        position[active] += direc * step[:, None]
        tau[active] -= alpha * step

        escaped = (~can_scatter) & hit_high & (cell == grid.n_mesh - 1)
        dead[active[escaped]] = True

        crossing = (~can_scatter) & ~escaped
        tau[active[crossing]] = tau[active[crossing]]
        collision_index = active[can_scatter]
        if collision_index.size:
            real, gamma, electron_beta = _sample_collision(
                particle, cell[can_scatter], energy[collision_index], direction[collision_index], rng
            )
            real_index = collision_index[real]
            if real_index.size:
                scatter_radius.append(np.linalg.norm(position[real_index], axis=1))
                new_energy, new_direction = _scatter_kn(
                    energy[real_index], direction[real_index], gamma, electron_beta, rng
                )
                energy[real_index] = new_energy
                direction[real_index] = new_direction
                n_scatter[real_index] += 1
            tau[collision_index] = rng.exponential(1, collision_index.size)

        position[active[~can_scatter & ~escaped]] += (
            direc[~can_scatter & ~escaped] * (eps / 2)
        )

    if not np.all(dead):
        raise RuntimeError("MC batch reached max_steps before all photons escaped")

    escaped_position = position[dead]
    escaped_direction = direction[dead]
    escaped_radius = np.linalg.norm(escaped_position, axis=1)
    mu = np.sum(escaped_position * escaped_direction, axis=1) / np.maximum(escaped_radius, 1e-300)
    if scatter_radius:
        scatter_radius = np.concatenate(scatter_radius)
    else:
        scatter_radius = np.empty(0)
    return {
        "energy": energy[dead],
        "mu": mu,
        "weight": weight[dead],
        "n_scatter": n_scatter[dead],
        "scatter_radius": scatter_radius,
        "position": escaped_position,
        "direction": escaped_direction,
    }


def _run_batch(args):
    return run_mc_batch(*args)


def run_mc(grid, particle, photon, n_particles, batch_size, n_jobs=1, seed=None, max_steps=100000):
    """Run batches and combine the escaped photon packets."""
    n_batch = int(np.ceil(n_particles / batch_size))
    seeds = np.random.SeedSequence(seed).spawn(n_batch)
    args = []
    for i in range(n_batch):
        count = min(batch_size, n_particles - i * batch_size)
        args.append((grid, particle, photon, count, int(seeds[i].generate_state(1)[0]), max_steps))

    if n_jobs == 1:
        results = [_run_batch(item) for item in args]
    else:
        results = Parallel(n_jobs=n_jobs)(delayed(_run_batch)(item) for item in args)

    return {
        key: np.concatenate([result[key] for result in results])
        for key in results[0]
    }
