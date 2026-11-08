import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator


def effective_sample_size(weight):
    """Return the effective sample size of weighted packets."""
    weight = np.asarray(weight)
    total = np.sum(weight)
    square = np.sum(weight**2)
    if square <= 0:
        return 0.0
    return total**2 / square


def spectrum(escaped, energy_edges):
    """Return a weighted escaped photon spectrum."""
    value, _ = np.histogram(escaped["energy"], bins=energy_edges, weights=escaped["weight"])
    return value


def radial_profile(escaped, radius, energy_edges, b_edges):
    """Return photon surface brightness in projected radial bins."""
    b = radius * np.sqrt(np.maximum(1 - escaped["mu"] ** 2, 0))
    image, _, _ = np.histogram2d(
        b, escaped["energy"], bins=(b_edges, energy_edges), weights=escaped["weight"]
    )
    area = np.pi * (b_edges[1:] ** 2 - b_edges[:-1] ** 2)
    return image / np.maximum(area[:, None], 1e-300)


def polar_image(
    escaped,
    radius,
    energy_edges,
    energy_band,
    radial_bins=64,
    angular_bins=128,
    kernel_width=2.0,
):
    """Build an area-conserving Gaussian-smoothed polar image."""
    u_edges = np.linspace(0, 1, radial_bins + 1)
    u_center = (u_edges[1:] + u_edges[:-1]) / 2
    b = radius * np.sqrt(np.maximum(1 - escaped["mu"] ** 2, 0))
    u = np.clip((b / radius) ** 2, 0, 1)
    if energy_edges[-1] < energy_band[0] or energy_edges[0] > energy_band[1]:
        raise ValueError("Energy band does not overlap energy bins")
    band_mask = (
        (escaped["energy"] >= energy_band[0])
        & (escaped["energy"] <= energy_band[1])
    )
    radial_mass, _ = np.histogram(
        u[band_mask],
        bins=u_edges,
        weights=escaped["weight"][band_mask],
    )

    # Each input u-bin is spread with a normalized Gaussian.
    sigma = max(kernel_width * (u_edges[1] - u_edges[0]), 1e-12)
    direct = np.exp(-0.5 * ((u_center[:, None] - u_center[None, :]) / sigma) ** 2)
    left_reflection = np.exp(-0.5 * ((u_center[:, None] + u_center[None, :]) / sigma) ** 2)
    right_reflection = np.exp(
        -0.5 * ((2 - u_center[:, None] - u_center[None, :]) / sigma) ** 2
    )
    kernel = direct + left_reflection + right_reflection
    kernel /= np.maximum(kernel.sum(axis=0, keepdims=True), 1e-300)
    smooth_mass = kernel @ radial_mass
    radial_value = smooth_mass / (np.pi * radius**2 * (u_edges[1] - u_edges[0]))

    b_center = radius * np.sqrt(u_center)
    b_grid = np.concatenate(([0], b_center, [radius]))
    radial_grid = np.concatenate(([radial_value[0]], radial_value, [0]))
    theta = np.linspace(0, 2 * np.pi, angular_bins, endpoint=False)
    theta_grid = np.concatenate((theta, [2 * np.pi]))
    polar_grid = np.repeat(radial_grid[None, :], angular_bins, axis=0)
    polar_grid = np.concatenate((polar_grid, polar_grid[:1]), axis=0)
    return theta_grid, b_grid, polar_grid


def circular_image(
    escaped,
    radius,
    energy_edges,
    energy_band,
    image_size=101,
    radial_bins=64,
    angular_bins=128,
    kernel_width=2.0,
):
    """Interpolate a polar image onto a Cartesian image grid."""
    theta, b, polar_value = polar_image(
        escaped,
        radius,
        energy_edges,
        energy_band,
        radial_bins=radial_bins,
        angular_bins=angular_bins,
        kernel_width=kernel_width,
    )
    interpolator = RegularGridInterpolator(
        (theta, b),
        polar_value,
        method="linear",
        bounds_error=False,
        fill_value=0,
    )
    axis = np.linspace(-radius, radius, image_size)
    xx, yy = np.meshgrid(axis, axis)
    bb = np.sqrt(xx**2 + yy**2)
    angle = np.mod(np.arctan2(yy, xx), 2 * np.pi)
    points = np.column_stack((angle.ravel(), bb.ravel()))
    image = interpolator(points).reshape(bb.shape)
    image[bb > radius] = 0

    # Enforce photon-number conservation on the Cartesian pixel grid.
    band_mask = (
        (escaped["energy"] >= energy_band[0])
        & (escaped["energy"] <= energy_band[1])
    )
    target_weight = np.sum(escaped["weight"][band_mask])
    pixel_area = (axis[1] - axis[0]) ** 2
    image_weight = np.sum(image[bb <= radius]) * pixel_area
    if image_weight > 0:
        image *= target_weight / image_weight
    return axis, image


def plot_bands(
    escaped,
    radius,
    energy_edges,
    energy_bands,
    image_size=101,
    radial_bins=64,
    angular_bins=128,
    kernel_width=2.0,
    **kwargs,
):
    """Plot circular sky maps for several energy bands."""
    n_band = len(energy_bands)
    fig, axes = plt.subplots(1, n_band, figsize=(4 * n_band, 4), squeeze=False)
    for i, band in enumerate(energy_bands):
        axis, image = circular_image(
            escaped,
            radius,
            energy_edges,
            band,
            image_size=image_size,
            radial_bins=radial_bins,
            angular_bins=angular_bins,
            kernel_width=kernel_width,
        )
        image_kwargs = {"cmap": "magma", "interpolation": "bilinear"}
        image_kwargs.update(kwargs)
        axes[0, i].imshow(
            image,
            origin="lower",
            extent=(axis[0], axis[-1], axis[0], axis[-1]),
            **image_kwargs
        )
        axes[0, i].set_title(f"{band[0]:.2g} - {band[1]:.2g}")
        axes[0, i].set_xlabel("projected radius")
        axes[0, i].set_ylabel("projected radius")
    fig.tight_layout()
    return fig, axes
