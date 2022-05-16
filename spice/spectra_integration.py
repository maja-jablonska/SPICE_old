from typing import Callable, Tuple
from jax import jit
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from spice.coordinate_utils import generate_meshgrid, transform, rescale, get_integration_weights, interpolate2d_vec
from spice.spectra import generate_spectrum_overabundance_params_vec, predict_spectra, redshift_wavelengths_vec


def generate_transformed_geometry(phi: jnp.float32,
                                  rotation: jnp.float32,
                                  inclination: jnp.float32,
                                  interpolation_dims: Tuple[int, int],
                                  abundance_map: jnp.array) -> jnp.array:

    phiv, thetav, dtheta, dphi = generate_meshgrid(*interpolation_dims)
    
    trans_phi, trans_theta, rotation_map = transform(phiv, thetav, rot=rotation, incl=inclination)
    
    integration_weights = jnp.nan_to_num(get_integration_weights(phiv, dtheta, dphi))

    transformated_coords = rescale((trans_phi+phi)%(2*jnp.pi), trans_theta, abundance_map.shape)

    return integration_weights, rotation_map, transformated_coords

def generate_spectrum_integration_function(wavelengths: jnp.array,
                                           rotation_velocity: jnp.float32,
                                           inclination: jnp.float32,
                                           teff: jnp.float32,
                                           logg: jnp.float32,
                                           vmic: jnp.float32,
                                           metallicity: jnp.float32,
                                           abundance_map: jnp.array,
                                           element: str,
                                           interpolation_dims: Tuple[int, int] = (50, 50),
                                           predict_spectra_fn: Callable[[jnp.array, jnp.array], jnp.array] = predict_spectra) -> Callable[[jnp.float32], jnp.array]:
    """Generate a function that outputs spectrum fluxes for given rotation phase phi, assuming stellar parameters, abundance map and desired wavelengths range

    Args:
        wavelengths (jnp.array): spectrum fluxes wavelengths range (in angstroms)
        rotation_velocity (jnp.float32): rotation velocity in km/s
        inclination (jnp.float32): inclination in [0, pi]
        teff (jnp.float32): effective temperature in K
        logg (jnp.float32): log g
        vmic (jnp.float32): microturbulence velocity in km/s
        metallicity (jnp.float32): metallicity in log format
        abundance_map (jnp.array): a matrix representing an element's abundances
        element (str): element that exhibits non-homogenous distribution across stellar disk
        interpolation_dims (Tuple[int, int], optional): number of points to sample the abundances at. Defaults to (50, 50).
        predict_spectra_fn (Callable[[jnp.array, jnp.array], jnp.array], optional): function calculating spectrum fluxes for given stellar parameters
            and wavelengths. Defaults to predict_spectra.

    Returns:
        Callable[[jnp.float32], jnp.array]: Function that outputs spectrum fluxes for given rotation phase phi.
    """

    @jit
    def integrate_for_phi(phi: jnp.float32) -> jnp.array:

        integration_weights, rotation_map, transformated_coords = generate_transformed_geometry(phi=phi,
                                                                                                rotation=rotation_velocity,
                                                                                                inclination=inclination,
                                                                                                interpolation_dims=interpolation_dims,
                                                                                                abundance_map=abundance_map)
        
        abundances = interpolate2d_vec(transformated_coords[:, 1], transformated_coords[:, 0], abundance_map, *interpolation_dims)
        
        integration_weights = integration_weights.flatten().reshape((-1, 1))
        rotation_map = rotation_map.flatten().reshape((-1, 1))
        
        spectrum_params = generate_spectrum_overabundance_params_vec(teff, logg, vmic, metallicity, abundances, element)

        shifted_wavelengths = redshift_wavelengths_vec(wavelengths, rotation_map).reshape((-1, wavelengths.shape[-1]))

        spectra = predict_spectra_fn(spectrum_params, shifted_wavelengths)
        
        return jnp.sum(jnp.multiply(spectra, integration_weights), axis=0)/jnp.sum(integration_weights)

    return integrate_for_phi


def generate_plot_function_for_phi(wavelengths: jnp.array,
                                   rotation_velocity: jnp.float32,
                                   inclination: jnp.float32,
                                   teff: jnp.float32,
                                   logg: jnp.float32,
                                   vmic: jnp.float32,
                                   metallicity: jnp.float32,
                                   abundance_map: jnp.array,
                                   element: str,
                                   interpolation_dims: Tuple[int, int] = (50, 50),
                                   predict_spectra_fn: Callable[[jnp.array, jnp.array], jnp.array] = predict_spectra) -> Callable[[jnp.float32], jnp.array]:
    """Generate a function that outputs spectrum fluxes for given rotation phase phi, assuming stellar parameters, abundance map and desired wavelengths range

    Args:
        wavelengths (jnp.array): spectrum fluxes wavelengths range (in angstroms)
        rotation_velocity (jnp.float32): rotation velocity in km/s
        inclination (jnp.float32): inclination in [0, pi]
        teff (jnp.float32): effective temperature in K
        logg (jnp.float32): log g
        vmic (jnp.float32): microturbulence velocity in km/s
        metallicity (jnp.float32): metallicity in log format
        abundance_map (jnp.array): a matrix representing an element's abundances
        element (str): element that exhibits non-homogenous distribution across stellar disk
        interpolation_dims (Tuple[int, int], optional): number of points to sample the abundances at. Defaults to (50, 50).
        predict_spectra_fn (Callable[[jnp.array, jnp.array], jnp.array], optional): function calculating spectrum fluxes for given stellar parameters
            and wavelengths. Defaults to predict_spectra.

    Returns:
        Callable[[jnp.float32], jnp.array]: Function that outputs spectrum fluxes for given rotation phase phi.
    """

    def plot_for_phi(phi: jnp.float32,
                     abundance_map_cmap: str = 'cividis') -> jnp.array:

        integration_weights, rotation_map, transformated_coords = generate_transformed_geometry(phi=phi,
                                                                   rotation=rotation_velocity,
                                                                   inclination=inclination,
                                                                   interpolation_dims=interpolation_dims,
                                                                   abundance_map=abundance_map)
        
        abundances = interpolate2d_vec(transformated_coords[:, 1], transformated_coords[:, 0], abundance_map, *interpolation_dims)
        
        integration_weights = integration_weights.flatten().reshape((-1, 1))
        rotation_map = rotation_map.flatten().reshape((-1, 1))
        
        spectrum_params = generate_spectrum_overabundance_params_vec(teff, logg, vmic, metallicity, abundances, element)

        shifted_wavelengths = redshift_wavelengths_vec(wavelengths, rotation_map).reshape((-1, wavelengths.shape[-1]))

        spectra = predict_spectra_fn(spectrum_params, shifted_wavelengths)
        
        spectrum = jnp.sum(jnp.multiply(spectra, integration_weights), axis=0)/jnp.sum(integration_weights)

        fig, (ax1, ax2) = plt.subplots(ncols=1, nrows=2, figsize=(7, 7))

        colormap_plain = ax1.imshow(abundance_map, cmap=abundance_map_cmap)
        fig.colorbar(colormap_plain, ax=ax1, shrink=1.)

        ax1.scatter(transformated_coords[:, 0], transformated_coords[:, 1], color='white', s=10., alpha=0.25)
        ax2.plot(wavelengths, spectrum, c='royalblue')

        ax1.set_yticks(np.linspace(0, abundance_map.shape[0], 5))
        ax1.set_yticklabels(['0', r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$", r"$\pi$"])

        ax1.set_xticks(np.linspace(0, abundance_map.shape[1], 5))
        ax1.set_xticklabels([ r"$\pi$", r"$\frac{3\pi}{2}$", '0', r"$\frac{\pi}{2}$", r"$\pi$"])
        ax1.set_xlabel(r"$\phi$")
        ax1.set_ylabel(r"$\theta$")

        ax2.set_ylabel('Normalized flux') 
        ax2.grid(linestyle='--')
        ax2.set_xlabel('Wavelength [$\AA$]')

        fig.patch.set_facecolor('white')

    return plot_for_phi