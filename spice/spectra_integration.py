from typing import Callable, Tuple
from jax import jit
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

    transformated_coords = rescale((trans_phi+phi)%(2*jnp.pi), trans_theta, (abundance_map.shape[1], abundance_map.shape[0]))

    return integration_weights, rotation_map, transformated_coords

def generate_spectrum_integration_function(wavelengths: jnp.array,
                                           rotation: jnp.float32,
                                           inclination: jnp.float32,
                                           teff: jnp.float32,
                                           logg: jnp.float32,
                                           vmic: jnp.float32,
                                           metallicity: jnp.float32,
                                           abundance_map: jnp.array,
                                           element: str,
                                           interpolation_dims: Tuple[int, int] = (50, 50),
                                           predict_spectra_fn: Callable[[jnp.array, jnp.array], jnp.array] = predict_spectra) -> Callable[[jnp.float32], jnp.array]:

    @jit
    def integrate_for_phi(phi: jnp.float32) -> jnp.array:

        integration_weights, rotation_map, transformated_coords = generate_transformed_geometry(phi=phi,
                                                                                                rotation=rotation,
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
