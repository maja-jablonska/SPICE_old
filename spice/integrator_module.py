from typing import Callable, Tuple
from jax import jit, vmap
import jax.numpy as jnp
import flax.linen as nn
from spice.coordinate_utils import generate_meshgrid, transform, rescale, get_integration_weights, interpolate2d_vec
from spice.spectra import generate_spectrum_overabundance_params_vec


class SpectrumIntegrator(nn.Module):
    interpolation_dims: Tuple[int, int]
    meshgrid_dims: Tuple[int, int]
    
    predict_spectrum_fn: Callable[[jnp.array, jnp.array], jnp.array]
    wavelengths: jnp.array

    rotation: jnp.float32
    inclination: jnp.float32

    # Stellar parameters
    teff: jnp.float32
    logg: jnp.float32
    vmic: jnp.float32
    metallicity: jnp.float32
    element: str

    abundance_map: jnp.array
    
    @nn.compact
    def __call__(self, phi: jnp.float32) -> jnp.array:
        """Simulate spectrum for given rotation phase phi.

        Args:
            phi (jnp.float32): phase in radians [0, 2pi]

        Returns:
            jnp.array: spectrum fluxes
        """

        phiv, thetav, dtheta, dphi = generate_meshgrid(*self.interpolation_dims)
        integration_weights = jnp.nan_to_num(get_integration_weights(phiv, dtheta, dphi)).flatten().reshape((-1, 1))
        
        trans_phi, trans_theta, rotation_map = transform(phiv,
                                                         thetav,
                                                         rot=self.rotation,
                                                         incl=self.inclination)
        
        integration_weights = jnp.nan_to_num(get_integration_weights(phiv,
                                                                     dtheta,
                                                                     dphi))

        transformated_coords = rescale((trans_phi+phi)%(2*jnp.pi),
                                       trans_theta, self.meshgrid_dims)
        
        abundances = interpolate2d_vec(transformated_coords[:, 1], transformated_coords[:, 0],
                                       self.abundance_map, *self.interpolation_dims)
        
        integration_weights = integration_weights.flatten().reshape((-1, 1))
        rotation_map = rotation_map.flatten().reshape((-1, 1))
        
        spectrum_params = generate_spectrum_overabundance_params_vec(self.teff,
                                                                     self.logg,
                                                                     self.vmic,
                                                                     self.metallicity,
                                                                     abundances,
                                                                     self.element)

        shifted_wavelengths = redshift_wavelengths_vec(self.wavelengths, rotation_map).reshape((-1, self.wavelengths.shape[-1]))

        spectra = self.predict_spectrum_fn(spectrum_params, shifted_wavelengths)
        
        return jnp.sum(jnp.multiply(spectra, integration_weights), axis=0)/jnp.sum(integration_weights)
