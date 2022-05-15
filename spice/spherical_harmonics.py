import jax.numpy as jnp
from jax import jit, vmap
from jax.scipy.special import sph_harm
from functools import partial


@partial(jit, static_argnums=(2, 3))
def spherical_harmonic(m: jnp.float32,
                       n: jnp.float32,
                       x_length: jnp.float32,
                       y_length: jnp.float32) -> jnp.array:
    """Create a spherical harmonic map with values rescaled to (0-1) range.

    Args:
        m (jnp.float32): m mode
        n (jnp.float32): n mode
        x_length (jnp.float32): length of the x coordinates dimension
        y_length (jnp.float32): length of the y coordinates dimension

    Returns:
        jnp.array: a 2D [x_length, y_length] map of (0-1) values
    """
    sphv = vmap(lambda yp: sph_harm(jnp.array([jnp.rint(m).astype(int)]),
                                    jnp.array([jnp.rint(n).astype(int)]),
                                    jnp.linspace(0, 2*jnp.pi, x_length),
                                    jnp.array([yp]), n_max=10).real)
    s: jnp.array = sphv(jnp.linspace(0, jnp.pi, y_length))

    # Rescale to the (0-1) range
    return (s-s.min())/(s.max()-s.min())
