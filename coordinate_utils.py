import jax.numpy as jnp
from jax import vmap
from typing import Tuple


def generate_meshgrid(N_r_prim: int = 100,
                      N_phi_prim: int = 100):
    d_phi = 2*jnp.pi/N_phi_prim
    phi_prim = jnp.linspace(d_phi/2, 2*jnp.pi-d_phi/2, N_phi_prim)
    d_r = 1/N_r_prim
    r_prim = jnp.linspace(d_r/2, 1-d_r/2, N_r_prim)

    r_v, phi_prim_v = jnp.meshgrid(r_prim, phi_prim)
    return r_v, phi_prim_v, 1/N_r_prim, 2*jnp.pi/N_phi_prim


def transform(r_prim, phi_prim, rot=100, incl=jnp.pi/4):
    c = jnp.arcsin(r_prim)
    x = r_prim*jnp.cos(phi_prim)
    y = r_prim*jnp.sin(phi_prim)
    theta = jnp.arcsin(jnp.cos(c)*jnp.cos(incl) + y*jnp.sin(incl)*jnp.sin(c)/r_prim)

    phi = jnp.arctan2(x*jnp.sin(c), r_prim*jnp.cos(c)*jnp.sin(incl) - y*jnp.sin(c)*jnp.cos(incl))
    return phi+jnp.pi, theta+jnp.pi/2, jnp.ones_like(theta)*rot*x*jnp.sin(incl)


def get_integration_weights(r_prim, d_r, d_phi, eps=0.6, limb_darkening: bool = False):
    if limb_darkening:
        return (1 - eps + eps*jnp.sqrt(1-r_prim**2)) * r_prim * d_phi * d_r
    else:
        return r_prim * d_phi * d_r
  

def rescale(trans_phi: jnp.array,
            trans_theta: jnp.array,
            dims: Tuple[float, float] = (128, 256)) -> jnp.array:
    
    trans_theta = jnp.nan_to_num(trans_theta)
    trans_phi = jnp.nan_to_num(trans_phi)
    
    rescaled_phi_v = trans_phi*(dims[0]/jnp.pi)
    rescaled_theta_v = trans_theta*(dims[1]/(2*jnp.pi))
    
    trans = jnp.concatenate([rescaled_phi_v.flatten().reshape((-1, 1)),
                             rescaled_theta_v.flatten().reshape((-1, 1))], axis=1)
    
    return trans


def interpolate2d(x: jnp.float64,
                  y: jnp.float64,
                  values_map: jnp.array,
                  phi_points: int = 128,
                  theta_points: int = 256) -> jnp.float64:
    x_floor = jnp.clip(jnp.array(jnp.floor(x), int), 0, phi_points-2)
    y_floor = jnp.clip(jnp.array(jnp.floor(y), int), 0, theta_points-2)
    
    x_dist = x-x_floor
    y_dist = y-y_floor
    
    return (values_map[x_floor][y_floor]*x_dist*y_dist+
            values_map[x_floor+1][y_floor]*y_dist*(1-x_dist)+
            values_map[x_floor][y_floor+1]*(1-y_dist)*x_dist+
            values_map[x_floor+1][y_floor+1]*(1-y_dist)*(1-x_dist))

interpolate2d_vec = vmap(interpolate2d, in_axes=(0, 0, None, None, None))