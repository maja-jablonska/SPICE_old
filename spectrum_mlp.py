from operator import indexOf
from typing import List, Sequence
from functools import partial
import jax.numpy as jnp
from jax import jit, vmap, random
import flax.linen as nn
from flax.serialization import from_bytes


c: jnp.float32 = jnp.float32(299792.458)


OVERABUNDANCES: List[str] = ['Mn', 'Fe', 'Si', 'Ca', 'C', 'N', 'O', 'Hg']


def scale_spectra_parameters(log_teff: jnp.float32,
                             logg: jnp.float32,
                             vmic: jnp.float32,
                             me: jnp.float32,
                             a_Mn: jnp.float32,
                             a_Fe: jnp.float32,
                             a_Si: jnp.float32,
                             a_Ca: jnp.float32,
                             a_C: jnp.float32,
                             a_N: jnp.float32,
                             a_O: jnp.float32,
                             a_Hg: jnp.float32):
    """Scale spectra parameters to (0-1) range (best for ML models)

    Args:
        log_teff (jnp.float32): logarythmic effective temperature [3.845098, 3.929419]
        logg (jnp.float32): log g [3.5, 5.0]
        vmic (jnp.float32): microturbulences velocity [0.0, 10.0]
        me (jnp.float32): metallicity [-1.0, 0.0]
        a_Mn (jnp.float32): magnesium abundance [-3.0, 3.0]
        a_Fe (jnp.float32): iron abundance [-3.0, 3.0]
        a_Si (jnp.float32): silicon abundance [-3.0, 3.0]
        a_Ca (jnp.float32): calcium abundance [-3.0, 3.0]
        a_C (jnp.float32): carbon abundance [-3.0, 3.0]
        a_N (jnp.float32): nitrogen abundance [-3.0, 3.0]
        a_O (jnp.float32): oxygen abundance [-3.0, 3.0]
        a_Hg (jnp.float32): mercury abundance [-3.0, 3.0]

    Returns:
        tuple: values rescaled to (0-1)
    """
    min_p = [3.845098, 3.5, 0.0, -1.0, -3.0,-3.0,-3.0,-3.0,-3.0,-3.0,-3.0,-3.0]
    max_p = [3.929419, 5.0, 10.0, 0.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]
    log_teff = (log_teff - min_p[0])/(max_p[0] - min_p[0])
    logg = (logg - min_p[1])/(max_p[1] - min_p[1])
    vmic = (vmic - min_p[2])/(max_p[2] - min_p[2])
    me = (me - min_p[3])/(max_p[3] - min_p[3])
    a_Mn = (a_Mn - min_p[4])/(max_p[4] - min_p[4])
    a_Fe = (a_Fe - min_p[5])/(max_p[5] - min_p[5])
    a_Si = (a_Si - min_p[6])/(max_p[6] - min_p[6])
    a_Ca = (a_Ca - min_p[7])/(max_p[7] - min_p[7])
    a_C = (a_C - min_p[8])/(max_p[8] - min_p[8])
    a_N = (a_N - min_p[9])/(max_p[9] - min_p[9])
    a_O = (a_O - min_p[10])/(max_p[10] - min_p[10])
    a_Hg = (a_Hg - min_p[11])/(max_p[11] - min_p[11])
       
    return log_teff, logg, vmic, me, a_Mn, a_Fe, a_Si, a_Ca, a_C, a_N, a_O, a_Hg


@partial(jit, static_argnums=(5,))
def generate_spectrum_overabundance_params(log_teff: jnp.float32,
                                           logg: jnp.float32,
                                           vmic: jnp.float32,
                                           me: jnp.float32,
                                           abundance: jnp.float32,
                                           element: str,) -> jnp.array:
    """Generate spectrum parameters for assumed spectrum model and single element overabundance

    Args:
        log_teff (jnp.float32): logarythmic effective temperature [3.845098, 3.929419]
        logg (jnp.float32): log g [3.5, 5.0]
        vmic (jnp.float32): microturbulences velocity [0.0, 10.0]
        me (jnp.float32): metallicity [-1.0, 0.0]
        abundance (jnp.float32): element's abundance [-3.0, 3.0]
        element (str): element symbol ('Mn', 'Fe', 'Si', 'Ca', 'C', 'N', 'O', 'Hg')

    Returns:
        jnp.array: _description_
    """
    if element not in OVERABUNDANCES:
        raise ValueError(f'Element symbol must be one of {str(OVERABUNDANCES)}.')
    else:
        element_index: int = indexOf(OVERABUNDANCES, element)
    
    input_params = jnp.zeros((len(OVERABUNDANCES,)))

    return jnp.array(scale_spectra_parameters(log_teff, logg, vmic, me, *input_params.at[element_index].set(abundance)))


generate_spectrum_overabundance_params_vec = vmap(generate_spectrum_overabundance_params, in_axes=(None, None, None, None, 0, None))


def frequency_encoding(x, min_period, max_period, dimension):
    periods = jnp.logspace(jnp.log10(min_period), jnp.log10(max_period), num=dimension)
    
    y = jnp.sin(2*jnp.pi/periods*x)
    return y

class SpectrumMLP(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, p, w):
        enc_w = frequency_encoding(w, min_period=1e-7, max_period=0.05, dimension=64)
        x = jnp.hstack([p, enc_w])
        for feat in self.features[:-1]:
            x = nn.gelu(nn.Dense(feat)(x))
        x = 1.0-nn.sigmoid(nn.Dense(self.features[-1])(x))
        return x
    
architecture = tuple([512, 512, 512, 1])
model = SpectrumMLP(architecture)
params = model.init(random.PRNGKey(0), jnp.ones(12,), jnp.ones(1,))


with open("model_weights/SpectrumMLP_wave_DI.bin",'br') as f:
    bin_data = f.read()
    loaded_params = {"params":from_bytes(params["params"], bin_data)}
params = loaded_params

predict_spectrum = jit(vmap(
    lambda spectrum_parameters, log_wavelengths : model.apply(params, spectrum_parameters, log_wavelengths), 
             in_axes=(None, 0), out_axes=0
                                    )
                          )

predict_spectra = jit(vmap(predict_spectrum, in_axes=(0, None), out_axes=0))

def predict_spectrum_with_rot_velocity(params: jnp.array,
                                       log_wave: jnp.array,
                                       rot_vel: jnp.float32) -> jnp.array:
    return predict_spectrum(params, log_wave+jnp.log10(1+rot_vel/c))

predict_spectra_with_rot_velocity = jit(vmap(predict_spectrum_with_rot_velocity, in_axes=(0, None, 0)))
