# SpectrumIntegrator

A module created for fast, differentiable spectra simulation.

The main use case for now is for chemically peculiar stars with overabundances, which change the spectra during the star's rotation. We are planning to add e.g. pulsations in the future.

![Example for iron overabundances](https://github.com/maja-jablonska/SpectrumIntegrator/blob/master/fe_lines.gif)

The module is built with jax and flax.

## Usage

### SpectrumIntegrator

A flax module integrating spectrum using the provided stellar disk map (of some element's abundances) and stellar parameters for flux simulation for different rotation phases.

The class parameters are:
- **interpolation_dims:** _Tuple[int, int]_ - dimensions of interpolation grid (points at which the disk will be sampled)
- **meshgrid_dims:** _Tuple[int, int]_ - abundance map dimensions (array dimensions of the matrix representing the abundances)
- **predict_spectra_fn:** _Callable_ - function taking (parameters: jnp.array, rotation_map: jnp.array) as arguments and returning spectrum fluxes.
- **rotation:** _jnp.float32_ - rotation velocity in km/s
- **inclination:** _jnp.float32_ - inclination in [0, pi/2]
- **teff:** _jnp.float32_ - effective temperature in [7000, 8500] K
- **logg:** _jnp.float32_ - log g in [3.5, 5.0]
- **vmic:** _jnp.float32_ - microturbulences velocity in [0., 10.] km/s
- **metallicity:** _jnp.float32_ - metallicity in [-1.0, 0.0]
- **element:** _str_ - element mapped in the abundance map - one of ['Mn', 'Fe', 'Si', 'Ca', 'C', 'N', 'O', 'Hg']
- **abundance_map:** _jnp.array_ - map of dimensions _meshgrid_dims_ with abundance values in [-3.0, 3.0]

### Spectra model

A sample spectra model for wavelengths in [5900, 6300] angstroms is provided.
The following function generates a Callable[[jnp.array, jnp.array], jnp.array] (required by the SpectrumIntegrator module) which simulates spectra in range [wave_min, wave_max] for many points on the stellar disk.

#### spectra_prediction_function
- **wave_min**: _jnp.float32_ - minimum wavelength in angstroms
- **wave_max**: _jnp.float32_ - maximum wavelength in angstroms
- **wave_points**: _jnp.float32_ - number of points in the wavelength range

The resulting function can be passed to the SpectrumIntegrator constructor.

### Spherical harmonics generator

Many abundance maps can also be described by a spherical harmonic function. To generate an abundance map with dimensions [x_length, y_length] for a spherical harmonic, use the following function.

#### spherical_harmonic
- **m**: _jnp.float32_ - m mode
- **n**: _jnp.float64_ - n mode
- **x_length**: _jnp.float64_ - length of the x coordinates dimension
- **y_length**: _jnp.float64_ - length of the y coordinates dimension
