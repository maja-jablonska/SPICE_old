# SPICE

SPICE (SPectrum Integration Compiled Engine) is a Flax module created for fast, differentiable spectra simulation.

The main use case for now is for chemically peculiar stars with overabundances, which change the spectra during the star's rotation. We are planning to add e.g. pulsations in the future.

![Example for iron overabundances](https://github.com/maja-jablonska/SpectrumIntegrator/blob/master/example_img/fe_lines.gif)

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

Many abundance maps can also be described by a spherical harmonic function. To generate an abundance map with dimensions [x_length, y_length] for a spherical harmonic, use the following function. Mind you that the resulting values are rescaled to the (0, 1) range.

#### spherical_harmonic
- **m**: _jnp.float32_ - m mode
- **n**: _jnp.float64_ - n mode
- **x_length**: _jnp.float64_ - length of the x coordinates dimension
- **y_length**: _jnp.float64_ - length of the y coordinates dimension

## Example

To use the SpectrumIntegrator, provide a spectra model first - this can of course be the product of spectra_prediction_function. (You can write your custom one, though!)

```python
ps: Callable[[jnp.array, jnp.array], jnp.array] = spectra_prediction_function(wave_min=6064.5, wave_max=6067.0, wave_points=100)
```

Then use it in the SpectrumIntegrator constructor. We are going to model the spectra for iron and add a spherical harmonic as the abundance map.

```python
si: SpectrumIntegrator = SpectrumIntegrator(interpolation_dims=(50, 50),
                                            meshgrid_dims=(128, 256),
                                            predict_spectra_fn=ps,
                                            rotation=jnp.float32(45.),
                                            inclination=jnp.pi/2,
                                            teff=jnp.float32(7469.74),
                                            logg=jnp.float32(3.95),
                                            vmic=jnp.float32(1.41),
                                            metallicity=jnp.float32(-0.4),
                                            element='Fe',
                                            abundance_map=spherical_harmonic(1, 1, 50, 50))
```

(Refer to the flax documentation when in doubt!)

```python
params = si.init(random.PRNGKey(0), jnp.ones((1,)))
```

Simulate the spectrum for rotation phase 0.:
```python
spectrum = si.apply(params, 0.)
```

The result is a single spectrum array:

![](https://github.com/maja-jablonska/SpectrumIntegrator/blob/master/example_img/fe_spectrum.png)

Example for 20 spectra for 20 rotation phases:
```python
spectra = lax.map(lambda p: si.apply(params, p), jnp.linspace(0, 2*jnp.pi, 20))
```

This time the result is 20 spectra, one for each phase.

![](https://github.com/maja-jablonska/SpectrumIntegrator/blob/master/example_img/fe_spectra.png)

### Authors and citations
Maja Jabłońska and Tomasz Różański (2022)

SPICE is built on top of JAX (https://github.com/google/jax) and Flax (https://github.com/google/flax)
