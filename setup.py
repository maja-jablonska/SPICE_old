from setuptools import setup, find_packages

setup(
    name='spectrum_integrator',
    version='0.1.0',
    packages=find_packages(include=['spectrumintegrator', 'spectrumintegrator.*']),
    install_requires=[
        'jax>=0.3.10',
        'flax>=0.4.2'
    ]
)