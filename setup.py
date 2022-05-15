from setuptools import setup, find_packages

setup(
    name='spice',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'jax>=0.3.10',
        'flax>=0.4.2'
    ],
    include_package_data=True
)