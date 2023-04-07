from setuptools import setup

setup(
    name="chemical_potentials",
    version="0.1.0",
    description="Chemical potentials library based the stat mech in McQuarrie.",
    author="Jonathan Li",
    license="MIT License",
    packages=["chemical_potentials"],
    install_requires=[
        "httpstan",
        "numpy",
        "pystan",
    ],
    include_package_data=True,
)