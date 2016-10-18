#! /usr/bin/env python
# -*- coding: utf-8 -*-

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name="gaia-kepler",
    author="Daniel Foreman-Mackey",
    author_email="danfm@nyu.edu",
    packages=["gaia_kepler"],
    url="https://github.com/dfm/gaia-kepler",
    license="MIT",
    description="A cross match of the Gaia and Kepler catalog",
    package_data={"": ["LICENSE"]},
    include_package_data=True,
    install_requires=["numpy", "scipy", "pandas", "six", "gaia_tools"],
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
)
