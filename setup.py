#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mine",
    version="0.1",
    description="Mutual information neural estimation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jrestrepo86/biomechanics-mine.git",
    author="Juan F. Restrepo",
    author_email="juan.restrepo@uner.edu.ar",
    license="MIT",
    packages=find_packages(exclude=[]),
    keywords="Mutual-Information Neural-Networks",
    python_requires=">=3.6",
    install_requires=[
        "torch",
        "torch_vision",
        "scikit_learn",
        "tqdm",
        "numpy",
        "matplotlib",
    ],
    test_suite="nose.collector",
    tests_require=["nose", "nose-cover3"],
    zip_safe=False,
)
