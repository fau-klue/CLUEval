#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
from setuptools import find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))

# description
with open(os.path.join(here, 'README.md'), mode='rt', encoding='utf-8') as f:
    long_description = f.read()

# version
version = {}
with open(os.path.join(here, 'clueval', 'version.py'), mode='rt', encoding='utf-8') as f:
    exec(f.read(), version)

# requirements
with open(os.path.join(here, 'requirements.txt'), mode='rt', encoding='utf-8') as f:
    install_requires = f.read().strip().split("\n")


setup(
    name="clueval",
    version=version["__version__"],
    description="Evaluation of span predictions",
    license="GPL-3.0-or-later",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Bao Minh Doan Dang, Stephanie Evert, Philipp Heinrich",
    author_email="baominh.d.dang@fau.de",
    url="https://github.com/fau-klue/CLUEval",
    packages=find_packages(),
    scripts=[
        'bin/cluevaluate',
    ],
    python_requires='>=3.9.0',
    install_requires=install_requires,
    classifiers=[
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Development Status :: 3 - Alpha",
        "Operating System :: Unix",
        "Programming Language :: Python :: 3",
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11'
    ],
)
