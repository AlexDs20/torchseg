#!/usr/bin/env python3

import os
from setuptools import setup

directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(directory, 'README.md'), encoding='utf-8') as f:
  long_description = f.read()

setup(name='segmentation',
      version='0.0.1',
      description='A library for semantic segmentation',
      author='Alexandre De Spiegeleer',
      license='MIT',
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages = ['segmentation'],
      install_requires=['numpy', 'torch', 'pytorch-lightning', 'pyyaml', 'torchmetrics'])

