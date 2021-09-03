from setuptools import setup
import os
import sys

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'RDME.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
  name = 'TRA',
  packages = ['TRA'],
  version = '2.15',
  license='MIT',
  description = 'Trust region algorithms for non-linear optimisation.',
  long_description_content_type='text/markdown',
  long_description = long_description,
  author = 'Joel Hampton',
  author_email = 'joelelihampton@outlook.com',
  url = 'https://github.com/Joel-H-dot/TRA',
  keywords = ['non_linear optimisation'],
  install_requires=[
          'numpy',
      ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Science/Research ',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
  ],
)