#!/usr/bin/env python3

# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Use this script to create a wheel with the telemetry library code:
$ python setup.py sdist bdist_wheel
"""

import sys
import os
import re
from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.build_py import build_py


class BuildCmd(build_py):
    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        return [
            (pkg, module, filename)
            for (pkg, module, filename) in modules
            if not filename.endswith('_test.py')
        ]

packages = ['openvino.telemetry', 'openvino.telemetry.backend', 'openvino.telemetry.utils']

setup(name='openvino-telemetry',
      version='0.0.0',
      author='Intel Corporation',
      author_email='openvino_pushbot@intel.com',
      url='https://github.com/openvinotoolkit/openvino',
      packages=packages,
      py_modules=[],
      cmdclass={
          'build_py': BuildCmd,
      },
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
      ],
      install_requires=['requests>=2.20.0'],
)
