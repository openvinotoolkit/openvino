#!/usr/bin/env python3

# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Use this script to create a wheel with Model Optimizer code:

$ python setup.py sdist bdist_wheel
"""

import sys
import os
import re
from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.build_py import build_py
from shutil import copyfile

package_name = 'mo'

# Detect all the framework specific requirements_*.txt files.
requirements_txt = []
py_modules = []
for name in os.listdir():
    if re.match('requirements(.*)\.txt', name):
        requirements_txt.append(name)
    if re.match('mo(.*)\.py', name):
        py_modules.append(name.split('.')[0])

# Minimal set of dependencies
deps_whitelist = ('networkx', 'defusedxml', 'numpy')
deps = []
with open('requirements.txt', 'rt') as f:
    for line in f.read().split('\n'):
        if line.startswith(deps_whitelist):
            deps.append(line)


class InstallCmd(install):
    def run(self):
        install.run(self)
        # Create requirements.txt files for all the frameworks
        for name in requirements_txt:
            path = os.path.join(self.install_purelib, package_name, name)
            with open(path, 'wt') as f:
                f.write('\n'.join(deps))
        # Add version.txt if exists
        version_txt = 'version.txt'
        if os.path.exists(version_txt):
            copyfile(os.path.join(version_txt),
                     os.path.join(self.install_purelib, package_name, version_txt))
        
        path = os.path.join(self.install_purelib, package_name, '__init__.py')
        with open(path, 'wt') as f:
            f.write('import os, sys\n')
            f.write('from {} import mo\n'.format(package_name))
            # This is required to fix internal imports
            f.write('sys.path.append(os.path.dirname(__file__))\n')
            # We install a package into custom folder "package_name".
            # Redirect import to model-optimizer/mo/__init__.py
            f.write('sys.modules["mo"] = mo')


class BuildCmd(build_py):
    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        return [
            (pkg, module, filename)
            for (pkg, module, filename) in modules
            if not filename.endswith('_test.py')
        ]


packages = find_packages()
packages = [package_name + '.' + p for p in packages]

setup(name='openvino-mo',
      version='0.0.0',
      author='Intel Corporation',
      author_email='openvino_pushbot@intel.com',
      url='https://github.com/openvinotoolkit/openvino',
      packages=packages,
      package_dir={package_name: '.'},
      py_modules=py_modules,
      cmdclass={
          'install': InstallCmd,
          'build_py': BuildCmd,
      },
      entry_points={
          'console_scripts': [
              'mo = mo.__main__:main',
           ],
      },
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
      ],
      install_requires=deps,
)
