#!/usr/bin/env python
"""
Use this script to create a wheel with Model Optimizer code:

$ python setup.py sdist
$ python -m pip wheel --no-index --no-deps --wheel-dir dist dist/*.tar.gz
"""

import sys
import os
import re
from setuptools import setup, find_packages
from setuptools.command.install import install

package_name = 'mo'

# Detect all the framework specific requirements_*.txt files.
requirements_txt = []
py_modules = []
for name in os.listdir():
    if re.match('requirements_(.*)\.txt', name):
        requirements_txt.append(name)
    if re.match('mo_(.*)\.py', name):
        py_modules.append(name.split('.')[0])

# Minimal set of dependencies
deps = [
    'networkx>=1.11',
    'defusedxml>=0.5.0',
    'numpy>=1.14.0',
]


class InstallCmd(install):
    def run(self):
        install.run(self)
        # Create requirements.txt files for all the frameworks
        for name in requirements_txt:
            path = os.path.join(self.install_purelib, package_name, name)
            with open(path, 'wt') as f:
                f.write('\n'.join(deps))
        
        path = os.path.join(self.install_purelib, package_name, '__init__.py')
        with open(path, 'wt') as f:
            f.write('import os, sys\n')
            f.write('from {} import mo\n'.format(package_name))
            # This is required to fix internal imports
            f.write('sys.path.append(os.path.dirname(__file__))\n')
            # We install a package into custom folder "package_name".
            # Redirect import to model-optimizer/mo/__init__.py
            f.write('sys.modules["mo"] = mo')


packages = find_packages()
packages = [package_name + '.' + p for p in packages]

setup(name='openvino-mo',
      author='Intel',
      url='https://github.com/openvinotoolkit/openvino',
      packages=packages,
      package_dir={package_name: '.'},
      py_modules=py_modules,
      cmdclass={
          'install': InstallCmd,
      },
      install_requires=deps,
      include_package_data=True,
)
